"""
Compute the correct quaternion offsets for bvh_to_v11.json by comparing
BVH skeleton T-pose FK with V11 robot T-pose FK.

Formula (from motion_retarget.py:275):
    target_quat = R_bvh * R_config
For perfect T-pose alignment:
    R_config = inv(R_bvh_tpose) * R_robot_tpose
"""

import json
import sys
import pathlib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

# Add project root to path
HERE = pathlib.Path(__file__).parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from general_motion_retargeting.utils.soma import load_soma_bvh_file


# World rotation: 90° around Z to align BVH world frame (+X lateral) with V11 (+Y lateral)
WORLD_ROTATION = R.from_euler('z', 90, degrees=True)
WORLD_ROTATION_QUAT = WORLD_ROTATION.as_quat(scalar_first=True)

# Joint mapping: V11 link name -> BVH joint name
JOINT_MAPPING = {
    "base_link": "Hips",
    "waist_pitch_link": "Spine2",
    "head_yaw_link": "Head",
    "left_hip_roll_link": "LeftUpLeg",
    "left_knee_link": "LeftLeg",
    "left_ankle_roll_link": "LeftFootMod",
    "right_hip_roll_link": "RightUpLeg",
    "right_knee_link": "RightLeg",
    "right_ankle_roll_link": "RightFootMod",
    "left_shoulder_roll_link": "LeftArm",
    "left_elbow_link": "LeftForeArm",
    "left_wrist_yaw_link": "LeftHand",
    "right_shoulder_roll_link": "RightArm",
    "right_elbow_link": "RightForeArm",
    "right_wrist_yaw_link": "RightHand",
}


def get_bvh_tpose_quats(bvh_file):
    """Load BVH and return frame 0 world-frame quaternions (scalar-first),
    with world_rotation applied to align BVH frame with robot frame."""
    frames, _, _ = load_soma_bvh_file(bvh_file)
    frame0 = frames[0]
    result = {}
    for joint_name, (pos, quat) in frame0.items():
        # Apply world rotation to BVH orientation
        rotated_quat = (WORLD_ROTATION * R.from_quat(quat, scalar_first=True)).as_quat(scalar_first=True)
        result[joint_name] = rotated_quat
    return result


def get_v11_tpose_quats(xml_path):
    """Load V11 in mujoco, set T-pose, return link world quaternions (scalar-first)."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Set T-pose: shoulder_roll = ±π/2
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name == "left_shoulder_roll_joint":
            # Find qpos address for this joint
            qpos_addr = model.jnt_qposadr[i]
            data.qpos[qpos_addr] = np.pi / 2
        elif joint_name == "right_shoulder_roll_joint":
            qpos_addr = model.jnt_qposadr[i]
            data.qpos[qpos_addr] = -np.pi / 2

    # Set floating base upright (quat w,x,y,z = 1,0,0,0)
    data.qpos[0:3] = [0, 0, 1.0]  # position: standing height
    data.qpos[3:7] = [1, 0, 0, 0]  # orientation: identity

    mujoco.mj_forward(model, data)

    result = {}
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name:
            # xquat is (w, x, y, z) - scalar first
            result[body_name] = data.xquat[i].copy()
    return result


def compute_config_quaternions(bvh_quats, robot_quats, mapping):
    """
    Compute R_config = inv(R_bvh) * R_robot for each mapped pair.
    All quaternions in scalar-first format.
    """
    results = {}
    for robot_link, bvh_joint in mapping.items():
        if bvh_joint not in bvh_quats:
            print(f"WARNING: BVH joint '{bvh_joint}' not found, skipping {robot_link}")
            continue
        if robot_link not in robot_quats:
            print(f"WARNING: Robot link '{robot_link}' not found, skipping")
            continue

        R_bvh = R.from_quat(bvh_quats[bvh_joint], scalar_first=True)
        R_robot = R.from_quat(robot_quats[robot_link], scalar_first=True)

        R_config = R_bvh.inv() * R_robot
        config_quat = R_config.as_quat(scalar_first=True)

        results[robot_link] = config_quat

        # Debug output
        print(f"\n{robot_link} <-> {bvh_joint}:")
        print(f"  BVH quat:    [{', '.join(f'{v:.8f}' for v in bvh_quats[bvh_joint])}]")
        print(f"  Robot quat:  [{', '.join(f'{v:.8f}' for v in robot_quats[robot_link])}]")
        print(f"  Config quat: [{', '.join(f'{v:.8f}' for v in config_quat)}]")

    return results


def format_quat_for_json(quat, precision=8):
    """Format quaternion as JSON-compatible list."""
    return [round(float(v), precision) for v in quat]


def update_config(config_path, computed_quats):
    """Update the bvh_to_v11.json config with computed quaternions."""
    with open(config_path) as f:
        config = json.load(f)

    for table_name in ["ik_match_table1", "ik_match_table2"]:
        table = config[table_name]
        for robot_link, entry in table.items():
            if robot_link in computed_quats:
                old_quat = entry[4]
                new_quat = format_quat_for_json(computed_quats[robot_link])
                entry[4] = new_quat
                print(f"\n{table_name}/{robot_link}:")
                print(f"  OLD: {old_quat}")
                print(f"  NEW: {new_quat}")

    return config


if __name__ == "__main__":
    bvh_file = str(PROJECT_ROOT / "seed_dataset_path_placeholder")
    # Use the actual soma_base_skel_minimal.bvh
    bvh_file = "/home/user2/seed_dataset/soma_shapes/soma_base_rig/soma_base_skel_minimal.bvh"
    xml_path = PROJECT_ROOT / "assets" / "v11" / "v11.xml"
    config_path = PROJECT_ROOT / "general_motion_retargeting" / "ik_configs" / "bvh_to_v11.json"

    print("=" * 60)
    print("Step 1: Loading BVH T-pose quaternions")
    print("=" * 60)
    bvh_quats = get_bvh_tpose_quats(bvh_file)
    print(f"Loaded {len(bvh_quats)} BVH joints")
    for name, q in sorted(bvh_quats.items()):
        if name in JOINT_MAPPING.values():
            print(f"  {name}: [{', '.join(f'{v:.6f}' for v in q)}]")

    print("\n" + "=" * 60)
    print("Step 2: Loading V11 T-pose quaternions")
    print("=" * 60)
    robot_quats = get_v11_tpose_quats(xml_path)
    print(f"Loaded {len(robot_quats)} robot bodies")
    for name, q in sorted(robot_quats.items()):
        if name in JOINT_MAPPING:
            print(f"  {name}: [{', '.join(f'{v:.6f}' for v in q)}]")

    print("\n" + "=" * 60)
    print("Step 3: Computing config quaternions")
    print("=" * 60)
    computed_quats = compute_config_quaternions(bvh_quats, robot_quats, JOINT_MAPPING)

    print("\n" + "=" * 60)
    print("Step 4: Generating updated config")
    print("=" * 60)
    updated_config = update_config(config_path, computed_quats)

    # Add world_rotation and init_qpos
    updated_config["world_rotation"] = format_quat_for_json(WORLD_ROTATION_QUAT)
    updated_config["init_qpos"] = {
        "left_shoulder_roll_joint": 1.5708,
        "right_shoulder_roll_joint": -1.5708,
    }

    # Save updated config
    output_path = config_path.parent / "bvh_to_v11_computed.json"
    with open(output_path, "w") as f:
        json.dump(updated_config, f, indent=4)
    print(f"\nSaved updated config to: {output_path}")

    # Also print the final quaternion summary
    print("\n" + "=" * 60)
    print("FINAL QUATERNION SUMMARY (scalar-first: w, x, y, z)")
    print("=" * 60)
    for robot_link, bvh_joint in JOINT_MAPPING.items():
        if robot_link in computed_quats:
            q = computed_quats[robot_link]
            print(f"  {robot_link:30s} -> {bvh_joint:15s}: [{', '.join(f'{v:.8f}' for v in q)}]")
