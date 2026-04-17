import re
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils


CHANNEL_TO_AXIS = {
    "Xrotation": "x",
    "Yrotation": "y",
    "Zrotation": "z",
}


@dataclass
class SomaAnim:
    quats: np.ndarray
    pos: np.ndarray
    offsets: np.ndarray
    parents: np.ndarray
    bones: list[str]
    frame_time: float


def detect_soma_bvh(bvh_file):
    """Best-effort detection for SOMA-style BVH files."""
    with open(bvh_file, "r") as f:
        header = "".join(f.readlines()[:256])

    if "ROOT Root" in header and "JOINT Hips" in header:
        return True
    if "LeftHandThumb1" in header or "RightHandThumb1" in header:
        return True
    if "LeftToeBase" in header or "RightToeBase" in header:
        return True
    return False


def _parse_soma_bvh(filename):
    bones = []
    parents = []
    offsets = []
    channel_specs = []
    active = -1

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    frame_time = 1 / 30
    positions = None
    rotations = None

    while i < len(lines):
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if stripped in {"HIERARCHY", "MOTION", "{"}:
            i += 1
            continue

        if stripped == "}":
            if active != -1:
                active = parents[active]
            i += 1
            continue

        root_match = re.match(r"\s*ROOT\s+([^\s]+)", line)
        joint_match = re.match(r"\s*JOINT\s+([^\s]+)", line)
        if root_match or joint_match:
            name = (root_match or joint_match).group(1)
            bones.append(name)
            parents.append(active)
            offsets.append([0.0, 0.0, 0.0])
            channel_specs.append([])
            active = len(bones) - 1
            i += 1
            continue

        if "End Site" in stripped:
            # SOMA files of interest use explicit joints for terminal links.
            # Skip end sites entirely because they carry no motion channels.
            depth = 0
            i += 1
            while i < len(lines):
                if "{" in lines[i]:
                    depth += 1
                if "}" in lines[i]:
                    depth -= 1
                    if depth <= 0:
                        break
                i += 1
            i += 1
            continue

        offset_match = re.match(
            r"\s*OFFSET\s+([\-\d\.eE]+)\s+([\-\d\.eE]+)\s+([\-\d\.eE]+)", line
        )
        if offset_match:
            offsets[active] = [float(v) for v in offset_match.groups()]
            i += 1
            continue

        channel_match = re.match(r"\s*CHANNELS\s+(\d+)\s+(.+)", line)
        if channel_match:
            count = int(channel_match.group(1))
            channels = channel_match.group(2).split()
            if len(channels) != count:
                raise ValueError(
                    f"Invalid channel declaration for {bones[active]}: expected {count}, got {len(channels)}"
                )
            channel_specs[active] = channels
            i += 1
            continue

        frames_match = re.match(r"\s*Frames:\s+(\d+)", line)
        if frames_match:
            num_frames = int(frames_match.group(1))
            positions = np.asarray(offsets, dtype=np.float64)[np.newaxis].repeat(num_frames, axis=0)
            rotations = np.zeros((num_frames, len(bones), 3), dtype=np.float64)
            i += 1
            continue

        frame_time_match = re.match(r"\s*Frame Time:\s+([\d\.eE\-]+)", line)
        if frame_time_match:
            frame_time = float(frame_time_match.group(1))
            i += 1
            frame_idx = 0
            while i < len(lines):
                data_line = lines[i].strip()
                i += 1
                if not data_line:
                    continue
                values = np.fromstring(data_line, sep=" ", dtype=np.float64)
                cursor = 0
                for joint_idx, channels in enumerate(channel_specs):
                    if not channels:
                        continue
                    local_pos = positions[frame_idx, joint_idx].copy()
                    local_rot = rotations[frame_idx, joint_idx].copy()
                    rot_channel_idx = 0
                    for channel in channels:
                        value = values[cursor]
                        cursor += 1
                        if channel == "Xposition":
                            local_pos[0] = value
                        elif channel == "Yposition":
                            local_pos[1] = value
                        elif channel == "Zposition":
                            local_pos[2] = value
                        elif channel in CHANNEL_TO_AXIS:
                            local_rot[rot_channel_idx] = value
                            rot_channel_idx += 1
                        else:
                            raise ValueError(f"Unsupported BVH channel: {channel}")
                    positions[frame_idx, joint_idx] = local_pos
                    rotations[frame_idx, joint_idx] = local_rot
                if cursor != len(values):
                    raise ValueError(
                        f"Frame {frame_idx} value count mismatch: consumed {cursor}, got {len(values)}"
                    )
                frame_idx += 1
            break

        i += 1

    if positions is None or rotations is None:
        raise ValueError(f"Failed to parse BVH motion data from {filename}")

    orders = []
    for channels in channel_specs:
        rotation_channels = [c for c in channels if c in CHANNEL_TO_AXIS]
        if rotation_channels:
            order = "".join(CHANNEL_TO_AXIS[c] for c in rotation_channels)
        else:
            order = "zyx"
        orders.append(order)

    quats = np.zeros((rotations.shape[0], rotations.shape[1], 4), dtype=np.float64)
    quats[..., 0] = 1.0
    for joint_idx, order in enumerate(orders):
        if channel_specs[joint_idx]:
            quats[:, joint_idx] = utils.euler_to_quat(
                np.radians(rotations[:, joint_idx]), order=order
            )

    quats = utils.remove_quat_discontinuities(quats)
    return SomaAnim(
        quats=quats,
        pos=positions,
        offsets=np.asarray(offsets, dtype=np.float64),
        parents=np.asarray(parents, dtype=int),
        bones=bones,
        frame_time=frame_time,
    )


def _compute_binding_rotation(data, global_data=None):
    """Compute the SOMA rig's binding rotation purely from skeleton offsets.

    SOMA rigs encode the skeleton with X as the primary bone direction,
    while the LAFAN1 convention (and the IK configs tuned for it) assumes Y.
    This function recovers the rotation that maps the SOMA Hips offset frame
    to the standard BVH frame by comparing offset directions of the Hips
    children with their expected anatomical directions in standard convention.

    The computation is entirely geometric — it uses only the skeleton offsets
    and never reads frame data, so the result is correct regardless of whether
    frame 0 is a T-pose or an arbitrary motion frame.
    """
    if "Hips" not in data.bones:
        return np.array([1.0, 0.0, 0.0, 0.0])

    hips_idx = data.bones.index("Hips")
    children = [i for i, p in enumerate(data.parents) if p == hips_idx]

    # Quick check: if the spine offset already points along Y,
    # no binding correction is needed (standard BVH convention).
    spine_child = None
    left_leg_child = None
    right_leg_child = None
    for c in children:
        name = data.bones[c]
        if "Spine" in name:
            spine_child = c
            spine_off = data.offsets[c]
            spine_len = np.linalg.norm(spine_off)
            if spine_len > 1e-6 and abs(spine_off[1] / spine_len) > 0.9:
                return np.array([1.0, 0.0, 0.0, 0.0])
        elif name in ("LeftLeg", "LeftUpLeg"):
            left_leg_child = c
        elif name in ("RightLeg", "RightUpLeg"):
            right_leg_child = c

    if spine_child is None:
        return np.array([1.0, 0.0, 0.0, 0.0])

    # Compute binding rotation geometrically from offset directions.
    #
    # In SOMA offset space, bones run along X (spine offset ≈ +X).
    # In standard BVH (LAFAN1), the spine offset points +Y (up).
    # The binding rotation maps the SOMA offset frame to the standard frame.
    #
    # Primary constraint: spine offset direction → +Y
    spine_dir = data.offsets[spine_child].copy()
    spine_dir /= np.linalg.norm(spine_dir)
    spine_dst = np.array([0.0, 1.0, 0.0])

    src_dirs = [spine_dir]
    dst_dirs = [spine_dst]

    # Secondary constraint: left-right leg spread direction → +X
    # In SOMA the lateral (left-right) axis is Z; in standard BVH it is X.
    if left_leg_child is not None and right_leg_child is not None:
        lr_vec = data.offsets[left_leg_child] - data.offsets[right_leg_child]
        lr_len = np.linalg.norm(lr_vec)
        if lr_len > 1e-6:
            src_dirs.append(lr_vec / lr_len)
            dst_dirs.append(np.array([1.0, 0.0, 0.0]))

    r_binding, _ = R.align_vectors(np.array(dst_dirs), np.array(src_dirs))
    return r_binding.as_quat(scalar_first=True)


# SOMA uses different bone names from LAFAN1 for the legs.
# "LeftLeg"/"RightLeg" in SOMA = upper leg (LAFAN1 "LeftUpLeg"/"RightUpLeg").
# "LeftShin"/"RightShin" in SOMA = lower leg (LAFAN1 "LeftLeg"/"RightLeg").
_SOMA_BONE_RENAME = {
    # Legs: SOMA upper/lower leg names differ from LAFAN1.
    "LeftLeg": "LeftUpLeg",
    "LeftShin": "LeftLeg",
    "RightLeg": "RightUpLeg",
    "RightShin": "RightLeg",
    # Spine: SOMA chain is Spine1→Spine2→Chest; LAFAN1 is Spine→Spine1→Spine2.
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Chest": "Spine2",
    # Neck: SOMA uses Neck1/Neck2; LAFAN1 uses Neck.
    "Neck1": "Neck",
}


def _add_aliases(frame):
    alias_map = {
        "LeftUpLeg": "LeftLeg",
        "RightUpLeg": "RightLeg",
        "LeftLeg": "LeftShin",
        "RightLeg": "RightShin",
        "LeftToe": "LeftToeBase",
        "RightToe": "RightToeBase",
    }

    for alias, original in alias_map.items():
        if alias not in frame and original in frame:
            frame[alias] = frame[original]

    if "LeftFoot" in frame:
        left_toe_key = "LeftToe" if "LeftToe" in frame else "LeftFoot"
        frame["LeftFootMod"] = (frame["LeftFoot"][0], frame[left_toe_key][1])
    if "RightFoot" in frame:
        right_toe_key = "RightToe" if "RightToe" in frame else "RightFoot"
        frame["RightFootMod"] = (frame["RightFoot"][0], frame[right_toe_key][1])

    return frame


def _estimate_human_height(frames):
    heights = []
    for frame in frames:
        if "Head" not in frame:
            continue
        foot_candidates = []
        for key in ("LeftFootMod", "RightFootMod", "LeftFoot", "RightFoot"):
            if key in frame:
                foot_candidates.append(frame[key][0][2])
        if not foot_candidates:
            continue
        heights.append(frame["Head"][0][2] - min(foot_candidates))

    if not heights:
        return 1.75

    # Use a robust high percentile because staircase and crouched motions can compress
    # the instantaneous height estimate substantially.
    height = float(np.percentile(heights, 95))
    if not 1.2 <= height <= 2.3:
        return 1.75
    return height


def load_soma_bvh_file(bvh_file):
    """
    Parse a SOMA-style BVH and return the same output contract as load_lafan1_file.
    """
    data = _parse_soma_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    # The SOMA rig reference USD declares upAxis = "Y". Map BVH Y-up coordinates
    # into the robot-facing Z-up convention: target = [x, -z, y].
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    frames = []
    for frame_idx in range(data.pos.shape[0]):
        frame = {}
        for bone_idx, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame_idx, bone_idx])
            position = global_data[1][frame_idx, bone_idx] @ rotation_matrix.T / 100.0
            output_name = _SOMA_BONE_RENAME.get(bone, bone)
            frame[output_name] = (position, orientation)
        frames.append(_add_aliases(frame))

    human_height = _estimate_human_height(frames) if frames else 1.75
    motion_fps = int(round(1.0 / data.frame_time)) if data.frame_time > 0 else 30

    return frames, human_height, motion_fps
