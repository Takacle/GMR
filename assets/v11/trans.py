import os
import torch
import numpy as np
import pinocchio as pin
import pink
import smplx
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pinocchio.visualize import MeshcatVisualizer
import meshcat

# ================= 配置路径 =================
URDF_PATH = "assets/v11/urdf/v11.urdf"
MESH_DIR = "assets/v11"
SMPLX_MODEL_PATH = "/home/user2/GMR/" # 注意：这里填目录路径，不要填到 .npz 文件名
AMASS_NPZ_PATH = "/home/user2/GMR/ACCAD/Male2General_c3d/A1-_Stand_stageii.npz"

# 坐标转换：AMASS(Y-up) -> Robot(Z-up)
def to_robot(p):
    return np.array([p[1], -p[2], p[0]])

def init_smplx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path 指向包含模型文件的目录
    model = smplx.create(SMPLX_MODEL_PATH, model_type='smplx', gender='neutral').to(device)
    return model, device

def init_robot():
    robot = pin.RobotWrapper.BuildFromURDF(..., root_joint=pin.JointModelFreeFlyer())
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    
    # === 新增：读取 JSON 配置 ===
    with open("smplx_to_v11.json", "r") as f:
        config_data = json.load(f)
    
    match_table = config_data["ik_match_table1"]
    
    tasks = {
        "posture": PostureTask(cost=1e-3),
    }
    tasks["posture"].set_target(robot.q0)

    # 动态创建任务
    # 注意：这里需要你把 smpl_data 的键名也对应起来，或者在 get_smpl_joints 里修改
    # 这里只展示如何利用 JSON 里的 offset
    for robot_link, params in match_table.items():
        smpl_name, pos_weight, rot_weight, pos_off, rot_off = params
        
        # 将列表转为 numpy 数组 / pin.Quaternion
        # 注意 JSON 里的四元数顺序通常是 [x, y, z, w]
        offset_quat = pin.Quaternion(np.array(rot_off)).matrix()
        
        # 你可以将这个 offset_quat 存到一个字典里，
        # 在主循环 update 时： set_target(smpl_rot @ offset_quat)
        
        # 创建任务
        task = FrameTask(robot_link, position_cost=pos_weight, orientation_cost=rot_weight)
        tasks[robot_link] = task

    return robot, configuration, tasks

def get_smpl_joints(smpl_model, device, poses, trans, frame_idx, name_to_idx):
    # 提取当前帧参数
    body_pose = torch.from_numpy(poses[frame_idx:frame_idx+1, 3:66]).float().to(device)
    global_orient = torch.from_numpy(poses[frame_idx:frame_idx+1, :3]).float().to(device)
    transl = torch.from_numpy(trans[frame_idx:frame_idx+1]).float().to(device)
    
    output = smpl_model(body_pose=body_pose, global_orient=global_orient, transl=transl)
    joints = output.joints.detach().cpu().numpy()[0]
    
    return {
        "pelvis_p": joints[name_to_idx['pelvis']],
        "l_hand_p": joints[name_to_idx['left_wrist']],
        "r_hand_p": joints[name_to_idx['right_wrist']]
    }

def run_retargeting():
    smpl_model, device = init_smplx()
    robot, config, tasks = init_robot()
    
    # 手动定义索引
    name_to_idx = {
        'pelvis': 0, 'left_wrist': 20, 'right_wrist': 21
    }

    bdata = np.load(AMASS_NPZ_PATH)
    poses, trans = bdata['poses'], bdata['trans']
    
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    scale = 0.85 
    dt = 0.02 

    for f in range(poses.shape[0]):
        smpl_data = get_smpl_joints(smpl_model, device, poses, trans, f, name_to_idx)
        
        # 应用转换和缩放
        p_pelvis = to_robot(smpl_data["pelvis_p"]) * scale
        p_l_hand = to_robot(smpl_data["l_hand_p"]) * scale
        p_r_hand = to_robot(smpl_data["r_hand_p"]) * scale

        # 设置目标。注意：旋转设为单位阵，因为 orientation_cost 已设为 0
        tasks["pelvis"].set_target(pin.SE3(np.eye(3), p_pelvis))
        tasks["l_hand"].set_target(pin.SE3(np.eye(3), p_l_hand))
        tasks["r_hand"].set_target(pin.SE3(np.eye(3), p_r_hand))

        # 求解
        velocity = solve_ik(config, tasks.values(), dt, solver="quadprog")
        config.update(config.integrate(velocity, dt))
        
        viz.display(config.q)

if __name__ == "__main__":
    run_retargeting()