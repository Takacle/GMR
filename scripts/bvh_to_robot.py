import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_lafan1_file
from general_motion_retargeting.utils.soma import detect_soma_bvh, load_soma_bvh_file
from rich import print
from tqdm import tqdm
import os
import numpy as np


def load_bvh_motion(bvh_file, bvh_format):
    if bvh_format == "lafan1":
        frames, human_height = load_lafan1_file(bvh_file)
        return frames, human_height, 30, "lafan1"

    if bvh_format == "soma":
        frames, human_height, motion_fps = load_soma_bvh_file(bvh_file)
        return frames, human_height, motion_fps, "soma"

    if bvh_format == "auto":
        if detect_soma_bvh(bvh_file):
            frames, human_height, motion_fps = load_soma_bvh_file(bvh_file)
            return frames, human_height, motion_fps, "soma"
        frames, human_height = load_lafan1_file(bvh_file)
        return frames, human_height, 30, "lafan1"

    raise ValueError(f"Unsupported bvh_format: {bvh_format}")


def update_fps_stats(fps_counter, fps_start_time, fps_display_interval):
    fps_counter += 1
    current_time = time.time()
    if current_time - fps_start_time >= fps_display_interval:
        actual_fps = fps_counter / (current_time - fps_start_time)
        print(f"Actual rendering FPS: {actual_fps:.2f}")
        return 0, current_time
    return fps_counter, fps_start_time


def clone_human_motion_data(human_motion_data):
    return {
        body_name: (pos.copy(), rot.copy())
        for body_name, (pos, rot) in human_motion_data.items()
    }


def viewer_is_running(robot_motion_viewer):
    viewer = getattr(robot_motion_viewer, "viewer", None)
    is_running = getattr(viewer, "is_running", None)
    if callable(is_running):
        return is_running()
    return True


if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "v11"],
        default="unitree_g1",
    )

    parser.add_argument(
        "--bvh_format",
        choices=["auto", "lafan1", "soma"],
        default="auto",
        help="BVH parser to use. 'auto' detects SOMA-style BVH and otherwise falls back to the legacy LAFAN1 parser.",
    )
        
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="Loop playback in the viewer. If used with --save_path, the motion is still saved only once.",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    
    args = parser.parse_args()
    

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
    qpos_list = []
    human_motion_cache = [] if args.loop else None

    
    bvh_data_frames, actual_human_height, motion_fps, detected_format = load_bvh_motion(
        args.bvh_file, args.bvh_format
    )
    print(f"Using BVH parser: {detected_format}")
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human="bvh",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(bvh_data_frames), desc="Retargeting")

    for bvh_frame in bvh_data_frames:
        if not viewer_is_running(robot_motion_viewer):
            break

        fps_counter, fps_start_time = update_fps_stats(
            fps_counter, fps_start_time, fps_display_interval
        )
        pbar.update(1)

        qpos = retargeter.retarget(bvh_frame)
        human_motion_data = retargeter.scaled_human_data

        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=human_motion_data,
            rate_limit=args.rate_limit,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        if args.save_path is not None or args.loop:
            qpos_list.append(qpos.copy())

        if args.loop:
            human_motion_cache.append(clone_human_motion_data(human_motion_data))
    
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()

    if args.loop and qpos_list and viewer_is_running(robot_motion_viewer):
        print("Loop playback enabled. Close the viewer window to stop.")
        while viewer_is_running(robot_motion_viewer):
            for qpos, human_motion_data in zip(qpos_list, human_motion_cache):
                if not viewer_is_running(robot_motion_viewer):
                    break
                fps_counter, fps_start_time = update_fps_stats(
                    fps_counter, fps_start_time, fps_display_interval
                )
                robot_motion_viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=human_motion_data,
                    rate_limit=args.rate_limit,
                )
    
    robot_motion_viewer.close()
       
