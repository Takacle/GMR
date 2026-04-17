# AGENTS.md

This file is the primary agent-facing guide for this repository. It is intended to be usable by Claude, Codex, and similar coding agents.

## Project Summary

GMR is a Python package for retargeting human motion to humanoid robots.

- Main purpose: convert human motion data into robot root pose and joint positions.
- Main inputs: SMPL-X, BVH, GVHMR output, OptiTrack-derived FBX offline data.
- Main outputs: retargeted robot motion sequences, visualization, and dataset conversion artifacts.
- This repository is not a training framework. Its center of gravity is retargeting, kinematics, motion conversion, and visualization.

## Architecture Overview

The codebase is organized around a narrow core and a broad resource layer.

### Core library

- `general_motion_retargeting/motion_retarget.py`
  - Defines `GeneralMotionRetargeting`, the main retargeting engine.
  - Loads MuJoCo robot XML and the human-to-robot IK config.
  - Builds Mink frame tasks from JSON config.
  - Applies scaling, offsets, optional ground correction, then solves IK iteratively.
  - This is the first file to read when behavior is wrong.

- `general_motion_retargeting/params.py`
  - Central registry for robot XML paths, IK config paths, robot base body names, and viewer camera defaults.
  - If a robot or format is "supported", it is usually because it is registered here.
  - Adding a robot almost always requires editing this file.

- `general_motion_retargeting/kinematics_model.py`
  - Lightweight XML-based forward kinematics model used during dataset export.
  - Parses MuJoCo XML directly, builds joint/body topology, and computes body positions/rotations from `qpos`.
  - Important for batch conversion scripts that need `local_body_pos`.

- `general_motion_retargeting/robot_motion_viewer.py`
  - MuJoCo passive viewer wrapper for replaying robot motion and optionally drawing human target frames.
  - Also handles offscreen video writing with `imageio`.

- `general_motion_retargeting/data_loader.py`
  - Loads saved robot motion `.pkl` files for playback.

- `general_motion_retargeting/utils/`
  - Format-specific loading and preprocessing utilities.
  - `smpl.py`: SMPL-X loading and frame extraction.
  - `lafan1.py`: BVH/LAFAN1 loading.
  - `smpl.py` and `lafan1.py` are the actual input adapters behind the scripts.

### Resource/configuration layer

- `general_motion_retargeting/ik_configs/`
  - JSON files describing how human frames map onto robot frames.
  - These files are the main behavior-control layer for retargeting quality.
  - Most robot support work lives here rather than in Python logic.

- `assets/`
  - Robot MuJoCo XML, URDFs, meshes, and body-model assets.
  - Each robot usually has its own subdirectory.
  - `assets/body_models/smplx/` must contain SMPL-X model files for SMPL-X-based workflows.

### Entry-point layer

- `scripts/`
  - CLI wrappers around the core library.
  - Most scripts do some combination of: load source motion, instantiate `GeneralMotionRetargeting`, optionally view output, optionally save output.

## Data Flow

The common path is:

1. Load human motion frames from an input-specific loader.
2. Convert each frame into a dict of `human_body_name -> (position, quaternion)`.
3. Create `GeneralMotionRetargeting(src_human=..., tgt_robot=...)`.
4. Use IK config from `general_motion_retargeting/ik_configs/`.
5. Solve frame-wise IK to get MuJoCo `qpos`.
6. Split `qpos` into robot root translation, root quaternion, and DoF positions.
7. Optionally run visualization or save dataset outputs.

Saved robot motion data is typically a pickle dict containing keys like:

- `fps`
- `root_pos`
- `root_rot`
- `dof_pos`
- `local_body_pos`
- `link_body_list`

## Directory Map

- `general_motion_retargeting/`
  - Core package code. Most real logic changes should happen here.

- `scripts/`
  - Operational CLIs. Prefer editing these only for argument behavior, workflow glue, or format-specific handling.

- `assets/`
  - Robot and body-model resources. Large and mostly data-oriented. Avoid broad edits without understanding downstream XML references.

- `general_motion_retargeting/ik_configs/`
  - JSON retargeting configs. Small changes here can materially change retargeting behavior.

- `third_party/poselib/`
  - Vendored dependency code. Avoid modifying unless the change truly belongs there.

- `smplx/`
  - Bundled upstream package tree. Treat as external/vendor code unless a task explicitly targets it.

- `md/`
  - Ad hoc notes and reports, not currently the canonical agent-doc source.

- `ACCAD/`
  - Large local motion data tree currently present in the working directory.
  - Treat as data, not source code.

## Main CLI Entry Points

Use these first before inventing new workflows.

- `scripts/smplx_to_robot.py`
  - Single-file SMPL-X to robot retargeting with visualization.

- `scripts/smplx_to_robot_dataset.py`
  - Batch SMPL-X conversion pipeline.
  - Also computes FK-derived body positions for exported motion pickles.
  - Heavier script; includes memory-pressure handling.

- `scripts/bvh_to_robot.py`
  - Single-file BVH/LAFAN1 retargeting.

- `scripts/bvh_to_robot_dataset.py`
  - Batch BVH conversion.

- `scripts/gvhmr_to_robot.py`
  - Consumes GVHMR monocular-video pose output and retargets to a robot.

- `scripts/fbx_offline_to_robot.py`
  - Retargets offline OptiTrack FBX-derived pickle data.

- `scripts/optitrack_to_robot.py`
  - Real-time or streaming-oriented OptiTrack path.
  - Read carefully before changing; it is operationally different from offline conversion.

- `scripts/vis_robot_motion.py`
  - Replays previously exported robot motion `.pkl`.

- `scripts/vis_robot_urdf.py`
  - Isaac Gym-based utility, not part of the normal MuJoCo path.
  - Has extra dependency expectations and should be treated as a standalone tool.

## Supported Formats and Registration Model

Support is not inferred automatically.

- Robot support is declared in `ROBOT_XML_DICT`, `ROBOT_BASE_DICT`, and `VIEWER_CAM_DISTANCE_DICT` inside `general_motion_retargeting/params.py`.
- Input-format-to-config support is declared in `IK_CONFIG_DICT` inside the same file.
- Script-level `argparse` choices may lag behind `params.py`.
  - If a robot exists in `params.py` but not in a script's `choices=[...]`, the script still needs updating.

When adding support, check all three layers:

1. Asset files exist under `assets/<robot>/`
2. `params.py` registers the robot/config
3. relevant script arguments expose the robot

## How To Add A New Robot

The usual path is:

1. Add robot assets under `assets/<robot_name>/`
   - MuJoCo XML is mandatory for the core pipeline.
   - URDF and meshes are usually needed for completeness and debugging.
2. Register XML/base/viewer settings in `general_motion_retargeting/params.py`.
3. Add at least one IK config JSON in `general_motion_retargeting/ik_configs/`.
4. Update script `argparse` choices where necessary.
5. Validate with a single-motion script before touching dataset pipelines.

If retargeting quality is poor, assume the problem is in IK config and offsets before assuming the solver is wrong.

## How To Add A New Human Motion Format

The usual path is:

1. Implement or adapt a loader that produces frame dicts of body name to `(position, quaternion)`.
2. Add a new `src_human` branch in `IK_CONFIG_DICT`.
3. Create per-robot JSON configs for the new format.
4. Add or update a `scripts/*.py` entry point.

Keep the core `GeneralMotionRetargeting` API stable if possible. New formats should usually be adapters, not solver rewrites.

## Dependencies And Runtime Assumptions

From `setup.py`, the key runtime dependencies are:

- `mink`
- `mujoco`
- `numpy`
- `scipy`
- `qpsolvers[proxqp]`
- `opencv-python`
- `smplx`
- `imageio[ffmpeg]`
- `torch` is used in dataset/export paths even though the package metadata is incomplete for fully describing that stack

Operational assumptions:

- Python 3.10+
- Ubuntu-oriented setup
- MuJoCo works locally
- SMPL-X body model files are available in `assets/body_models/smplx/`
- Some scripts expect CUDA-capable PyTorch for FK/export workloads

## Known Footguns

- `README.md` notes that some SMPL-X setups require changing `smplx/body_models.py` from `npz` to `pkl` handling. This is a repo-specific operational note, not a clean package abstraction.
- Some scripts contain hard-coded assumptions or one-off paths. `scripts/convert_omomo_to_smplx.py` is the clearest example and should not be treated as a polished public CLI.
- Script support matrices are inconsistent. `params.py` may know about more robots than a given script exposes.
- `scripts/vis_robot_urdf.py` depends on Isaac Gym and is not representative of the rest of the repo.
- Behavior is often controlled more by IK JSON than by Python code. Editing the solver before reading the config is usually wasted effort.
- MuJoCo quaternions in this repo often require careful ordering review. Saved robot motion uses conversions between `xyzw` and `wxyz` in a few places.

## Current Working Tree State

This repository is currently dirty and includes many user-side data/resource changes.

Observed high-risk areas include:

- `ACCAD/` large added dataset content
- `assets/v11/` added or modified robot assets
- `general_motion_retargeting/ik_configs/smplx_to_v11.json`
- `general_motion_retargeting/params.py`
- `scripts/smplx_to_robot.py`
- `scripts/bvh_to_robot.py`

Agent rule:

- Do not revert unrelated user changes.
- Assume local asset/data edits are intentional unless the task explicitly says to clean them up.
- Read `git status` before making changes that touch assets, configs, or scripts.

## Recommended Reading Order For Agents

When starting work, read in this order:

1. `README.md`
2. `AGENTS.md`
3. `general_motion_retargeting/params.py`
4. `general_motion_retargeting/motion_retarget.py`
5. The specific `scripts/*.py` entry point related to the task
6. Relevant file in `general_motion_retargeting/ik_configs/`

If the task is about exported motion structure, also read:

7. `general_motion_retargeting/data_loader.py`
8. `general_motion_retargeting/kinematics_model.py`

## Recommended Modification Strategy

- For retargeting quality issues:
  - inspect IK JSON first
  - inspect per-format loader second
  - inspect solver behavior third

- For "new robot" work:
  - assets and `params.py` first
  - script exposure second
  - config tuning third

- For "batch export is broken":
  - focus on dataset scripts plus `kinematics_model.py`

- For "viewer is broken":
  - focus on `robot_motion_viewer.py`, quaternion conventions, and saved motion schema

## Claude Compatibility

Claude should read this file first, then read `CLAUDE.md` for the smaller set of Claude-specific instructions.
