# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Primary Project Guide

Read `AGENTS.md` first.

`AGENTS.md` is the canonical agent-facing architecture guide for this repository and is written to be shared across Claude, Codex, and similar agents.

## Claude-Specific Working Notes

- Start with `README.md`, then `AGENTS.md`, then the task-specific entry-point script.
- For retargeting behavior questions, read `general_motion_retargeting/params.py` and `general_motion_retargeting/motion_retarget.py` before changing anything.
- Treat `general_motion_retargeting/ik_configs/*.json` as first-class behavior files. Many "logic bugs" are actually config issues.
- Be careful with the current dirty worktree. Do not revert unrelated asset/data changes.
- Script-level robot choices may be narrower than `params.py`. If a robot appears supported in one place but not another, check both.

## Minimal Setup Reminder

This is a Python package for motion retargeting to humanoid robots. Basic development setup:

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
conda install -c conda-forge libstdcxx-ng -y
```

SMPL-X-based workflows also require body model files under `assets/body_models/smplx/`.

## Common Commands

```bash
# SMPL-X to robot
python scripts/smplx_to_robot.py --smplx_file <path> --robot <robot_name> --save_path <output.pkl>

# BVH to robot
python scripts/bvh_to_robot.py --bvh_file <path> --robot <robot_name> --save_path <output.pkl>

# Visualize saved robot motion
python scripts/vis_robot_motion.py --robot <robot_name> --robot_motion_path <path.pkl>
```
