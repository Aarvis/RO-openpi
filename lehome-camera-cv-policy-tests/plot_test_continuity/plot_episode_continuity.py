#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import as_frames
from common import continuous_quaternions
from common import load_json
from common import parse_model_type
from common import pose16_to_pose8_pair
from common import pose8_quat_to_rpy
from common import pose8_quat_wxyz
from common import prepare_import_path
from common import resolve_data_path
from common import resolve_default_policy_data_paths
from common import write_mp4_from_rgb_frames


JOINT_NAMES = (
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
)


def _plot_observation_pose16(
    *,
    frame_idx: np.ndarray,
    obs_pose16: np.ndarray,
    quat_order: str,
    out_xyz_png: Path,
    out_orientation_png: Path,
    dpi: int,
    fig_w: float,
    fig_h: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib") from exc

    left_pose8 = []
    right_pose8 = []
    for row in obs_pose16:
        left, right = pose16_to_pose8_pair(row)
        left_pose8.append(left)
        right_pose8.append(right)

    left_pose8_arr = np.asarray(left_pose8, dtype=np.float64)
    right_pose8_arr = np.asarray(right_pose8, dtype=np.float64)

    left_xyz = left_pose8_arr[:, :3]
    right_xyz = right_pose8_arr[:, :3]
    left_quat = continuous_quaternions(
        np.asarray([pose8_quat_wxyz(p, quat_order=quat_order) for p in left_pose8_arr], dtype=np.float64)
    )
    right_quat = continuous_quaternions(
        np.asarray([pose8_quat_wxyz(p, quat_order=quat_order) for p in right_pose8_arr], dtype=np.float64)
    )
    left_rpy = np.asarray(
        [pose8_quat_to_rpy(p, quat_order=quat_order, orientation_unit="deg") for p in left_pose8_arr],
        dtype=np.float64,
    )
    right_rpy = np.asarray(
        [pose8_quat_to_rpy(p, quat_order=quat_order, orientation_unit="deg") for p in right_pose8_arr],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)
    colors_xyz = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}
    axes[0].plot(frame_idx, left_xyz[:, 0], color=colors_xyz["x"], linewidth=1.2, label="x")
    axes[0].plot(frame_idx, left_xyz[:, 1], color=colors_xyz["y"], linewidth=1.2, label="y")
    axes[0].plot(frame_idx, left_xyz[:, 2], color=colors_xyz["z"], linewidth=1.2, label="z")
    axes[0].set_ylabel("Left EE (m)")
    axes[0].set_title("Observation Pose16 FK in Camera Frame (XYZ)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(frame_idx, right_xyz[:, 0], color=colors_xyz["x"], linewidth=1.2, label="x")
    axes[1].plot(frame_idx, right_xyz[:, 1], color=colors_xyz["y"], linewidth=1.2, label="y")
    axes[1].plot(frame_idx, right_xyz[:, 2], color=colors_xyz["z"], linewidth=1.2, label="z")
    axes[1].set_ylabel("Right EE (m)")
    axes[1].set_xlabel("frame_index")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()
    out_xyz_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_xyz_png, dpi=dpi)
    plt.close(fig)

    fig2, axes2 = plt.subplots(4, 1, figsize=(fig_w, fig_h * 1.8), sharex=True)
    quat_colors = {"w": "#d62728", "x": "#2ca02c", "y": "#1f77b4", "z": "#9467bd"}
    axes2[0].plot(frame_idx, left_quat[:, 0], color=quat_colors["w"], label="w")
    axes2[0].plot(frame_idx, left_quat[:, 1], color=quat_colors["x"], label="x")
    axes2[0].plot(frame_idx, left_quat[:, 2], color=quat_colors["y"], label="y")
    axes2[0].plot(frame_idx, left_quat[:, 3], color=quat_colors["z"], label="z")
    axes2[0].set_ylabel("Left quat")
    axes2[0].set_title("Observation Pose16 FK Orientation in Camera Frame")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend(loc="best")

    axes2[1].plot(frame_idx, right_quat[:, 0], color=quat_colors["w"], label="w")
    axes2[1].plot(frame_idx, right_quat[:, 1], color=quat_colors["x"], label="x")
    axes2[1].plot(frame_idx, right_quat[:, 2], color=quat_colors["y"], label="y")
    axes2[1].plot(frame_idx, right_quat[:, 3], color=quat_colors["z"], label="z")
    axes2[1].set_ylabel("Right quat")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend(loc="best")

    rpy_colors = {"roll": "#d62728", "pitch": "#2ca02c", "yaw": "#1f77b4"}
    axes2[2].plot(frame_idx, left_rpy[:, 0], color=rpy_colors["roll"], label="roll")
    axes2[2].plot(frame_idx, left_rpy[:, 1], color=rpy_colors["pitch"], label="pitch")
    axes2[2].plot(frame_idx, left_rpy[:, 2], color=rpy_colors["yaw"], label="yaw")
    axes2[2].set_ylabel("Left RPY (deg)")
    axes2[2].grid(True, alpha=0.3)
    axes2[2].legend(loc="best")

    axes2[3].plot(frame_idx, right_rpy[:, 0], color=rpy_colors["roll"], label="roll")
    axes2[3].plot(frame_idx, right_rpy[:, 1], color=rpy_colors["pitch"], label="pitch")
    axes2[3].plot(frame_idx, right_rpy[:, 2], color=rpy_colors["yaw"], label="yaw")
    axes2[3].set_ylabel("Right RPY (deg)")
    axes2[3].set_xlabel("frame_index")
    axes2[3].grid(True, alpha=0.3)
    axes2[3].legend(loc="best")

    fig2.tight_layout()
    out_orientation_png.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out_orientation_png, dpi=dpi)
    plt.close(fig2)


def _plot_joint_roundtrip(
    *,
    frame_idx: np.ndarray,
    action_original12: np.ndarray,
    action_roundtrip12: np.ndarray,
    out_png: Path,
    dpi: int,
    fig_w: float,
    row_h: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib") from exc

    fig, axes = plt.subplots(6, 2, figsize=(fig_w, row_h * 6), sharex=True)
    left_color = "#1f77b4"
    right_color = "#d62728"
    for joint_idx in range(6):
        left_ax = axes[joint_idx, 0]
        right_ax = axes[joint_idx, 1]

        left_ax.plot(
            frame_idx,
            action_original12[:, joint_idx],
            color=left_color,
            linewidth=1.2,
            label="orig",
        )
        left_ax.plot(
            frame_idx,
            action_roundtrip12[:, joint_idx],
            color=left_color,
            linewidth=1.2,
            linestyle="--",
            label="roundtrip",
        )
        left_ax.set_ylabel(JOINT_NAMES[joint_idx])
        left_ax.grid(True, alpha=0.3)
        if joint_idx == 0:
            left_ax.set_title("Left Arm Joint Angles Across Time")
            left_ax.legend(loc="best")

        right_dim = joint_idx + 6
        right_ax.plot(
            frame_idx,
            action_original12[:, right_dim],
            color=right_color,
            linewidth=1.2,
            label="orig",
        )
        right_ax.plot(
            frame_idx,
            action_roundtrip12[:, right_dim],
            color=right_color,
            linewidth=1.2,
            linestyle="--",
            label="roundtrip",
        )
        right_ax.set_ylabel(JOINT_NAMES[right_dim])
        right_ax.grid(True, alpha=0.3)
        if joint_idx == 0:
            right_ax.set_title("Right Arm Joint Angles Across Time")
            right_ax.legend(loc="best")

    axes[-1, 0].set_xlabel("frame_index")
    axes[-1, 1].set_xlabel("frame_index")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    script_path = Path(__file__)
    prepare_import_path(script_path)
    default_fk_json, default_camera_json = resolve_default_policy_data_paths(script_path)
    from openpi.training.config import LeRobotLehomeCameraCVDataConfig

    data_cfg = LeRobotLehomeCameraCVDataConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Run LehomeCameraCVInputs/Outputs on one episode JSON, plot observation pose16 continuity, "
            "plot action roundtrip joint continuity, and stitch top camera video."
        )
    )
    parser.add_argument("--episode-json", type=Path, required=True)
    parser.add_argument("--state-key", type=str, default="observation.state")
    parser.add_argument("--action-key", type=str, default="action")
    parser.add_argument(
        "--top-image-key",
        type=str,
        default="observation.image.top_rgb",
        help="Frame image path key for top camera.",
    )
    parser.add_argument("--fk-json", type=Path, default=default_fk_json)
    parser.add_argument("--camera-json", type=Path, default=default_camera_json)
    parser.add_argument("--state-unit", type=str, choices=["rad", "deg"], default=data_cfg.state_unit)
    parser.add_argument(
        "--dataset-joint-order",
        type=str,
        default=data_cfg.dataset_joint_order_csv,
    )
    parser.add_argument("--pose-quat-order", type=str, choices=["wxyz", "xyzw"], default=data_cfg.pose_quat_order)
    parser.add_argument("--model-type", type=str, default="pi05", help="pi0 | pi05 | pi0_fast")
    parser.add_argument("--damping", type=float, default=data_cfg.damping)
    parser.add_argument("--alpha", type=float, default=data_cfg.alpha)
    parser.add_argument("--line-search-alphas", type=str, default=data_cfg.line_search_alphas_csv)
    parser.add_argument("--fallback-tol-factor", type=float, default=data_cfg.fallback_tol_factor)
    parser.add_argument("--pos-weight", type=float, default=data_cfg.pos_weight)
    parser.add_argument("--rot-weight", type=float, default=data_cfg.rot_weight)
    parser.add_argument("--max-iters", type=int, default=data_cfg.max_iters)
    parser.add_argument("--tol-pos-m", type=float, default=data_cfg.tol_pos_m)
    parser.add_argument("--tol-rot-deg", type=float, default=data_cfg.tol_rot_deg)
    parser.add_argument("--max-step-norm", type=float, default=data_cfg.max_step_norm)
    parser.add_argument("--no-enforce-limits", action="store_true", default=not data_cfg.enforce_limits)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: output/plot_test_continuity/<episode_stem>",
    )
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--fig-w", type=float, default=14.0)
    parser.add_argument("--fig-h", type=float, default=8.0)
    parser.add_argument("--joint-row-h", type=float, default=2.0)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    script_path = Path(__file__)
    prepare_import_path(script_path)

    from openpi.policies.lehome_camera_cv_policy import LehomeCameraCVInputs
    from openpi.policies.lehome_camera_cv_policy import LehomeCameraCVOutputs
    from openpi.training.config import LeRobotLehomeCameraCVDataConfig

    data_cfg = LeRobotLehomeCameraCVDataConfig()

    episode_json = args.episode_json.resolve()
    if not episode_json.exists():
        raise FileNotFoundError(f"Episode JSON not found: {episode_json}")

    fk_json = args.fk_json.resolve()
    camera_json = args.camera_json.resolve()
    if not fk_json.exists():
        raise FileNotFoundError(f"FK json not found: {fk_json}")
    if not camera_json.exists():
        raise FileNotFoundError(f"Camera json not found: {camera_json}")

    if args.out_dir is None:
        out_dir = (script_path.resolve().parent / "output" / episode_json.stem).resolve()
    else:
        out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = as_frames(load_json(episode_json), src=str(episode_json))
    model_type = parse_model_type(args.model_type)

    input_tf = LehomeCameraCVInputs(
        model_type=model_type,
        fk_json_path=str(fk_json),
        camera_config_json_path=str(camera_json),
        state_unit=str(args.state_unit),
        dataset_joint_order_csv=str(args.dataset_joint_order),
        pose_quat_order=str(args.pose_quat_order),
    )
    output_tf = LehomeCameraCVOutputs(
        model_action_dim=int(data_cfg.action_dim),
        output_action_dim=int(data_cfg.output_action_dim),
        fk_json_path=str(fk_json),
        camera_config_json_path=str(camera_json),
        state_unit=str(args.state_unit),
        dataset_joint_order_csv=str(args.dataset_joint_order),
        pose_quat_order=str(args.pose_quat_order),
        damping=float(args.damping),
        alpha=float(args.alpha),
        line_search_alphas_csv=str(args.line_search_alphas),
        fallback_tol_factor=float(args.fallback_tol_factor),
        pos_weight=float(args.pos_weight),
        rot_weight=float(args.rot_weight),
        max_iters=int(args.max_iters),
        tol_pos_m=float(args.tol_pos_m),
        tol_rot_deg=float(args.tol_rot_deg),
        max_step_norm=float(args.max_step_norm),
        enforce_limits=not bool(args.no_enforce_limits),
    )

    dummy_top = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_left = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_right = np.zeros((480, 640, 3), dtype=np.uint8)

    frame_index = []
    obs_pose16 = []
    action_original12 = []
    action_input16 = []
    action_roundtrip12 = []
    top_image_paths: list[Path] = []
    frame_records: list[dict[str, Any]] = []

    for i, frame in enumerate(frames):
        if args.state_key not in frame or args.action_key not in frame:
            raise KeyError(
                f"Frame {i} missing keys: state={args.state_key in frame}, action={args.action_key in frame}"
            )

        q_t = np.asarray(frame[args.state_key], dtype=np.float64).reshape(-1)
        a_t = np.asarray(frame[args.action_key], dtype=np.float64).reshape(-1)
        if q_t.size < 12 or a_t.size < 12:
            raise ValueError(f"Frame {i}: expected >=12D state/action, got {q_t.size}/{a_t.size}")
        q_t = q_t[:12]
        a_t = a_t[:12]

        in_payload = {
            "observation/top_rgb": dummy_top,
            "observation/left_rgb": dummy_left,
            "observation/right_rgb": dummy_right,
            "observation/state": q_t,
            "actions": a_t,
            "prompt": str(frame.get("prompt", "")),
        }
        in_out = input_tf(in_payload)

        obs_state16 = np.asarray(in_out["state"], dtype=np.float64).reshape(-1)
        if obs_state16.size < 16:
            raise ValueError(f"Frame {i}: transformed observation state invalid size {obs_state16.size}")

        act16 = np.asarray(in_out["actions"], dtype=np.float64).reshape(-1)
        if act16.size < 16:
            raise ValueError(f"Frame {i}: transformed action invalid size {act16.size}")

        out_payload = {
            "state": obs_state16,
            "state_joint": np.asarray(in_out["state_joint"], dtype=np.float64),
            "actions": act16[np.newaxis, :],
        }
        out_out = output_tf(out_payload)
        act12_rt = np.asarray(out_out["actions"], dtype=np.float64).reshape(-1)
        if act12_rt.size < 12:
            raise ValueError(f"Frame {i}: roundtrip IK action invalid size {act12_rt.size}")
        act12_rt = act12_rt[:12]

        img_key = args.top_image_key
        if img_key not in frame:
            fallback_key = "observation.image.top_rgb_frame"
            if img_key == "observation.image.top_rgb" and fallback_key in frame:
                img_key = fallback_key
            else:
                raise KeyError(f"Frame {i} missing top image key: {args.top_image_key}")
        top_path = resolve_data_path(str(frame[img_key]), json_path=episode_json)
        if not top_path.exists():
            raise FileNotFoundError(f"Top image path not found for frame {i}: {top_path}")

        cur_frame_index = int(frame.get("frame_index", i))
        frame_index.append(cur_frame_index)
        obs_pose16.append(obs_state16[:16].copy())
        action_original12.append(a_t.copy())
        action_input16.append(act16[:16].copy())
        action_roundtrip12.append(act12_rt.copy())
        top_image_paths.append(top_path)
        frame_records.append(
            {
                "frame_index": cur_frame_index,
                "observation_pose16": obs_state16[:16].astype(float).tolist(),
                "action_input16": act16[:16].astype(float).tolist(),
                "action_roundtrip12": act12_rt.astype(float).tolist(),
                "top_image_path": str(top_path),
            }
        )

    frame_index_arr = np.asarray(frame_index, dtype=np.int64)
    obs_pose16_arr = np.asarray(obs_pose16, dtype=np.float64)
    action_original12_arr = np.asarray(action_original12, dtype=np.float64)
    action_roundtrip12_arr = np.asarray(action_roundtrip12, dtype=np.float64)

    obs_xyz_png = out_dir / "observation_pose16_xyz.png"
    obs_orientation_png = out_dir / "observation_pose16_orientation.png"
    action_joint_png = out_dir / "action_joint_roundtrip.png"
    top_video_mp4 = out_dir / "top_rgb.mp4"
    continuity_json = out_dir / "continuity_data.json"

    _plot_observation_pose16(
        frame_idx=frame_index_arr,
        obs_pose16=obs_pose16_arr,
        quat_order=str(args.pose_quat_order),
        out_xyz_png=obs_xyz_png,
        out_orientation_png=obs_orientation_png,
        dpi=int(args.dpi),
        fig_w=float(args.fig_w),
        fig_h=float(args.fig_h),
    )
    _plot_joint_roundtrip(
        frame_idx=frame_index_arr,
        action_original12=action_original12_arr,
        action_roundtrip12=action_roundtrip12_arr,
        out_png=action_joint_png,
        dpi=int(args.dpi),
        fig_w=float(args.fig_w),
        row_h=float(args.joint_row_h),
    )
    frames_written = write_mp4_from_rgb_frames(top_image_paths, top_video_mp4, fps=int(args.fps))

    continuity_json.write_text(
        json.dumps(
            {
                "episode_json": str(episode_json),
                "outputs": {
                    "observation_pose16_xyz_png": str(obs_xyz_png),
                    "observation_pose16_orientation_png": str(obs_orientation_png),
                    "action_joint_roundtrip_png": str(action_joint_png),
                    "top_rgb_video": str(top_video_mp4),
                },
                "video_frames_written": frames_written,
                "frames": frame_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved observation XYZ plot:        {obs_xyz_png}")
    print(f"Saved observation orientation:     {obs_orientation_png}")
    print(f"Saved action roundtrip plot:       {action_joint_png}")
    print(f"Saved top camera video:            {top_video_mp4}")
    print(f"Saved continuity data JSON:        {continuity_json}")


if __name__ == "__main__":
    main()
