#!/usr/bin/env python3
"""
Evaluate LeHome camera-CV policy roundtrip using policy transform classes directly.

Per frame:
1) Call LehomeCameraCVInputs on JSON state/action -> observation_input_16d, action_input_16d.
2) Call LehomeCameraCVOutputs on action_input_16d (as model output) + state/state_joint -> action_ik_12d.
3) Call LehomeCameraCVInputs again with action_ik_12d -> action_roundtrip_16d.
4) Compare action_input_16d vs action_roundtrip_16d:
   - EE position error (meters)
   - EE orientation error (degrees)

This validates the exact transform class logic in:
  - openpi.policies.lehome_camera_cv_policy.LehomeCameraCVInputs
  - openpi.policies.lehome_camera_cv_policy.LehomeCameraCVOutputs
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm import tqdm


def _resolve_default_policy_data_paths(script_path: Path) -> tuple[Path, Path]:
    repo_root = script_path.resolve().parents[1]
    policy_data_dir = repo_root / "src" / "openpi" / "policies" / "lehome_camera_cv"
    fk_json = policy_data_dir / "fk_from_usd_common.json"
    camera_json = policy_data_dir / "top_camera_config_runtime_cv.json"
    return fk_json, camera_json


def _prepare_import_path(script_path: Path) -> None:
    repo_root = script_path.resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_frames(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("frames"), list):
        return obj["frames"]
    raise ValueError("Unsupported episode JSON structure: expected list or dict with 'frames'.")


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {}
    return {
        "count": float(arr.size),
        "min": float(np.min(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p98": float(np.percentile(arr, 98)),
        "p99": float(np.percentile(arr, 99)),
        "p99_5": float(np.percentile(arr, 99.5)),
        "p99_9": float(np.percentile(arr, 99.9)),
        "p99_99": float(np.percentile(arr, 99.99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
    }


def _quat_to_xyzw(quat: np.ndarray, quat_order: str) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    if quat_order == "xyzw":
        return q
    if quat_order == "wxyz":
        w, x, y, z = q.tolist()
        return np.asarray([x, y, z, w], dtype=np.float64)
    raise ValueError(f"Unsupported quat order: {quat_order}")


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray, quat_order: str) -> float:
    a = _quat_to_xyzw(q1, quat_order=quat_order)
    b = _quat_to_xyzw(q2, quat_order=quat_order)
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-12 or bn < 1e-12:
        return 0.0
    a = a / an
    b = b / bn
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(math.degrees(2.0 * math.acos(dot)))


def _pose8_errors_deg_m(
    pose8_a: np.ndarray,
    pose8_b: np.ndarray,
    quat_order: str,
) -> tuple[float, float]:
    pa = np.asarray(pose8_a, dtype=np.float64).reshape(8)
    pb = np.asarray(pose8_b, dtype=np.float64).reshape(8)
    pos_err_m = float(np.linalg.norm(pa[:3] - pb[:3]))
    rot_err_deg = _quat_angle_deg(pa[3:7], pb[3:7], quat_order=quat_order)
    return pos_err_m, rot_err_deg


def _find_episode_jsons(episodes_dir: Path, glob_pattern: str, *, recursive: bool) -> list[Path]:
    iterator = episodes_dir.rglob(glob_pattern) if recursive else episodes_dir.glob(glob_pattern)
    return sorted(p for p in iterator if p.is_file())


def _parse_model_type(name: str):
    from openpi.models import model as _model

    name_l = str(name).strip().lower()
    if name_l == "pi0":
        return _model.ModelType.PI0
    if name_l == "pi05":
        return _model.ModelType.PI05
    if name_l in ("pi0_fast", "pi0-fast"):
        return _model.ModelType.PI0_FAST
    raise ValueError(f"Unsupported model_type={name}; expected pi0|pi05|pi0_fast")


def _evaluate_episode_worker(
    episode_path_str: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    # Ensure openpi source path is available inside worker process.
    src_dir = str(cfg["src_dir"])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from openpi.policies.lehome_camera_cv_policy import LehomeCameraCVInputs
    from openpi.policies.lehome_camera_cv_policy import LehomeCameraCVOutputs

    model_type = _parse_model_type(cfg["model_type"])

    input_tf = LehomeCameraCVInputs(
        model_type=model_type,
        fk_json_path=str(cfg["fk_json"]),
        camera_config_json_path=str(cfg["camera_json"]),
        state_unit=str(cfg["state_unit"]),
        dataset_joint_order_csv=str(cfg["dataset_joint_order"]),
        pose_quat_order=str(cfg["pose_quat_order"]),
    )
    output_tf = LehomeCameraCVOutputs(
        model_action_dim=int(cfg["model_action_dim"]),
        output_action_dim=int(cfg["output_action_dim"]),
        fk_json_path=str(cfg["fk_json"]),
        camera_config_json_path=str(cfg["camera_json"]),
        state_unit=str(cfg["state_unit"]),
        dataset_joint_order_csv=str(cfg["dataset_joint_order"]),
        pose_quat_order=str(cfg["pose_quat_order"]),
        damping=float(cfg["damping"]),
        alpha=float(cfg["alpha"]),
        line_search_alphas_csv=str(cfg["line_search_alphas_csv"]),
        fallback_tol_factor=float(cfg["fallback_tol_factor"]),
        pos_weight=float(cfg["pos_weight"]),
        rot_weight=float(cfg["rot_weight"]),
        max_iters=int(cfg["max_iters"]),
        tol_pos_m=float(cfg["tol_pos_m"]),
        tol_rot_deg=float(cfg["tol_rot_deg"]),
        max_step_norm=float(cfg["max_step_norm"]),
        enforce_limits=bool(cfg["enforce_limits"]),
    )

    dummy_top = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_left = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_right = np.zeros((480, 640, 3), dtype=np.uint8)

    ep_path = Path(episode_path_str)
    try:
        frames = _as_frames(_load_json(ep_path))
        ep_pos_all: list[float] = []
        ep_rot_all: list[float] = []
        ep_pos_left: list[float] = []
        ep_pos_right: list[float] = []
        ep_rot_left: list[float] = []
        ep_rot_right: list[float] = []

        state_key = str(cfg["state_key"])
        action_key = str(cfg["action_key"])
        pose_quat_order = str(cfg["pose_quat_order"])

        for i, frame in enumerate(frames):
            if state_key not in frame or action_key not in frame:
                raise KeyError(
                    f"Frame {i} missing keys: state={state_key in frame}, action={action_key in frame}"
                )
            q_t = np.asarray(frame[state_key], dtype=np.float64).reshape(-1)
            a_t = np.asarray(frame[action_key], dtype=np.float64).reshape(-1)
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
            obs_input_16d = np.asarray(in_out["state"], dtype=np.float64).reshape(-1)
            action_input_16d = np.asarray(in_out["actions"], dtype=np.float64).reshape(-1)
            if obs_input_16d.size < 16 or action_input_16d.size < 16:
                raise ValueError(
                    f"Transform output dims invalid at frame {i}: state={obs_input_16d.size}, actions={action_input_16d.size}"
                )

            out_payload = {
                "state": obs_input_16d,
                "state_joint": np.asarray(in_out["state_joint"], dtype=np.float64),
                "actions": action_input_16d[np.newaxis, :],
            }
            out_out = output_tf(out_payload)
            action_ik_12d = np.asarray(out_out["actions"], dtype=np.float64).reshape(-1)
            if action_ik_12d.size < 12:
                raise ValueError(f"Output transform returned invalid action dim at frame {i}: {action_ik_12d.size}")
            action_ik_12d = action_ik_12d[:12]

            rt_payload = {
                "observation/top_rgb": dummy_top,
                "observation/left_rgb": dummy_left,
                "observation/right_rgb": dummy_right,
                "observation/state": q_t,
                "actions": action_ik_12d,
                "prompt": str(frame.get("prompt", "")),
            }
            rt_out = input_tf(rt_payload)
            action_roundtrip_16d = np.asarray(rt_out["actions"], dtype=np.float64).reshape(-1)
            if action_roundtrip_16d.size < 16:
                raise ValueError(
                    f"Roundtrip action transform returned invalid dim at frame {i}: {action_roundtrip_16d.size}"
                )

            lp, lr = _pose8_errors_deg_m(
                action_input_16d[:8], action_roundtrip_16d[:8], quat_order=pose_quat_order
            )
            rp, rr = _pose8_errors_deg_m(
                action_input_16d[8:16], action_roundtrip_16d[8:16], quat_order=pose_quat_order
            )
            ep_pos_left.append(lp)
            ep_rot_left.append(lr)
            ep_pos_right.append(rp)
            ep_rot_right.append(rr)
            ep_pos_all.extend((lp, rp))
            ep_rot_all.extend((lr, rr))

        return {
            "ok": True,
            "episode_json": str(ep_path),
            "frames": int(len(frames)),
            "pos_all": ep_pos_all,
            "pos_left": ep_pos_left,
            "pos_right": ep_pos_right,
            "rot_all": ep_rot_all,
            "rot_left": ep_rot_left,
            "rot_right": ep_rot_right,
            "ee_position_error_m": {
                "all_arms": _summary(ep_pos_all),
                "left": _summary(ep_pos_left),
                "right": _summary(ep_pos_right),
            },
            "ee_orientation_error_deg": {
                "all_arms": _summary(ep_rot_all),
                "left": _summary(ep_rot_left),
                "right": _summary(ep_rot_right),
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "episode_json": str(ep_path),
            "error": str(exc),
        }


def main() -> None:
    script_path = Path(__file__)
    _prepare_import_path(script_path)
    default_fk_json, default_camera_json = _resolve_default_policy_data_paths(script_path)
    from openpi.training.config import LeRobotLehomeCameraCVDataConfig

    data_cfg = LeRobotLehomeCameraCVDataConfig()

    parser = argparse.ArgumentParser(
        description="Evaluate LehomeCameraCVInputs/Outputs class roundtrip error on episode JSONs."
    )
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="episode_*.json")
    parser.add_argument("--recursive", action="store_true", default=False)
    parser.add_argument("--state-key", type=str, default="observation.state")
    parser.add_argument("--action-key", type=str, default="action")
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
    parser.add_argument(
        "--line-search-alphas",
        type=str,
        default=data_cfg.line_search_alphas_csv,
        help="Comma-separated positive line-search multipliers.",
    )
    parser.add_argument("--fallback-tol-factor", type=float, default=data_cfg.fallback_tol_factor)
    parser.add_argument("--pos-weight", type=float, default=data_cfg.pos_weight)
    parser.add_argument("--rot-weight", type=float, default=data_cfg.rot_weight)
    parser.add_argument("--max-iters", type=int, default=data_cfg.max_iters)
    parser.add_argument("--tol-pos-m", type=float, default=data_cfg.tol_pos_m)
    parser.add_argument("--tol-rot-deg", type=float, default=data_cfg.tol_rot_deg)
    parser.add_argument("--max-step-norm", type=float, default=data_cfg.max_step_norm)
    parser.add_argument("--no-enforce-limits", action="store_true", default=not data_cfg.enforce_limits)
    parser.add_argument("--workers", type=int, default=32, help="Episode-parallel workers. Use 1 for serial.")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means all episodes.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("output/lehome_camera_cv_policy_roundtrip_stats.json"),
    )
    parser.add_argument("--fail-fast", action="store_true", default=False)
    args = parser.parse_args()

    episodes_dir = args.episodes_dir.resolve()
    fk_json = args.fk_json.resolve()
    camera_json = args.camera_json.resolve()
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes dir not found: {episodes_dir}")
    if not fk_json.exists():
        raise FileNotFoundError(f"FK json not found: {fk_json}")
    if not camera_json.exists():
        raise FileNotFoundError(f"Camera json not found: {camera_json}")

    episode_paths = _find_episode_jsons(episodes_dir, args.glob, recursive=bool(args.recursive))
    if args.max_episodes and args.max_episodes > 0:
        episode_paths = episode_paths[: args.max_episodes]
    if not episode_paths:
        raise FileNotFoundError(f"No episode json found under {episodes_dir} with glob {args.glob!r}")

    line_search_alphas = tuple(
        float(v.strip()) for v in str(args.line_search_alphas).split(",") if v.strip()
    )
    if not line_search_alphas:
        raise ValueError("line_search_alphas must contain at least one positive value.")
    if any(v <= 0.0 for v in line_search_alphas):
        raise ValueError("line_search_alphas must be positive.")

    workers = max(int(args.workers), 1)
    worker_cfg = {
        "src_dir": str((script_path.resolve().parents[1] / "src").resolve()),
        "state_key": str(args.state_key),
        "action_key": str(args.action_key),
        "fk_json": str(fk_json),
        "camera_json": str(camera_json),
        "state_unit": str(args.state_unit),
        "dataset_joint_order": str(args.dataset_joint_order),
        "pose_quat_order": str(args.pose_quat_order),
        "model_type": str(args.model_type),
        "model_action_dim": int(data_cfg.action_dim),
        "output_action_dim": int(data_cfg.output_action_dim),
        "damping": float(args.damping),
        "alpha": float(args.alpha),
        "line_search_alphas_csv": str(args.line_search_alphas),
        "fallback_tol_factor": float(args.fallback_tol_factor),
        "pos_weight": float(args.pos_weight),
        "rot_weight": float(args.rot_weight),
        "max_iters": int(args.max_iters),
        "tol_pos_m": float(args.tol_pos_m),
        "tol_rot_deg": float(args.tol_rot_deg),
        "max_step_norm": float(args.max_step_norm),
        "enforce_limits": bool(not args.no_enforce_limits),
    }

    pos_err_all: list[float] = []
    rot_err_all: list[float] = []
    pos_err_left: list[float] = []
    pos_err_right: list[float] = []
    rot_err_left: list[float] = []
    rot_err_right: list[float] = []

    episode_records: list[dict[str, Any]] = []
    episodes_failed = 0
    frames_total = 0

    if workers == 1:
        results_iter = (
            _evaluate_episode_worker(str(ep_path), worker_cfg) for ep_path in episode_paths
        )
        for result in tqdm(results_iter, total=len(episode_paths), desc="Evaluating episodes", unit="episode"):
            if result["ok"]:
                frames_total += int(result["frames"])
                pos_err_all.extend(result["pos_all"])
                pos_err_left.extend(result["pos_left"])
                pos_err_right.extend(result["pos_right"])
                rot_err_all.extend(result["rot_all"])
                rot_err_left.extend(result["rot_left"])
                rot_err_right.extend(result["rot_right"])
                episode_records.append(
                    {
                        "episode_json": result["episode_json"],
                        "frames": int(result["frames"]),
                        "ee_position_error_m": result["ee_position_error_m"],
                        "ee_orientation_error_deg": result["ee_orientation_error_deg"],
                    }
                )
            else:
                episodes_failed += 1
                if args.fail_fast:
                    raise RuntimeError(
                        f"Episode failed: {result['episode_json']}\n{result['error']}"
                    )
                episode_records.append(
                    {
                        "episode_json": result["episode_json"],
                        "error": result["error"],
                    }
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_to_path = {
                ex.submit(_evaluate_episode_worker, str(ep_path), worker_cfg): ep_path for ep_path in episode_paths
            }
            for fut in tqdm(as_completed(fut_to_path), total=len(fut_to_path), desc="Evaluating episodes", unit="episode"):
                result = fut.result()
                if result["ok"]:
                    frames_total += int(result["frames"])
                    pos_err_all.extend(result["pos_all"])
                    pos_err_left.extend(result["pos_left"])
                    pos_err_right.extend(result["pos_right"])
                    rot_err_all.extend(result["rot_all"])
                    rot_err_left.extend(result["rot_left"])
                    rot_err_right.extend(result["rot_right"])
                    episode_records.append(
                        {
                            "episode_json": result["episode_json"],
                            "frames": int(result["frames"]),
                            "ee_position_error_m": result["ee_position_error_m"],
                            "ee_orientation_error_deg": result["ee_orientation_error_deg"],
                        }
                    )
                else:
                    episodes_failed += 1
                    if args.fail_fast:
                        for pending in fut_to_path:
                            pending.cancel()
                        raise RuntimeError(
                            f"Episode failed: {result['episode_json']}\n{result['error']}"
                        )
                    episode_records.append(
                        {
                            "episode_json": result["episode_json"],
                            "error": result["error"],
                        }
                    )

    report = {
        "metadata": {
            "episodes_dir": str(episodes_dir),
            "glob": args.glob,
            "recursive": bool(args.recursive),
            "episodes_found": int(len(episode_paths)),
            "episodes_failed": int(episodes_failed),
            "workers": int(workers),
            "frames_processed": int(frames_total),
            "state_key": args.state_key,
            "action_key": args.action_key,
            "fk_json": str(fk_json),
            "camera_json": str(camera_json),
            "state_unit": args.state_unit,
            "dataset_joint_order": args.dataset_joint_order,
            "pose_quat_order": args.pose_quat_order,
            "model_type": args.model_type,
            "policy_transform_classes_tested": [
                "LehomeCameraCVInputs",
                "LehomeCameraCVOutputs",
            ],
            "ik_params": {
                "damping": float(args.damping),
                "alpha": float(args.alpha),
                "line_search_alphas": [float(v) for v in line_search_alphas],
                "fallback_tol_factor": float(max(args.fallback_tol_factor, 1.0)),
                "pos_weight": float(args.pos_weight),
                "rot_weight": float(args.rot_weight),
                "max_iters": int(args.max_iters),
                "tol_pos_m": float(args.tol_pos_m),
                "tol_rot_deg": float(args.tol_rot_deg),
                "max_step_norm": float(args.max_step_norm),
                "enforce_limits": bool(not args.no_enforce_limits),
            },
        },
        "global_summary": {
            "ee_position_error_m": {
                "all_arms": _summary(pos_err_all),
                "left": _summary(pos_err_left),
                "right": _summary(pos_err_right),
            },
            "ee_orientation_error_deg": {
                "all_arms": _summary(rot_err_all),
                "left": _summary(rot_err_left),
                "right": _summary(rot_err_right),
            },
        },
        "episodes": episode_records,
    }

    out_json = args.out_json.resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved stats JSON: {out_json}")
    print(
        "Done: "
        f"episodes={len(episode_paths)}, failed={episodes_failed}, frames_processed={frames_total}, "
        f"arm_frame_samples={len(pos_err_all)}"
    )
    g_pos = report["global_summary"]["ee_position_error_m"]["all_arms"]
    g_rot = report["global_summary"]["ee_orientation_error_deg"]["all_arms"]
    if g_pos and g_rot:
        print(
            "Global all-arms position stats: "
            f"min={g_pos['min']:.6g}, p5={g_pos['p5']:.6g}, median={g_pos['median']:.6g}, "
            f"p99={g_pos['p99']:.6g}, p99.5={g_pos['p99_5']:.6g}, p99.9={g_pos['p99_9']:.6g}, "
            f"max={g_pos['max']:.6g}"
        )
        print(
            "Global all-arms orientation stats: "
            f"min={g_rot['min']:.6g}, p5={g_rot['p5']:.6g}, median={g_rot['median']:.6g}, "
            f"p99={g_rot['p99']:.6g}, p99.5={g_rot['p99_5']:.6g}, p99.9={g_rot['p99_9']:.6g}, "
            f"max={g_rot['max']:.6g}"
        )


if __name__ == "__main__":
    main()
