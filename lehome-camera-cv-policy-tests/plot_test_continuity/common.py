from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


def resolve_default_policy_data_paths(script_path: Path) -> tuple[Path, Path]:
    repo_root = script_path.resolve().parents[2]
    policy_data_dir = repo_root / "src" / "openpi" / "policies" / "lehome_camera_cv"
    fk_json = policy_data_dir / "fk_from_usd_common.json"
    camera_json = policy_data_dir / "top_camera_config_runtime_cv.json"
    return fk_json, camera_json


def prepare_import_path(script_path: Path) -> None:
    repo_root = script_path.resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_frames(obj: Any, src: str) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        if not obj:
            raise ValueError(f"Empty episode JSON: {src}")
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("frames"), list):
        frames = obj["frames"]
        if not frames:
            raise ValueError(f"Empty 'frames' in episode JSON: {src}")
        return frames
    raise ValueError(f"Unsupported episode JSON structure: {src}")


def resolve_data_path(path_str: str, *, json_path: Path) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate.resolve()
    rel_to_json = (json_path.parent / candidate).resolve()
    if rel_to_json.exists():
        return rel_to_json
    rel_to_cwd = (Path.cwd() / candidate).resolve()
    if rel_to_cwd.exists():
        return rel_to_cwd
    return rel_to_json


def parse_model_type(name: str):
    from openpi.models import model as _model

    lowered = str(name).strip().lower()
    if lowered == "pi0":
        return _model.ModelType.PI0
    if lowered == "pi05":
        return _model.ModelType.PI05
    if lowered in ("pi0_fast", "pi0-fast"):
        return _model.ModelType.PI0_FAST
    raise ValueError(f"Unsupported model_type={name}; expected pi0|pi05|pi0_fast")


def quat_wxyz_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError(f"Expected 4D quaternion, got {q}")
    w, x, y, z = q.tolist()
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError(f"Expected 4D quaternion, got {q}")
    x, y, z, w = q.tolist()
    return quat_wxyz_to_rot(np.asarray([w, x, y, z], dtype=np.float64))


def pose8_quat_to_rpy(pose8: np.ndarray, quat_order: str, orientation_unit: str) -> np.ndarray:
    q = np.asarray(pose8[3:7], dtype=np.float64).reshape(-1)
    if quat_order == "wxyz":
        R = quat_wxyz_to_rot(q)
    elif quat_order == "xyzw":
        R = quat_xyzw_to_rot(q)
    else:
        raise ValueError(f"Unsupported quat order: {quat_order}")

    sy = math.sqrt(float(R[0, 0] ** 2 + R[1, 0] ** 2))
    singular = sy < 1e-8
    if not singular:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = 0.0

    rpy = np.asarray([roll, pitch, yaw], dtype=np.float64)
    if orientation_unit == "deg":
        rpy = np.rad2deg(rpy)
    return rpy


def pose8_quat_wxyz(pose8: np.ndarray, quat_order: str) -> np.ndarray:
    q = np.asarray(pose8[3:7], dtype=np.float64).reshape(-1)
    if quat_order == "wxyz":
        return q.copy()
    if quat_order == "xyzw":
        x, y, z, w = q.tolist()
        return np.asarray([w, x, y, z], dtype=np.float64)
    raise ValueError(f"Unsupported quat order: {quat_order}")


def continuous_quaternions(q_wxyz_seq: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz_seq, dtype=np.float64).copy()
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected quaternion sequence (N,4), got {q.shape}")
    if q.shape[0] == 0:
        return q
    for i in range(1, q.shape[0]):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] = -q[i]
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return q / norms


def pose16_to_pose8_pair(pose16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose16, dtype=np.float64).reshape(-1)
    if pose.size < 16:
        raise ValueError(f"Expected at least 16D pose, got {pose.size}")
    return pose[:8].copy(), pose[8:16].copy()


def load_rgb_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_mp4_from_rgb_frames(frame_paths: list[Path], out_path: Path, fps: int) -> int:
    if not frame_paths:
        raise ValueError("No frame paths provided for video writing.")

    first = load_rgb_image(frame_paths[0])
    height, width = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    written = 0
    try:
        for frame_path in frame_paths:
            frame_rgb = load_rgb_image(frame_path)
            if frame_rgb.shape[:2] != (height, width):
                frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            written += 1
    finally:
        writer.release()

    return written

