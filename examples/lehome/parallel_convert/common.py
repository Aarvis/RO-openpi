from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
import numpy as np


def resolve_path(path_str: str, *, root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_rgb_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def infer_fps(rows: list[dict[str, Any]], default_fps: int = 30) -> int:
    timestamps = np.asarray([float(r["timestamp"]) for r in rows], dtype=np.float64)
    if timestamps.size < 2:
        return default_fps
    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 1e-8]
    if deltas.size == 0:
        return default_fps
    median_dt = float(np.median(deltas))
    return int(round(1.0 / median_dt))


def find_episode_jsons(json_root: Path, json_glob: str) -> list[Path]:
    return sorted(path for path in json_root.glob(json_glob) if path.is_file())


def load_rows(json_path: Path) -> list[dict[str, Any]]:
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected non-empty list in JSON: {json_path}")
    return rows


def derive_schema(first_rows: list[dict[str, Any]], source_root: Path) -> dict[str, Any]:
    first = first_rows[0]
    state_dim = len(first["observation.state"])
    action_dim = len(first["action"])
    prompt = str(first.get("prompt", "unknown task"))

    sample_top = load_rgb_image(resolve_path(first["observation.image.top_rgb"], root=source_root))
    sample_left = load_rgb_image(resolve_path(first["observation.image.left_rgb"], root=source_root))
    sample_right = load_rgb_image(resolve_path(first["observation.image.right_rgb"], root=source_root))
    if sample_top.shape != sample_left.shape or sample_top.shape != sample_right.shape:
        raise ValueError(
            "Expected top/left/right image shapes to match. "
            f"Got top={sample_top.shape}, left={sample_left.shape}, right={sample_right.shape}"
        )

    return {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "prompt": prompt,
        "top_shape": tuple(sample_top.shape),
        "left_shape": tuple(sample_left.shape),
        "right_shape": tuple(sample_right.shape),
    }


def local_repo_output_path(repo_name: str) -> Path:
    return HF_LEROBOT_HOME / repo_name


def sanitize_repo_name(repo_name: str) -> str:
    return repo_name.replace("/", "__").replace("\\", "__").replace(":", "_")


def chunk_ordered_items(items: list[Path], num_chunks: int) -> list[list[tuple[int, Path]]]:
    if num_chunks <= 1:
        return [[(i, item) for i, item in enumerate(items)]]

    total = len(items)
    chunk_size = (total + num_chunks - 1) // num_chunks
    chunks: list[list[tuple[int, Path]]] = []
    for start in range(0, total, chunk_size):
        subset = items[start : start + chunk_size]
        chunks.append([(i, item) for i, item in enumerate(subset, start=start)])
    return chunks

