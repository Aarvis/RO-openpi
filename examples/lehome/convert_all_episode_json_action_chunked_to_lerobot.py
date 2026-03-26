"""
Convert multiple LeHome extracted episode JSON files with precomputed action chunks
into one local LeRobot dataset.

Each input JSON is treated as exactly one episode. Unlike the standard converter,
each saved frame already contains a full action chunk under `action_chunk`, so the
training data loader must not reconstruct horizons from neighboring timesteps.

Example:
uv run examples/lehome/convert_all_episode_json_action_chunked_to_lerobot.py \
  --json-root ../lehome-challenge/Datasets/sample \
  --json-glob "**/episode_*.json" \
  --repo-name local/lehome_action_chunked_sample \
  --workers 36
"""

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import json
from pathlib import Path
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm
import tyro


def _resolve_path(path_str: str, *, root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _load_rgb_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _infer_fps(rows: list[dict], default_fps: int = 30) -> int:
    timestamps = np.asarray([float(r["timestamp"]) for r in rows], dtype=np.float64)
    if timestamps.size < 2:
        return default_fps
    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 1e-8]
    if deltas.size == 0:
        return default_fps
    median_dt = float(np.median(deltas))
    return int(round(1.0 / median_dt))


def _find_episode_jsons(json_root: Path, json_glob: str) -> list[Path]:
    return sorted(p for p in json_root.glob(json_glob) if p.is_file())


def _load_rows(json_path: Path) -> list[dict]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload:
        return payload
    if isinstance(payload, dict):
        rows = payload.get("frames")
        if isinstance(rows, list) and rows:
            return rows
    raise ValueError(f"Expected non-empty list or dict with 'frames' in JSON: {json_path}")


def _get_first_present(row: dict, *keys: str) -> str:
    for key in keys:
        if key in row:
            return str(row[key])
    raise KeyError(f"None of the required keys were found: {keys}")


def _prepare_episode(
    json_path: Path,
    source_root: Path,
    state_dim: int,
    action_horizon: int,
    action_dim: int,
    default_prompt: str,
) -> dict:
    try:
        rows = _load_rows(json_path)
    except Exception as exc:
        return {
            "ok": False,
            "json_path": str(json_path),
            "error": f"invalid JSON: {exc}",
        }

    row0 = rows[0]
    state_dim_found = len(row0["observation.state"])
    action_chunk_shape = np.asarray(row0["action_chunk"], dtype=np.float32).shape
    if state_dim_found != state_dim or action_chunk_shape != (action_horizon, action_dim):
        return {
            "ok": False,
            "json_path": str(json_path),
            "error": (
                "shape mismatch "
                f"state_dim={state_dim_found} (expected {state_dim}), "
                f"action_chunk_shape={action_chunk_shape} (expected {(action_horizon, action_dim)})"
            ),
        }

    episode_prompt = str(row0.get("prompt", default_prompt))
    frames: list[dict] = []
    for row in rows:
        frames.append(
            {
                "observation.images.top_rgb": _load_rgb_image(
                    _resolve_path(
                        _get_first_present(row, "observation.image.top_rgb", "observation.image.top_rgb_frame"),
                        root=source_root,
                    )
                ),
                "observation.images.left_rgb": _load_rgb_image(
                    _resolve_path(
                        _get_first_present(row, "observation.image.left_rgb", "observation.image.left_rgb_frame"),
                        root=source_root,
                    )
                ),
                "observation.images.right_rgb": _load_rgb_image(
                    _resolve_path(
                        _get_first_present(row, "observation.image.right_rgb", "observation.image.right_rgb_frame"),
                        root=source_root,
                    )
                ),
                "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                "actions": np.asarray(row["action_chunk"], dtype=np.float32),
                "task": str(row.get("prompt", episode_prompt)),
            }
        )

    return {
        "ok": True,
        "json_path": str(json_path),
        "frames": frames,
    }


def main(
    *,
    json_root: str = "../lehome-challenge/Datasets/sample",
    json_glob: str = "**/episode_*.json",
    repo_name: str = "local/lehome_action_chunked_sample",
    source_root: str = "..",
    overwrite: bool = True,
    default_fps: int = 30,
    workers: int = 36,
):
    json_root_obj = Path(json_root).resolve()
    source_root_obj = Path(source_root).resolve()

    if not json_root_obj.exists():
        raise FileNotFoundError(f"JSON root not found: {json_root_obj}")

    episode_jsons = _find_episode_jsons(json_root_obj, json_glob)
    if not episode_jsons:
        raise FileNotFoundError(f"No JSON files found under {json_root_obj} with glob '{json_glob}'")

    first_rows: list[dict] | None = None
    first_json: Path | None = None
    for candidate in episode_jsons:
        try:
            first_rows = _load_rows(candidate)
            first_json = candidate
            break
        except Exception:
            continue

    if first_rows is None or first_json is None:
        raise ValueError("No valid non-empty episode JSON found.")

    first = first_rows[0]
    state_dim = len(first["observation.state"])
    action_chunk = np.asarray(first["action_chunk"], dtype=np.float32)
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action_chunk shape (T,D), got {action_chunk.shape} in {first_json}")
    action_horizon, action_dim = map(int, action_chunk.shape)
    prompt = str(first.get("prompt", "unknown task"))

    sample_top = _load_rgb_image(
        _resolve_path(
            _get_first_present(first, "observation.image.top_rgb", "observation.image.top_rgb_frame"),
            root=source_root_obj,
        )
    )
    sample_left = _load_rgb_image(
        _resolve_path(
            _get_first_present(first, "observation.image.left_rgb", "observation.image.left_rgb_frame"),
            root=source_root_obj,
        )
    )
    sample_right = _load_rgb_image(
        _resolve_path(
            _get_first_present(first, "observation.image.right_rgb", "observation.image.right_rgb_frame"),
            root=source_root_obj,
        )
    )
    if sample_top.shape != sample_left.shape or sample_top.shape != sample_right.shape:
        raise ValueError(
            "Expected top/left/right image shapes to match. "
            f"Got top={sample_top.shape}, left={sample_left.shape}, right={sample_right.shape}"
        )

    fps = _infer_fps(first_rows, default_fps=default_fps)

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="so101",
        fps=fps,
        features={
            "observation.images.top_rgb": {
                "dtype": "image",
                "shape": tuple(sample_top.shape),
                "names": ["height", "width", "channel"],
            },
            "observation.images.left_rgb": {
                "dtype": "image",
                "shape": tuple(sample_left.shape),
                "names": ["height", "width", "channel"],
            },
            "observation.images.right_rgb": {
                "dtype": "image",
                "shape": tuple(sample_right.shape),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_horizon, action_dim),
                "names": ["time", "actions"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=2,
    )

    stats = {
        "episode_jsons_found": len(episode_jsons),
        "episodes_saved": 0,
        "episodes_skipped": 0,
        "frames_saved": 0,
    }

    workers = max(int(workers), 1)
    if workers == 1:
        prepared_iter = (
            _prepare_episode(p, source_root_obj, state_dim, action_horizon, action_dim, prompt) for p in episode_jsons
        )
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        prepared_iter = pool.map(
            _prepare_episode,
            episode_jsons,
            repeat(source_root_obj),
            repeat(state_dim),
            repeat(action_horizon),
            repeat(action_dim),
            repeat(prompt),
        )

    try:
        for prepared in tqdm(prepared_iter, total=len(episode_jsons), desc="Converting JSON episodes", unit="episode"):
            if not prepared["ok"]:
                stats["episodes_skipped"] += 1
                print(f"[Warn] Skipping {prepared['json_path']}: {prepared['error']}")
                continue

            frames = prepared["frames"]
            for frame in frames:
                dataset.add_frame(frame)

            dataset.save_episode()
            stats["episodes_saved"] += 1
            stats["frames_saved"] += len(frames)
    finally:
        if workers > 1:
            pool.shutdown(wait=True)

    print(f"\nSaved local LeRobot dataset to: {output_path}")
    print(f"repo_id={repo_name}, fps={fps}, action_horizon={action_horizon}, action_dim={action_dim}")
    print(
        "stats: "
        f"episode_jsons_found={stats['episode_jsons_found']}, "
        f"episodes_saved={stats['episodes_saved']}, "
        f"episodes_skipped={stats['episodes_skipped']}, "
        f"frames_saved={stats['frames_saved']}"
    )


if __name__ == "__main__":
    tyro.cli(main)
