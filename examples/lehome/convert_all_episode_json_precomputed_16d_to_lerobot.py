"""
Convert recursive LeHome episode JSON files with precomputed 16D camera-CV state/actions
and top-camera images into one local LeRobot dataset.

Each input JSON is treated as exactly one episode, so action horizons never cross episodes.
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
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError(f"Expected non-empty list in JSON: {json_path}")
    return rows


def _prepare_episode(
    json_path: Path,
    source_root: Path,
    state_dim: int,
    action_dim: int,
    default_prompt: str,
    state_key: str,
    action_key: str,
    top_image_key: str,
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
    state_dim_found = len(row0[state_key])
    action_dim_found = len(row0[action_key])
    if state_dim_found != state_dim or action_dim_found != action_dim:
        return {
            "ok": False,
            "json_path": str(json_path),
            "error": (
                "shape mismatch "
                f"state_dim={state_dim_found} (expected {state_dim}), "
                f"action_dim={action_dim_found} (expected {action_dim})"
            ),
        }

    episode_prompt = str(row0.get("prompt", default_prompt))
    frames: list[dict] = []
    for row in rows:
        frames.append(
            {
                "observation.images.top_rgb": _load_rgb_image(
                    _resolve_path(row[top_image_key], root=source_root)
                ),
                "observation.state": np.asarray(row[state_key], dtype=np.float32),
                "actions": np.asarray(row[action_key], dtype=np.float32),
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
    json_root: str = "../lehome-challenge/Datasets/all_episode_exports",
    json_glob: str = "**/json/episode_*.json",
    repo_name: str = "local/lehome_precomputed_16d_pretrain",
    source_root: str = "..",
    overwrite: bool = True,
    default_fps: int = 30,
    workers: int = 36,
    state_key: str = "observation.state",
    action_key: str = "action",
    top_image_key: str = "observation.image.top_rgb",
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
    state_dim = len(first[state_key])
    action_dim = len(first[action_key])
    prompt = str(first.get("prompt", "unknown task"))

    sample_top = _load_rgb_image(_resolve_path(first[top_image_key], root=source_root_obj))
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
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
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
            _prepare_episode(
                p,
                source_root_obj,
                state_dim,
                action_dim,
                prompt,
                state_key=state_key,
                action_key=action_key,
                top_image_key=top_image_key,
            )
            for p in episode_jsons
        )
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        prepared_iter = pool.map(
            _prepare_episode,
            episode_jsons,
            repeat(source_root_obj),
            repeat(state_dim),
            repeat(action_dim),
            repeat(prompt),
            repeat(state_key),
            repeat(action_key),
            repeat(top_image_key),
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
    print(f"repo_id={repo_name}, fps={fps}")
    print(
        "stats: "
        f"episode_jsons_found={stats['episode_jsons_found']}, "
        f"episodes_saved={stats['episodes_saved']}, "
        f"episodes_skipped={stats['episodes_skipped']}, "
        f"frames_saved={stats['frames_saved']}"
    )


if __name__ == "__main__":
    tyro.cli(main)
