"""
Convert multiple LeHome extracted episode JSON files into one local LeRobot dataset.

Each input JSON is treated as exactly one episode, so action horizons never cross episodes.

Example:
uv run examples/lehome/convert_episode_json_to_lerobot.py \
  --json-root ../lehome-challenge/Datasets/all_episode_exports \
  --json-glob "**/json/episode_*.json" \
  --repo-name local/lehome_all_episodes
"""

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


def main(
    *,
    json_root: str = "../lehome-challenge/Datasets/all_episode_exports",
    json_glob: str = "**/json/episode_*.json",
    repo_name: str = "local/lehome_all_episodes",
    source_root: str = "..",
    overwrite: bool = True,
    default_fps: int = 30,
):
    json_root_obj = Path(json_root).resolve()
    source_root_obj = Path(source_root).resolve()

    if not json_root_obj.exists():
        raise FileNotFoundError(f"JSON root not found: {json_root_obj}")

    episode_jsons = _find_episode_jsons(json_root_obj, json_glob)
    if not episode_jsons:
        raise FileNotFoundError(f"No JSON files found under {json_root_obj} with glob '{json_glob}'")

    # Initialize dataset schema from first valid JSON.
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
    action_dim = len(first["action"])
    prompt = str(first.get("prompt", "unknown task"))

    sample_top = _load_rgb_image(_resolve_path(first["observation.image.top_rgb"], root=source_root_obj))
    sample_left = _load_rgb_image(_resolve_path(first["observation.image.left_rgb"], root=source_root_obj))
    sample_right = _load_rgb_image(_resolve_path(first["observation.image.right_rgb"], root=source_root_obj))
    if sample_top.shape != sample_left.shape or sample_top.shape != sample_right.shape:
        raise ValueError(
            "Expected top/left/right image shapes to match. "
            f"Got top={sample_top.shape}, left={sample_left.shape}, right={sample_right.shape}"
        )

    fps = _infer_fps(first_rows, default_fps=default_fps)

    # Local output only; nothing is pushed to Hub in this script.
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

    for json_path in tqdm(episode_jsons, desc="Converting JSON episodes", unit="episode"):
        try:
            rows = _load_rows(json_path)
        except Exception as exc:
            stats["episodes_skipped"] += 1
            print(f"[Warn] Skipping invalid JSON {json_path}: {exc}")
            continue

        row0 = rows[0]
        if len(row0["observation.state"]) != state_dim or len(row0["action"]) != action_dim:
            stats["episodes_skipped"] += 1
            print(
                f"[Warn] Skipping {json_path}: shape mismatch "
                f"state_dim={len(row0['observation.state'])} (expected {state_dim}), "
                f"action_dim={len(row0['action'])} (expected {action_dim})"
            )
            continue

        episode_prompt = str(row0.get("prompt", prompt))
        for row in rows:
            frame = {
                "observation.images.top_rgb": _load_rgb_image(
                    _resolve_path(row["observation.image.top_rgb"], root=source_root_obj)
                ),
                "observation.images.left_rgb": _load_rgb_image(
                    _resolve_path(row["observation.image.left_rgb"], root=source_root_obj)
                ),
                "observation.images.right_rgb": _load_rgb_image(
                    _resolve_path(row["observation.image.right_rgb"], root=source_root_obj)
                ),
                "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                "actions": np.asarray(row["action"], dtype=np.float32),
                "task": str(row.get("prompt", episode_prompt)),
            }
            dataset.add_frame(frame)

        # Important: keeps episode boundaries for action horizon/windowing.
        dataset.save_episode()
        stats["episodes_saved"] += 1
        stats["frames_saved"] += len(rows)

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
