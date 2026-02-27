"""
Convert a LeHome extracted episode JSON into a LeRobot dataset.

Example:
uv run examples/lehome/convert_episode_json_to_lerobot.py \
  --json-path ../lehome-challenge/Datasets/episode_200.json \
  --repo-name your_hf_username/lehome_robot
"""

import json
from pathlib import Path
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
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


def main(
    json_path: str,
    *,
    repo_name: str = "your_hf_username/lehome_robot",
    source_root: str = "..",
    push_to_hub: bool = False,
    overwrite: bool = True,
):
    json_path_obj = Path(json_path).resolve()
    source_root_obj = Path(source_root).resolve()

    if not json_path_obj.exists():
        raise FileNotFoundError(f"JSON not found: {json_path_obj}")

    rows = json.loads(json_path_obj.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError(f"Expected non-empty list in JSON: {json_path_obj}")

    first = rows[0]
    state_dim = len(first["observation.state"])
    action_dim = len(first["action"])
    prompt = str(first["prompt"])

    sample_top = _load_rgb_image(_resolve_path(first["observation.image.top_rgb"], root=source_root_obj))
    sample_left = _load_rgb_image(_resolve_path(first["observation.image.left_rgb"], root=source_root_obj))
    sample_right = _load_rgb_image(_resolve_path(first["observation.image.right_rgb"], root=source_root_obj))
    if sample_top.shape != sample_left.shape or sample_top.shape != sample_right.shape:
        raise ValueError(
            "Expected top/left/right image shapes to match. "
            f"Got top={sample_top.shape}, left={sample_left.shape}, right={sample_right.shape}"
        )

    fps = _infer_fps(rows)
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
            "task": str(row.get("prompt", prompt)),
        }
        dataset.add_frame(frame)
    dataset.save_episode()

    print(f"Saved LeRobot dataset to: {output_path}")
    print(f"repo_id={repo_name}, num_frames={len(rows)}, fps={fps}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["lehome", "garment", "so101"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

