from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Origami VLA OpenPI train/val manifests.")
    parser.add_argument("--config", type=Path, required=True, help="JSON config path.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_episode_selection(episode_uids: list[str], selection_cfg: dict[str, Any]) -> list[str]:
    requested = [str(uid) for uid in selection_cfg.get("episode_uids", []) if str(uid).strip()]
    if requested:
        missing = [uid for uid in requested if uid not in set(episode_uids)]
        if missing:
            raise RuntimeError(f"Requested episodes not present in split: {missing}")
        episode_uids = [uid for uid in episode_uids if uid in set(requested)]
    start_index = max(0, int(selection_cfg.get("start_index", 0)))
    max_episodes = selection_cfg.get("max_episodes")
    if max_episodes is not None:
        max_episodes = max(0, int(max_episodes))
    episode_uids = episode_uids[start_index:]
    if max_episodes is not None:
        episode_uids = episode_uids[:max_episodes]
    return episode_uids


def ensure_modalities(episode_root: Path, image_modalities: dict[str, str], fail_on_missing: bool) -> bool:
    missing = [str(episode_root / relpath) for relpath in image_modalities.values() if not (episode_root / relpath).exists()]
    if missing and fail_on_missing:
        raise FileNotFoundError(f"Missing required image modalities for {episode_root.name}: {missing}")
    return not missing


def build_split_frame(
    *,
    dataset_root: Path,
    split_name: str,
    episode_uids: list[str],
    view_modes: list[str],
    local_target_parquet_name: str,
    speed_weight_parquet_name: str | None,
    sample_weight_column: str,
    require_speed_weights: bool,
    planner_index_filename: str,
    planner_export_root: Path,
    image_modalities: dict[str, str],
    fail_on_missing_modalities: bool,
    drop_horizon_clipped: bool,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for episode_uid in tqdm(episode_uids, desc=f"Build {split_name} manifest", unit="episode", dynamic_ncols=True):
        episode_root = dataset_root / "episodes" / episode_uid
        if not ensure_modalities(episode_root, image_modalities, fail_on_missing_modalities):
            continue

        target_frame = pd.read_parquet(episode_root / "arrays" / local_target_parquet_name)
        target_frame = target_frame[target_frame["target_valid"]].copy()
        target_frame = target_frame[target_frame["npz_sample_index"] >= 0].copy()
        if drop_horizon_clipped:
            target_frame = target_frame[~target_frame["horizon_clipped_to_episode_end"].astype(bool)].copy()
        if target_frame.empty:
            continue

        base = target_frame[
            [
                "episode_uid",
                "current_frame_position",
                "current_frame_index",
                "current_u",
                "horizon_u",
                "npz_sample_index",
                "num_control_points",
                "num_knots_total",
                "num_dataset_frames_in_segment",
                "horizon_clipped_to_episode_end",
            ]
        ].rename(
            columns={
                "current_frame_position": "frame_position",
                "current_frame_index": "frame_index",
                "npz_sample_index": "local_target_npz_sample_index",
            }
        )

        if speed_weight_parquet_name:
            speed_weight_path = episode_root / "arrays" / speed_weight_parquet_name
            if not speed_weight_path.exists():
                if require_speed_weights:
                    raise FileNotFoundError(
                        f"Speed-weight sidecar missing for {episode_uid}: {speed_weight_path}"
                    )
                base["sample_weight"] = 1.0
            else:
                weight_frame = pd.read_parquet(speed_weight_path)
                if sample_weight_column not in weight_frame.columns:
                    raise KeyError(
                        f"Speed-weight parquet missing column {sample_weight_column!r}: {speed_weight_path}"
                    )
                weight_frame = weight_frame[
                    [
                        "episode_uid",
                        "current_frame_position",
                        "current_frame_index",
                        "npz_sample_index",
                        sample_weight_column,
                    ]
                ].rename(
                    columns={
                        "current_frame_position": "frame_position",
                        "current_frame_index": "frame_index",
                        "npz_sample_index": "local_target_npz_sample_index",
                        sample_weight_column: "sample_weight",
                    }
                )
                base = base.merge(
                    weight_frame,
                    on=["episode_uid", "frame_position", "frame_index", "local_target_npz_sample_index"],
                    how="left",
                    validate="one_to_one",
                )
                if require_speed_weights and base["sample_weight"].isna().any():
                    missing_count = int(base["sample_weight"].isna().sum())
                    raise RuntimeError(
                        f"Missing sample weights for {missing_count} local target rows in {episode_uid} "
                        f"after merging {speed_weight_path}"
                    )
                base["sample_weight"] = base["sample_weight"].fillna(1.0).astype("float32")
        else:
            base["sample_weight"] = 1.0

        for view_mode in view_modes:
            planner_output_dir = planner_export_root / episode_uid / view_mode
            planner_index_path = planner_output_dir / planner_index_filename
            if not planner_index_path.exists():
                raise FileNotFoundError(f"Planner export missing for {episode_uid}:{view_mode} -> {planner_index_path}")
            planner_index = pd.read_parquet(planner_index_path).reset_index(drop=True)
            if "planner_row_index" not in planner_index.columns:
                planner_index["planner_row_index"] = planner_index.index.astype("int64")
            planner_index = planner_index[["frame_position", "frame_index", "timestamp", "planner_row_index"]].copy()
            merged = base.merge(
                planner_index,
                on=["frame_position", "frame_index"],
                how="inner",
                validate="one_to_one",
                suffixes=("", "_planner"),
            )
            if merged.empty:
                continue
            merged["split"] = split_name
            merged["view_mode"] = view_mode
            merged["planner_output_dir"] = str(planner_output_dir)
            rows.append(merged)

    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, axis=0, ignore_index=True)
    combined["planner_row_index"] = combined["planner_row_index"].astype("int64")
    combined["local_target_npz_sample_index"] = combined["local_target_npz_sample_index"].astype("int64")
    combined["frame_position"] = combined["frame_position"].astype("int64")
    combined["frame_index"] = combined["frame_index"].astype("int64")
    return combined


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_root = Path(cfg["dataset_root"])
    planner_split_manifest_root = Path(cfg["checkpoint_planner_manifest_root"])
    planner_export_root = Path(cfg["planner_export_root"])
    output_root = Path(cfg["output_root"])

    split_manifest = json.loads((planner_split_manifest_root / "manifest.json").read_text(encoding="utf-8"))
    train_episodes = resolve_episode_selection(
        [str(uid) for uid in split_manifest["split"]["train_episode_uids"]],
        cfg.get("selection", {}),
    )
    val_episodes = resolve_episode_selection(
        [str(uid) for uid in split_manifest["split"]["val_episode_uids"]],
        cfg.get("selection", {}),
    )

    image_modalities = {str(k): str(v) for k, v in cfg.get("image_modalities", {}).items()}
    if not image_modalities:
        raise ValueError("image_modalities must be provided in the JSON config.")

    train_frame = build_split_frame(
        dataset_root=dataset_root,
        split_name="train",
        episode_uids=train_episodes,
        view_modes=[str(v) for v in cfg["train_view_modes"]],
        local_target_parquet_name=str(cfg["local_target_parquet_name"]),
        speed_weight_parquet_name=(
            str(cfg["speed_weight_parquet_name"]) if cfg.get("speed_weight_parquet_name") else None
        ),
        sample_weight_column=str(cfg.get("sample_weight_column", "sample_weight")),
        require_speed_weights=bool(cfg.get("require_speed_weights", False)),
        planner_index_filename=str(cfg.get("planner_index_filename", "planner_vla_rollout_index.parquet")),
        planner_export_root=planner_export_root,
        image_modalities=image_modalities,
        fail_on_missing_modalities=bool(cfg.get("fail_on_missing_modalities", True)),
        drop_horizon_clipped=bool(cfg.get("drop_horizon_clipped", False)),
    )
    val_frame = build_split_frame(
        dataset_root=dataset_root,
        split_name="val",
        episode_uids=val_episodes,
        view_modes=[str(v) for v in cfg["val_view_modes"]],
        local_target_parquet_name=str(cfg["local_target_parquet_name"]),
        speed_weight_parquet_name=(
            str(cfg["speed_weight_parquet_name"]) if cfg.get("speed_weight_parquet_name") else None
        ),
        sample_weight_column=str(cfg.get("sample_weight_column", "sample_weight")),
        require_speed_weights=bool(cfg.get("require_speed_weights", False)),
        planner_index_filename=str(cfg.get("planner_index_filename", "planner_vla_rollout_index.parquet")),
        planner_export_root=planner_export_root,
        image_modalities=image_modalities,
        fail_on_missing_modalities=bool(cfg.get("fail_on_missing_modalities", True)),
        drop_horizon_clipped=bool(cfg.get("drop_horizon_clipped", False)),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / str(cfg.get("train_index_name", "train_index.parquet"))
    val_path = output_root / str(cfg.get("val_index_name", "val_index.parquet"))
    train_frame.to_parquet(train_path, index=False)
    val_frame.to_parquet(val_path, index=False)

    manifest = {
        "dataset_root": str(dataset_root),
        "checkpoint_planner_manifest_root": str(planner_split_manifest_root),
        "planner_export_root": str(planner_export_root),
        "output_root": str(output_root),
        "local_target_parquet_name": str(cfg["local_target_parquet_name"]),
        "local_target_npz_name": str(cfg["local_target_npz_name"]),
        "speed_weight_parquet_name": (
            str(cfg["speed_weight_parquet_name"]) if cfg.get("speed_weight_parquet_name") else None
        ),
        "sample_weight_column": str(cfg.get("sample_weight_column", "sample_weight")),
        "require_speed_weights": bool(cfg.get("require_speed_weights", False)),
        "planner_index_filename": str(cfg.get("planner_index_filename", "planner_vla_rollout_index.parquet")),
        "planner_arrays_filename": str(cfg.get("planner_arrays_filename", "planner_vla_rollout_features.npz")),
        "train_index_name": train_path.name,
        "val_index_name": val_path.name,
        "train_view_modes": [str(v) for v in cfg["train_view_modes"]],
        "val_view_modes": [str(v) for v in cfg["val_view_modes"]],
        "image_modalities": image_modalities,
        "drop_horizon_clipped": bool(cfg.get("drop_horizon_clipped", False)),
        "counts": {
            "train_rows": int(len(train_frame)),
            "val_rows": int(len(val_frame)),
            "train_episodes": len(train_episodes),
            "val_episodes": len(val_episodes),
        },
        "sample_weight_stats": {
            "train_mean": (float(train_frame["sample_weight"].mean()) if not train_frame.empty else None),
            "train_min": (float(train_frame["sample_weight"].min()) if not train_frame.empty else None),
            "train_max": (float(train_frame["sample_weight"].max()) if not train_frame.empty else None),
            "val_mean": (float(val_frame["sample_weight"].mean()) if not val_frame.empty else None),
            "val_min": (float(val_frame["sample_weight"].min()) if not val_frame.empty else None),
            "val_max": (float(val_frame["sample_weight"].max()) if not val_frame.empty else None),
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Origami VLA manifest")
    print(f"  dataset_root : {dataset_root}")
    print(f"  output_root  : {output_root}")
    print(f"  train_rows   : {len(train_frame)}")
    print(f"  val_rows     : {len(val_frame)}")
    print(f"  train_file   : {train_path}")
    print(f"  val_file     : {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
