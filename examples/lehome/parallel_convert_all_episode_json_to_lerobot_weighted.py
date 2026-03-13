"""
Parallel weighted LeHome JSON -> LeRobot conversion.

Design:
1. Split episode JSONs into ordered shards.
2. Convert each shard in parallel into temporary episode bundles on disk.
3. Merge bundles back into one final LeRobot dataset in the exact original episode order.

This keeps the final dataset deterministic and aligned with the single-threaded weighted converter.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import tempfile

from tqdm import tqdm
import tyro

from parallel_convert import common
from parallel_convert import merge
from parallel_convert import worker


def main(
    *,
    json_root: str = "../lehome-challenge/Datasets/all_episode_exports",
    json_glob: str = "**/json/episode_*.json",
    repo_name: str = "local/lehome_all_episodes_weighted",
    source_root: str = "..",
    overwrite: bool = True,
    default_fps: int = 30,
    workers: int = 8,
    temp_root: str | None = None,
    keep_temp_shards: bool = False,
    compress_temp_shards: bool = False,
):
    json_root_obj = Path(json_root).resolve()
    source_root_obj = Path(source_root).resolve()
    if not json_root_obj.exists():
        raise FileNotFoundError(f"JSON root not found: {json_root_obj}")

    episode_jsons = common.find_episode_jsons(json_root_obj, json_glob)
    if not episode_jsons:
        raise FileNotFoundError(f"No JSON files found under {json_root_obj} with glob '{json_glob}'")

    first_rows = None
    for candidate in episode_jsons:
        try:
            first_rows = common.load_rows(candidate)
            break
        except Exception:
            continue
    if first_rows is None:
        raise ValueError("No valid non-empty episode JSON found.")

    schema = common.derive_schema(first_rows, source_root_obj)
    fps = common.infer_fps(first_rows, default_fps=default_fps)

    output_path = common.local_repo_output_path(repo_name)
    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    elif output_path.exists():
        raise FileExistsError(f"Output dataset already exists: {output_path}")

    if temp_root is None:
        sanitized = common.sanitize_repo_name(repo_name)
        temp_root_obj = Path(tempfile.gettempdir()) / f"openpi_parallel_convert_{sanitized}"
    else:
        temp_root_obj = Path(temp_root).resolve()

    if temp_root_obj.exists():
        shutil.rmtree(temp_root_obj)
    temp_root_obj.mkdir(parents=True, exist_ok=True)

    worker_count = max(1, min(int(workers), len(episode_jsons)))
    episode_chunks = common.chunk_ordered_items(episode_jsons, worker_count)

    shard_stats = {"episodes_saved": 0, "frames_saved": 0}
    manifest_paths: list[Path] = []

    print(f"[Info] JSON root:       {json_root_obj}")
    print(f"[Info] Source root:     {source_root_obj}")
    print(f"[Info] Repo name:       {repo_name}")
    print(f"[Info] Output path:     {output_path}")
    print(f"[Info] Temp root:       {temp_root_obj}")
    print(f"[Info] Episode JSONs:   {len(episode_jsons)}")
    print(f"[Info] Worker shards:   {worker_count}")
    print(f"[Info] Compress temp:   {compress_temp_shards}")

    futures = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for shard_index, chunk in enumerate(episode_chunks):
            futures.append(
                executor.submit(
                    worker.convert_shard,
                    shard_index=shard_index,
                    episode_specs=[(global_idx, str(path)) for global_idx, path in chunk],
                    source_root=str(source_root_obj),
                    temp_root=str(temp_root_obj),
                    state_dim=int(schema["state_dim"]),
                    action_dim=int(schema["action_dim"]),
                    default_prompt=str(schema["prompt"]),
                    compress_temp_shards=compress_temp_shards,
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing shards", unit="shard"):
            result = future.result()
            shard_stats["episodes_saved"] += int(result["episodes_saved"])
            shard_stats["frames_saved"] += int(result["frames_saved"])
            manifest_paths.append(Path(result["manifest_path"]))

    merge_stats = merge.merge_shards(
        manifest_paths=sorted(manifest_paths),
        repo_name=repo_name,
        fps=fps,
        top_shape=tuple(schema["top_shape"]),
        left_shape=tuple(schema["left_shape"]),
        right_shape=tuple(schema["right_shape"]),
        state_dim=int(schema["state_dim"]),
        action_dim=int(schema["action_dim"]),
    )

    summary = {
        "repo_name": repo_name,
        "output_path": str(output_path),
        "temp_root": str(temp_root_obj),
        "episode_jsons_found": len(episode_jsons),
        "shard_episodes_saved": shard_stats["episodes_saved"],
        "shard_frames_saved": shard_stats["frames_saved"],
        "merged_episodes_saved": merge_stats["episodes_saved"],
        "merged_frames_saved": merge_stats["frames_saved"],
        "fps": fps,
        "worker_shards": worker_count,
    }
    summary_path = output_path / "parallel_convert_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not keep_temp_shards:
        shutil.rmtree(temp_root_obj, ignore_errors=True)

    print(f"\nSaved local weighted LeRobot dataset to: {output_path}")
    print(f"repo_id={repo_name}, fps={fps}")
    print(
        "stats: "
        f"episode_jsons_found={summary['episode_jsons_found']}, "
        f"shard_episodes_saved={summary['shard_episodes_saved']}, "
        f"shard_frames_saved={summary['shard_frames_saved']}, "
        f"merged_episodes_saved={summary['merged_episodes_saved']}, "
        f"merged_frames_saved={summary['merged_frames_saved']}"
    )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    tyro.cli(main)
