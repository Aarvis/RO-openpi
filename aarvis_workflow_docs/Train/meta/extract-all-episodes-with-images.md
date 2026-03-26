# Extract All Episodes With Images

This document explains what [extract_all_episodes_with_images.py](/d:/LeHome-Challenge/lehome-challenge/Datasets/extract_all_episodes_with_images.py) does, what inputs it expects, what outputs it writes, and how to run it.

## What This Script Does

The script bulk-extracts all episodes from all dataset roots discovered under a source directory and produces:

1. One CSV and/or JSON file per episode
2. Decoded RGB image frames for each episode and camera

It is the multi-dataset, multi-episode version of [extract_episode_with_images.py](/d:/LeHome-Challenge/lehome-challenge/Datasets/extract_episode_with_images.py).

For every discovered episode, it builds per-row records containing:

- `garment_type`
- `dataset_name`
- `dataset_part`
- `data_source_file`
- `prompt`
- `observation.state`
- `action`
- `timestamp`
- `index`
- `episode_index`
- `frame_index`
- `task_index`
- `observation.image.top_rgb`
- `observation.image.right_rgb`
- `observation.image.left_rgb`

The image fields point to decoded JPEGs extracted from the episode’s top/right/left camera videos.

## What It Scans

The script is configured by constants inside the file and, by default, scans:

- `Datasets/garment_type_examples`

It looks for dataset roots that contain all of:

- `data/`
- `meta/`
- `meta/episodes/`
- `videos/`

Each such directory is treated as one dataset root.

## Inputs

This script does not use CLI arguments.

Instead, it uses constants defined near the top of the file:

- `EXAMPLE_ROOT`
  - root directory to scan for dataset roots
- `OUTPUT_ROOT`
  - where exported CSV/JSON/image outputs are written
- `OUTPUT_FORMAT`
  - `csv`, `json`, or `both`
- `DATASET_MAX_WORKERS`
  - number of dataset-level worker processes

Within each discovered dataset root, the script expects:

- parquet files under `data/`
- episode metadata parquet files under `meta/episodes/`
- camera videos under `videos/`

## How It Works

For each discovered dataset root, the script:

1. Finds episode metadata parquet files
2. Finds data parquet files
3. Matches episodes to the right data parquet when possible
4. Loads every episode’s timestep rows
5. Resolves the top/right/left camera videos for that episode
6. Extracts the relevant frame ranges from each video
7. Maps each timestep row to the nearest available extracted frame
8. Writes:
   - one JSON and/or CSV per episode
   - decoded JPEG frames under an episode image directory

It also tracks summary stats across all processed datasets:

- datasets discovered / processed / skipped
- episode parquet files discovered / processed / skipped
- data parquet files discovered / loaded
- episodes processed / failed
- total rows processed
- CSV / JSON files written
- total extracted frames
- extracted frames by camera

## Outputs

By default, output is written under:

- `Datasets/all_garment_type_exports`

The output layout is:

```text
all_garment_type_exports/
  <dataset_label>/
    <pair_id>/
      csv/
        episode_000000.csv
        episode_000001.csv
        ...
      json/
        episode_000000.json
        episode_000001.json
        ...
      images/
        episode_000000/
          top/
          right/
          left/
        episode_000001/
          top/
          right/
          left/
```

### Meaning of Output Path Pieces

- `<dataset_label>`
  - derived from the dataset root relative to `EXAMPLE_ROOT`
- `<pair_id>`
  - derived from the episode parquet location under `meta/episodes`

## Example Output JSON Shape

Each output JSON file contains one top-level list for one episode:

```json
[
  {
    "garment_type": "pant_short_merged",
    "dataset_name": "pant_short_merged",
    "dataset_part": "chunk-000__file-000",
    "data_source_file": "file-000.parquet",
    "prompt": "fold the garment on the table",
    "observation.state": [...],
    "action": [...],
    "timestamp": 0.0,
    "index": 0,
    "episode_index": 0,
    "frame_index": 0,
    "task_index": 0,
    "observation.image.top_rgb": ".../images/episode_000000/top/frame_000000.jpg",
    "observation.image.right_rgb": ".../images/episode_000000/right/frame_000000.jpg",
    "observation.image.left_rgb": ".../images/episode_000000/left/frame_000000.jpg"
  }
]
```

## How To Run

This script is intended to be run directly and does not take command-line arguments.

### Windows PowerShell

```powershell
python D:\LeHome-Challenge\lehome-challenge\Datasets\extract_all_episodes_with_images.py
```

### Linux / WSL

```bash
python /datadrive/LEHOME/lehome-challenge/Datasets/extract_all_episodes_with_images.py
```

## What You Need To Edit Before Running

If your source/output layout differs from the defaults, edit these constants inside the script:

- `EXAMPLE_ROOT`
- `OUTPUT_ROOT`
- `OUTPUT_FORMAT`
- `DATASET_MAX_WORKERS`

Typical edits:

- change `EXAMPLE_ROOT` if your input datasets live somewhere else
- change `OUTPUT_ROOT` if you want exports written to a different directory
- set `OUTPUT_FORMAT = "json"` if you only want JSON outputs
- reduce `DATASET_MAX_WORKERS` if CPU / disk load is too high

## Parallelism

The script parallelizes at the dataset-root level using `ProcessPoolExecutor`.

That means:

- different dataset roots can be processed in parallel
- each worker handles all episodes for one dataset root

This is useful when multiple garment-type datasets exist under the source root.

## Notes

- The script does not expose CLI arguments; configuration is file-based.
- It requires valid parquet metadata and readable video files.
- It uses OpenCV video decoding for frame extraction.
- If one episode fails, the script logs a warning and continues with others.
- At the end, it prints a summary of processed datasets, episodes, files, and extracted frames.
