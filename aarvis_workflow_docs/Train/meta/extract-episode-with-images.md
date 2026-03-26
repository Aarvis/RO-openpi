# Extract Episode With Images

This document explains what [extract_episode_with_images.py](/d:/LeHome-Challenge/lehome-challenge/Datasets/extract_episode_with_images.py) does, what inputs it expects, what outputs it writes, and how to run it.

## What This Script Does

The script extracts one episode from a LeRobot-style dataset stored in parquet files and produces two things:

1. A table of episode rows
2. Decoded RGB image frames for that episode

The table output contains one record per timestep in the episode. Each record includes:

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

The image fields in the table point to decoded JPEG frames extracted from the episode’s top/right/left camera videos.

## Inputs

The script expects:

- a data parquet file
  - contains per-timestep fields like state, action, timestamp, frame index, episode index
- an episodes parquet file
  - contains episode metadata, task/prompt info, and the video timestamp bounds for that episode
- an `episode_index`
  - selects which episode to extract

CLI arguments:

- `--data-parquet`
  - path to the main timestep parquet
- `--episodes-parquet`
  - path to the episode metadata parquet
- `--episode-index`
  - integer episode id to extract
- `--images-output-root`
  - where decoded camera frames will be written
- `--table-output-root`
  - where the episode JSON/CSV will be written
- `--output-format`
  - `csv`, `json`, or `both`
- `--decoder`
  - `auto`, `opencv`, or `ffmpeg`

## How It Works

The script does the following:

1. Loads episode metadata from `episodes parquet`
2. Finds all timestep rows in `data parquet` for the selected `episode_index`
3. Resolves the corresponding top/right/left video files for that episode
4. Uses the episode timestamp range to extract the relevant video frames
5. Maps each episode row to the nearest available extracted frame for each camera
6. Writes:
   - decoded JPEG images under the image output directory
   - a JSON and/or CSV table whose image paths point to those extracted frames

## Outputs

By default, the script writes:

- image frames under:
  - `Datasets/sample_episode_images/episode_<episode_index>/top`
  - `Datasets/sample_episode_images/episode_<episode_index>/right`
  - `Datasets/sample_episode_images/episode_<episode_index>/left`
- table files under:
  - `Datasets/episode_<episode_index>.json`
  - `Datasets/episode_<episode_index>.csv`

### Example Output Image Layout

```text
Datasets/sample_episode_images/
  episode_200/
    top/
      frame_008229.jpg
      frame_008230.jpg
      ...
    right/
      frame_008229.jpg
      frame_008230.jpg
      ...
    left/
      frame_008229.jpg
      frame_008230.jpg
      ...
```

### Example Output JSON Shape

```json
[
  {
    "prompt": "fold the garment on the table",
    "observation.state": [...],
    "action": [...],
    "timestamp": 0.0,
    "index": 8229,
    "episode_index": 200,
    "frame_index": 0,
    "task_index": 0,
    "observation.image.top_rgb": "Datasets/sample_episode_images/episode_200/top/frame_008229.jpg",
    "observation.image.right_rgb": "Datasets/sample_episode_images/episode_200/right/frame_008229.jpg",
    "observation.image.left_rgb": "Datasets/sample_episode_images/episode_200/left/frame_008229.jpg"
  }
]
```

## Detailed Command

Run from the `D:\LeHome-Challenge` repo root or adjust paths accordingly.

### Windows PowerShell

```powershell
python D:\LeHome-Challenge\lehome-challenge\Datasets\extract_episode_with_images.py `
  --data-parquet D:\LeHome-Challenge\lehome-challenge\Datasets\example\top_long_merged\data\chunk-000\file-000.parquet `
  --episodes-parquet D:\LeHome-Challenge\lehome-challenge\Datasets\example\top_long_merged\meta\episodes\chunk-000\file-000.parquet `
  --episode-index 200 `
  --images-output-root D:\LeHome-Challenge\lehome-challenge\Datasets\sample_episode_images `
  --table-output-root D:\LeHome-Challenge\lehome-challenge\Datasets `
  --output-format both `
  --decoder auto
```

### Linux / WSL

```bash
python /datadrive/LEHOME/lehome-challenge/Datasets/extract_episode_with_images.py \
  --data-parquet /datadrive/LEHOME/lehome-challenge/Datasets/example/top_long_merged/data/chunk-000/file-000.parquet \
  --episodes-parquet /datadrive/LEHOME/lehome-challenge/Datasets/example/top_long_merged/meta/episodes/chunk-000/file-000.parquet \
  --episode-index 200 \
  --images-output-root /datadrive/LEHOME/lehome-challenge/Datasets/sample_episode_images \
  --table-output-root /datadrive/LEHOME/lehome-challenge/Datasets \
  --output-format both \
  --decoder auto
```

## Decoder Options

- `auto`
  - tries OpenCV first, then falls back to ffmpeg if needed
- `opencv`
  - uses OpenCV only
- `ffmpeg`
  - uses ffmpeg software decode directly

Use `ffmpeg` explicitly if OpenCV has trouble decoding the video format.

## Output Format Options

- `--output-format json`
  - writes only JSON
- `--output-format csv`
  - writes only CSV
- `--output-format both`
  - writes both JSON and CSV

## Notes

- The script extracts frames only for the selected episode, not the whole dataset.
- It rewrites any stale `frame_*.jpg` files in the target episode image directory.
- The JSON output is usually the more useful artifact for downstream dataset conversion.
