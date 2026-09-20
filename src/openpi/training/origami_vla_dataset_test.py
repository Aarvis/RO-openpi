import pickle

import pandas as pd

from openpi.training import origami_vla_dataset as _dataset


def test_spawn_pickle_reloads_manifest_rows_in_worker(monkeypatch) -> None:
    """Spawn serialization must preserve rows while omitting the parent row list."""

    load_calls: list[str] = []

    def fake_load_manifest_rows(_settings, split: str) -> pd.DataFrame:
        load_calls.append(split)
        return pd.DataFrame(
            [
                {
                    "episode_uid": "episode_a",
                    "frame_position": 3,
                    "planner_enabled": True,
                    "view_mode": "fixed_7",
                    "planner_output_dir": "/tmp/planner/episode_a/fixed_7",
                },
                {
                    "episode_uid": "episode_b",
                    "frame_position": 9,
                    "planner_enabled": False,
                    "view_mode": "planner_disabled",
                    "planner_output_dir": "",
                },
            ]
        )

    monkeypatch.setattr(_dataset, "load_manifest_rows", fake_load_manifest_rows)
    settings = _dataset.OrigamiVlaSettings(
        dataset_root="/tmp/dataset",
        manifest_root="/tmp/manifest",
        action_source="action_chunk",
    )
    dataset = _dataset.OrigamiVlaDataset(settings, split="train")

    serialized_state = dataset.__getstate__()
    assert serialized_state["_rows"] is None
    assert serialized_state["_planner_output_dirs_by_episode"] is None

    restored = pickle.loads(pickle.dumps(dataset))

    assert load_calls == ["train", "train"]
    assert restored._rows == dataset._rows
    assert restored._planner_output_dirs_by_episode == dataset._planner_output_dirs_by_episode
    assert restored._episode_cache == {}
    assert restored._video_cache == {}
