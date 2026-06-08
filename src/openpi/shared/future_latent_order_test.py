import pytest

import openpi.shared.future_latent_order as _future_latent_order


def test_camera_reorder_indices_swaps_wrist_slots():
    assert _future_latent_order.camera_reorder_indices(
        ("top", "right_wrist", "left_wrist"),
        ("top", "left_wrist", "right_wrist"),
    ) == (0, 2, 1)


def test_camera_reorder_indices_rejects_missing_camera():
    with pytest.raises(ValueError, match="missing cameras"):
        _future_latent_order.camera_reorder_indices(
            ("top", "right_wrist"),
            ("top", "left_wrist", "right_wrist"),
        )
