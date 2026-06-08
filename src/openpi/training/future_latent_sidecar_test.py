import numpy as np

import openpi.training.future_latent_sidecar as _future_latent_sidecar


def _latent_bytes(value: float) -> bytes:
    return np.asarray([[value]], dtype=np.float32).tobytes(order="C")


def test_future_latent_sidecar_reads_policy_camera_order():
    transform = _future_latent_sidecar.FutureLatentSidecarTransform(
        sidecar_root="unused",
        source_repo_id="unused",
    )
    row = {
        "latent_dtype": "float32",
        "latent_shape": [1, 1],
        "top_future_latent_valid": True,
        "right_wrist_future_latent_valid": True,
        "left_wrist_future_latent_valid": True,
        "top_future_latent_pred": _latent_bytes(1.0),
        "right_wrist_future_latent_pred": _latent_bytes(2.0),
        "left_wrist_future_latent_pred": _latent_bytes(3.0),
        "top_future_latent_true": _latent_bytes(10.0),
        "right_wrist_future_latent_true": _latent_bytes(20.0),
        "left_wrist_future_latent_true": _latent_bytes(30.0),
    }

    future_latent = transform._future_latent_from_row(row)

    np.testing.assert_array_equal(future_latent["pred"][:, 0, 0], np.asarray([1.0, 3.0, 2.0]))
    np.testing.assert_array_equal(future_latent["true"][:, 0, 0], np.asarray([10.0, 30.0, 20.0]))
    np.testing.assert_array_equal(future_latent["valid_mask"], np.asarray([True, True, True]))
