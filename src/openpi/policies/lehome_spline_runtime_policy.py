from __future__ import annotations

from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import websocket_client_policy as _ws_policy


class LehomeSplineRuntimePolicy(_base_policy.BasePolicy):
    """Runtime wrapper: spline server -> spline-conditioned OpenPI policy."""

    def __init__(
        self,
        *,
        openpi_policy: _base_policy.BasePolicy,
        spline_server_url: str,
        default_prompt: str | None = None,
        fail_on_invalid_spline: bool = True,
        rotate_top_rgb_180: bool = True,
    ) -> None:
        self._openpi_policy = openpi_policy
        self._spline_client = _ws_policy.WebsocketClientPolicy(host=spline_server_url)
        self._default_prompt = default_prompt
        self._fail_on_invalid_spline = bool(fail_on_invalid_spline)
        self._rotate_top_rgb_180 = bool(rotate_top_rgb_180)

    @staticmethod
    def _rotate_image_180(image: Any) -> np.ndarray:
        image_np = np.asarray(image, dtype=np.uint8)
        if image_np.ndim < 2:
            raise ValueError(f"Expected image with at least 2 dimensions, got shape={image_np.shape}")
        return np.ascontiguousarray(np.rot90(image_np, 2, axes=(0, 1)))

    def _prepare_spline_observation(self, obs: dict) -> dict:
        prepared = dict(obs)
        if not self._rotate_top_rgb_180:
            return prepared

        for key in ("observation/top_rgb", "observation.images.top_rgb", "observation.image.top_rgb", "observation.top_rgb"):
            if key in prepared:
                prepared[key] = self._rotate_image_180(prepared[key])
                break
        return prepared

    def infer(self, obs: dict) -> dict:
        spline_obs = self._prepare_spline_observation(obs)
        spline_result = self._spline_client.infer(spline_obs)
        prediction_valid = bool(spline_result.get("prediction_valid", False))
        used_last_valid_fallback = bool(spline_result.get("used_last_valid_fallback", False))

        if not prediction_valid and not used_last_valid_fallback and self._fail_on_invalid_spline:
            raise RuntimeError(
                "Spline server returned no valid robot spline for the current request. "
                f"invalid_reason={spline_result.get('invalid_reason')!r}"
            )

        prompt = obs.get("prompt", self._default_prompt)
        if prompt is None:
            raise ValueError("OpenPI spline runtime requires a prompt in the request or default_prompt at server startup.")

        openpi_obs = {
            "observation/state": np.asarray(obs["observation/state"], dtype=np.float32),
            "robot_spline_coefficients": np.asarray(spline_result["predicted_robot_coefficients"], dtype=np.float32),
            "robot_spline_knots": np.asarray(spline_result["predicted_robot_knots"], dtype=np.float32),
            "prompt": prompt,
        }
        result = self._openpi_policy.infer(openpi_obs)
        result["spline_prediction_valid"] = prediction_valid
        result["spline_used_last_valid_fallback"] = used_last_valid_fallback
        result["spline_prediction_source"] = spline_result.get("prediction_source")
        result["spline_invalid_reason"] = spline_result.get("invalid_reason")
        result["spline_prompt_id"] = spline_result.get("prompt_id")
        result["spline_prompt_category_id"] = spline_result.get("prompt_category_id")
        result["spline_requested_category_id"] = spline_result.get("requested_category_id")
        result["spline_start_u"] = np.asarray(spline_result.get("predicted_human_start_u", np.nan), dtype=np.float32)
        result["spline_end_u"] = np.asarray(spline_result.get("predicted_human_end_u", np.nan), dtype=np.float32)
        result["spline_start_checkpoint_index"] = np.asarray(
            -1 if spline_result.get("predicted_start_checkpoint_index") is None else spline_result.get("predicted_start_checkpoint_index"),
            dtype=np.int32,
        )
        result["spline_end_checkpoint_index"] = np.asarray(
            -1 if spline_result.get("predicted_end_checkpoint_index") is None else spline_result.get("predicted_end_checkpoint_index"),
            dtype=np.int32,
        )
        result["spline_start_progress"] = np.asarray(spline_result.get("predicted_start_progress", np.nan), dtype=np.float32)
        result["spline_end_progress"] = np.asarray(
            np.nan if spline_result.get("predicted_end_progress") is None else spline_result.get("predicted_end_progress"),
            dtype=np.float32,
        )
        result["spline_projection_condition_proxy"] = np.asarray(
            spline_result.get("projection_condition_proxy", np.nan),
            dtype=np.float32,
        )
        result["spline_span_entropy"] = np.asarray(spline_result.get("span_entropy", np.nan), dtype=np.float32)
        result["spline_human_local_coefficient_count"] = np.asarray(
            spline_result.get("human_local_coefficient_count", np.nan),
            dtype=np.float32,
        )
        if "runtime_timing" in spline_result:
            result["spline_runtime_timing"] = spline_result["runtime_timing"]
        return result

    def reset(self) -> None:
        return None
