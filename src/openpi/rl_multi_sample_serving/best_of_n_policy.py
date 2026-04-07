from __future__ import annotations

import copy
import secrets
import time
from typing import Any

import numpy as np

from openpi.rl_multi_sample_serving.critic_score_client import CriticScoreClient
from openpi.rl_multi_sample_serving.multi_sample_policy import MultiSamplePolicy


def _select_sample(tree: dict[str, Any], index: int) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key, value in tree.items():
        if isinstance(value, np.ndarray):
            selected[key] = np.array(value[index], copy=True)
        elif isinstance(value, dict):
            selected[key] = copy.deepcopy(value)
        else:
            selected[key] = copy.deepcopy(value)
    return selected


def _cast_msgpack_compatible(tree: dict[str, Any]) -> dict[str, Any]:
    def _cast_leaf(x: Any):
        if not isinstance(x, np.ndarray):
            return x
        if x.dtype.kind == "V" or str(x.dtype) == "bfloat16":
            return x.astype(np.float32)
        return x

    return {
        key: _cast_msgpack_compatible(value)
        if isinstance(value, dict)
        else _cast_leaf(value)
        for key, value in tree.items()
    }


class BestOfNSamplePolicy:
    _CRITIC_INPUT_DIM = 16

    def __init__(
        self,
        *,
        policy: MultiSamplePolicy,
        critic_client: CriticScoreClient,
        num_samples: int,
        noise_scale: float = 1.0,
        send_policy_latent: bool = False,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        if noise_scale <= 0:
            raise ValueError(f"noise_scale must be > 0, got {noise_scale}")
        self._policy = policy
        self._critic_client = critic_client
        self._num_samples = int(num_samples)
        self._noise_scale = float(noise_scale)
        self._send_policy_latent = bool(send_policy_latent)

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(self._policy.metadata)
        metadata["best_of_n_enabled"] = True
        metadata["best_of_n_num_samples"] = self._num_samples
        metadata["best_of_n_noise_scale"] = self._noise_scale
        metadata["critic_reranking"] = True
        return metadata

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        total_start_time = time.monotonic()
        request_seed = int(secrets.randbelow(2**31 - 1) + 1)
        noise = self._sample_noise(request_seed)
        sampled_raw, raw_policy_timing = self._policy.infer_many_raw(
            obs,
            num_samples=self._num_samples,
            seed=request_seed,
            noise=noise,
        )
        if "policy_latent" not in sampled_raw:
            raise KeyError(
                "Multi-sample policy response is missing policy_latent. "
                "The critic reranker requires return_policy_latent support."
            )

        critic_state = self._extract_critic_state(sampled_raw, obs, self._num_samples)
        critic_action_chunk = self._extract_critic_action_chunk(sampled_raw, critic_state)
        critic_response = self._critic_client.score(
            policy_latent=sampled_raw["policy_latent"],
            state=critic_state,
            action_chunk=critic_action_chunk,
        )

        scores = np.asarray(critic_response["scores"], dtype=np.float32).reshape(-1)
        if scores.shape[0] != self._num_samples:
            raise ValueError(
                f"Critic returned {scores.shape[0]} scores for {self._num_samples} policy samples"
            )
        best_index = int(np.argmax(scores))
        selected_raw = _select_sample(sampled_raw, best_index)
        output_transform_start = time.monotonic()
        best = self._policy.apply_output_transform(selected_raw)
        output_transform_ms = (time.monotonic() - output_transform_start) * 1000.0
        if not self._send_policy_latent:
            best.pop("policy_latent", None)

        policy_timing = dict(raw_policy_timing)
        policy_timing.update(
            {
                "seed": request_seed,
                "num_samples": self._num_samples,
                "noise_scale": self._noise_scale,
                "selected_sample_index": best_index,
                "selected_estimated_return": float(scores[best_index]),
                "output_transform_ms": output_transform_ms,
                "total_ms": (time.monotonic() - total_start_time) * 1000.0,
            }
        )
        critic_timing = critic_response.get("critic_timing")
        if isinstance(critic_timing, dict):
            policy_timing["critic_ms"] = critic_timing.get("infer_ms")
        best["policy_timing"] = policy_timing
        return _cast_msgpack_compatible(best)

    def _sample_noise(self, seed: int) -> np.ndarray:
        action_horizon, action_dim = self._policy.action_shape
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(
            (self._num_samples, action_horizon, action_dim),
            dtype=np.float32,
        )
        if self._noise_scale != 1.0:
            noise = noise * np.float32(self._noise_scale)
        return noise

    def _extract_critic_state(
        self,
        sampled_raw: dict[str, Any],
        obs: dict[str, Any],
        num_samples: int,
    ) -> np.ndarray:
        if "state" in sampled_raw:
            state = np.asarray(sampled_raw["state"], dtype=np.float32)
            if state.ndim == 2 and state.shape[0] == num_samples:
                return self._normalize_critic_state(state)

        if "observation/state" in obs:
            state = obs["observation/state"]
        elif "observation.state" in obs:
            state = obs["observation.state"]
        elif "state" in obs:
            state = obs["state"]
        else:
            raise KeyError(
                "Unable to find observation state for critic reranking. "
                "Expected transformed sampled_raw['state'] or one of: observation/state, observation.state, state."
            )
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        return self._normalize_critic_state(np.repeat(state[np.newaxis, :], repeats=num_samples, axis=0))

    def _normalize_critic_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if state.ndim != 2:
            raise ValueError(f"Expected critic state with shape (B,D), got {state.shape}")
        if state.shape[-1] < self._CRITIC_INPUT_DIM:
            raise ValueError(
                f"Critic state dim mismatch. Expected at least {self._CRITIC_INPUT_DIM}, got {state.shape[-1]}"
            )
        return state[..., : self._CRITIC_INPUT_DIM]

    def _extract_critic_action_chunk(
        self,
        sampled_raw: dict[str, Any],
        critic_state: np.ndarray,
    ) -> np.ndarray:
        if "actions" not in sampled_raw:
            raise KeyError("Multi-sample policy response is missing actions for critic reranking.")

        action_chunk = np.asarray(sampled_raw["actions"], dtype=np.float32)
        if action_chunk.ndim != 3:
            raise ValueError(
                f"Expected sampled_raw['actions'] with shape (B,T,D), got {action_chunk.shape}"
            )

        critic_action_dim = self._CRITIC_INPUT_DIM
        if int(np.asarray(critic_state).shape[-1]) != critic_action_dim:
            raise ValueError(
                f"Critic state dim mismatch. Expected {critic_action_dim}, got {np.asarray(critic_state).shape[-1]}"
            )
        if action_chunk.shape[-1] < critic_action_dim:
            raise ValueError(
                "Raw sampled actions have fewer dims than critic action representation. "
                f"Got action dim {action_chunk.shape[-1]} for critic dim {critic_action_dim}."
            )
        return action_chunk[..., :critic_action_dim]
