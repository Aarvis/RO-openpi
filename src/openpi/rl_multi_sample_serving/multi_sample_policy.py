from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
import torch


def _stack_trees(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Expected at least one sample to stack.")
    return jax.tree.map(lambda *xs: np.stack(xs, axis=0), *samples)


def _copy_sample(tree: dict[str, Any], index: int) -> dict[str, Any]:
    return jax.tree.map(lambda x: np.array(np.asarray(x)[index], copy=True), tree)


class MultiSamplePolicy:
    """Separate multi-sample inference wrapper around an existing OpenPI policy."""

    def __init__(self, policy: Any):
        required = (
            "_model",
            "_input_transform",
            "_output_transform",
            "_sample_kwargs",
            "_is_pytorch_model",
            "_sample_actions",
        )
        missing = [name for name in required if not hasattr(policy, name)]
        if missing:
            raise TypeError(
                "Wrapped policy does not expose the internals required for multi-sample inference: "
                f"{', '.join(missing)}"
            )
        self._policy = policy

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(getattr(self._policy, "metadata", {}))
        metadata["supports_infer_many"] = True
        metadata["multi_sample_protocol_version"] = 1
        return metadata

    @property
    def action_shape(self) -> tuple[int, int]:
        return self._get_action_shape()

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self._policy.infer(obs)

    def infer_many(
        self,
        obs: dict[str, Any],
        *,
        num_samples: int,
        seed: int | None = None,
        noise: np.ndarray | None = None,
    ) -> dict[str, Any]:
        raw_outputs, timing = self.infer_many_raw(
            obs,
            num_samples=num_samples,
            seed=seed,
            noise=noise,
        )
        output_transform_start_time = time.monotonic()
        transformed_outputs = self._apply_output_transform_per_sample(raw_outputs)
        output_transform_ms = (time.monotonic() - output_transform_start_time) * 1000
        policy_timing = dict(timing)
        policy_timing["output_transform_ms"] = output_transform_ms
        policy_timing["infer_ms"] = (
            float(policy_timing.get("input_transform_ms", 0.0))
            + float(policy_timing.get("repeat_inputs_ms", 0.0))
            + float(policy_timing.get("model_infer_ms", 0.0))
            + output_transform_ms
        )
        transformed_outputs["policy_timing"] = policy_timing
        return transformed_outputs

    def infer_many_raw(
        self,
        obs: dict[str, Any],
        *,
        num_samples: int,
        seed: int | None = None,
        noise: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        total_start_time = time.monotonic()
        preprocess_start_time = total_start_time
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._policy._input_transform(inputs)
        input_transform_ms = (time.monotonic() - preprocess_start_time) * 1000

        repeat_start_time = time.monotonic()
        batched_inputs = self._repeat_inputs(inputs, num_samples)
        repeat_inputs_ms = (time.monotonic() - repeat_start_time) * 1000

        model_start_time = time.monotonic()
        if self._policy._is_pytorch_model:
            outputs = self._infer_many_pytorch(
                batched_inputs,
                num_samples=num_samples,
                seed=seed,
                noise=noise,
            )
        else:
            outputs = self._infer_many_jax(
                batched_inputs,
                num_samples=num_samples,
                seed=seed,
                noise=noise,
            )
        model_infer_ms = (time.monotonic() - model_start_time) * 1000

        total_infer_ms = (time.monotonic() - total_start_time) * 1000
        timing = {
            "infer_ms": total_infer_ms,
            "input_transform_ms": input_transform_ms,
            "repeat_inputs_ms": repeat_inputs_ms,
            "model_infer_ms": model_infer_ms,
            "output_transform_ms": 0.0,
            "num_samples": num_samples,
            "backend": "pytorch" if self._policy._is_pytorch_model else "jax",
            "has_explicit_noise": noise is not None,
            "has_seed": seed is not None,
        }
        return outputs, timing

    def _repeat_inputs(self, inputs: dict[str, Any], num_samples: int) -> dict[str, Any]:
        return jax.tree.map(
            lambda x: np.repeat(np.expand_dims(np.asarray(x), axis=0), repeats=num_samples, axis=0),
            inputs,
        )

    def _get_action_shape(self) -> tuple[int, int]:
        model = self._policy._model
        action_horizon = getattr(model, "action_horizon", None)
        action_dim = getattr(model, "action_dim", None)
        if action_horizon is None or action_dim is None:
            config = getattr(model, "config", None)
            action_horizon = getattr(config, "action_horizon", action_horizon)
            action_dim = getattr(config, "action_dim", action_dim)
        if action_horizon is None or action_dim is None:
            raise AttributeError("Unable to determine action_horizon/action_dim from wrapped policy model.")
        return int(action_horizon), int(action_dim)

    def _normalize_noise(self, noise: np.ndarray, num_samples: int) -> np.ndarray:
        action_horizon, action_dim = self._get_action_shape()
        normalized = np.asarray(noise, dtype=np.float32)
        if normalized.ndim == 2:
            if num_samples != 1:
                raise ValueError(
                    f"2D noise shape {normalized.shape} is only valid when num_samples=1, got {num_samples}"
                )
            normalized = normalized[np.newaxis, ...]
        if normalized.shape != (num_samples, action_horizon, action_dim):
            raise ValueError(
                "Noise shape mismatch. "
                f"Expected {(num_samples, action_horizon, action_dim)}, got {normalized.shape}"
            )
        return normalized

    def _apply_output_transform_per_sample(self, outputs: dict[str, Any]) -> dict[str, Any]:
        batch_size = int(np.asarray(outputs["actions"]).shape[0])
        per_sample_outputs = []
        for index in range(batch_size):
            sample = _copy_sample(outputs, index)
            per_sample_outputs.append(self._policy._output_transform(sample))
        return _stack_trees(per_sample_outputs)

    def apply_output_transform(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self._policy._output_transform(sample)

    def _infer_many_jax(
        self,
        batched_inputs: dict[str, Any],
        *,
        num_samples: int,
        seed: int | None,
        noise: np.ndarray | None,
    ) -> dict[str, Any]:
        inputs = jax.tree.map(jnp.asarray, batched_inputs)
        observation = _model.Observation.from_dict(inputs)
        sample_kwargs = dict(self._policy._sample_kwargs)

        if noise is not None:
            sample_kwargs["noise"] = jnp.asarray(self._normalize_noise(noise, num_samples))

        if seed is not None:
            sample_rng = jax.random.key(seed)
        else:
            self._policy._rng, sample_rng = jax.random.split(self._policy._rng)

        outputs = {
            "state": inputs["state"],
        }
        return_policy_latent = bool(sample_kwargs.pop("return_policy_latent", False))
        if return_policy_latent:
            sample_actions_with_policy_latent = getattr(self._policy, "_sample_actions_with_policy_latent", None)
            if sample_actions_with_policy_latent is None:
                raise ValueError("Wrapped policy does not support returning policy_latent.")
            outputs.update(sample_actions_with_policy_latent(sample_rng, observation, **sample_kwargs))
        else:
            outputs["actions"] = self._policy._sample_actions(sample_rng, observation, **sample_kwargs)
        if "state_joint" in inputs:
            outputs["state_joint"] = inputs["state_joint"]
        return jax.tree.map(np.asarray, outputs)

    def _sample_noise_pytorch(self, num_samples: int, device: str, seed: int) -> torch.Tensor:
        action_horizon, action_dim = self._get_action_shape()
        try:
            generator = torch.Generator(device=device)
        except TypeError:
            generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.randn(
            (num_samples, action_horizon, action_dim),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

    def _infer_many_pytorch(
        self,
        batched_inputs: dict[str, Any],
        *,
        num_samples: int,
        seed: int | None,
        noise: np.ndarray | None,
    ) -> dict[str, Any]:
        device = self._policy._pytorch_device
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(device), batched_inputs)
        observation = _model.Observation.from_dict(inputs)
        sample_kwargs = dict(self._policy._sample_kwargs)

        if noise is not None:
            sample_kwargs["noise"] = torch.from_numpy(self._normalize_noise(noise, num_samples)).to(device)
        elif seed is not None:
            sample_kwargs["noise"] = self._sample_noise_pytorch(num_samples, device, seed)

        outputs = {
            "state": inputs["state"],
        }
        return_policy_latent = bool(sample_kwargs.pop("return_policy_latent", False))
        if return_policy_latent:
            sample_actions_with_policy_latent = getattr(self._policy, "_sample_actions_with_policy_latent", None)
            if sample_actions_with_policy_latent is None:
                raise ValueError("Wrapped policy does not support returning policy_latent.")
            outputs.update(sample_actions_with_policy_latent(device, observation, **sample_kwargs))
        else:
            outputs["actions"] = self._policy._sample_actions(device, observation, **sample_kwargs)
        if "state_joint" in inputs:
            outputs["state_joint"] = inputs["state_joint"]
        return jax.tree.map(lambda x: np.asarray(x.detach().cpu()), outputs)
