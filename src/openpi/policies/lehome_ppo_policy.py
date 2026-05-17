from __future__ import annotations

import dataclasses
import os
import time
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi.policies import lehome_ppo_heads


@dataclasses.dataclass(frozen=True)
class LehomePPORuntimeConfig:
    action_horizon: int = 10
    action_dim: int = 12
    latent_dim: int = 1024
    state_dim: int = 12
    token_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    correction_scale: float = 0.03
    delta_clip: float = 2.0
    log_std_init: float = -0.5
    min_log_std: float = -5.0
    max_log_std: float = 1.0
    deterministic: bool = False
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    delta_coef: float = 0.001


def _as_scalar(value: float) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(())


def _select_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _configure_torch_cuda_fraction(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    raw_fraction = os.environ.get("OPENPI_PPO_GPU_MEM_FRACTION")
    if raw_fraction is None or raw_fraction == "":
        return None
    fraction = float(raw_fraction)
    if fraction <= 0:
        return None
    if fraction > 1.0:
        raise ValueError(f"OPENPI_PPO_GPU_MEM_FRACTION must be <= 1.0, got {fraction}")
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    torch.cuda.set_per_process_memory_fraction(fraction, device_index)
    return fraction


def _extract_state(obs: dict[str, Any], base_outputs: dict[str, Any], state_dim: int) -> np.ndarray:
    for key in ("state_joint", "observation/state", "observation.state", "state"):
        if key in base_outputs:
            value = base_outputs[key]
        elif key in obs:
            value = obs[key]
        else:
            continue
        state = np.asarray(value, dtype=np.float32).reshape(-1)
        if state.size >= state_dim:
            return state[:state_dim]
    raise ValueError(f"Could not find a state vector with at least {state_dim} dims for PPO heads.")


class LehomePPOPolicy(_base_policy.BasePolicy):
    """Frozen base VLA plus lightweight PPO correction/value heads for rollout."""

    def __init__(
        self,
        base_policy: _base_policy.BasePolicy,
        *,
        config: LehomePPORuntimeConfig,
        actor_head_path: str | None = None,
        value_head_path: str | None = None,
        device: str | None = None,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._base_policy = base_policy
        self._config = config
        self._device = _select_device(device)
        self._ppo_gpu_mem_fraction = _configure_torch_cuda_fraction(self._device)
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(int(seed))
        head_config = lehome_ppo_heads.PPOHeadConfig(
            action_horizon=config.action_horizon,
            action_dim=config.action_dim,
            latent_dim=config.latent_dim,
            state_dim=config.state_dim,
            token_dim=config.token_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            log_std_init=config.log_std_init,
            min_log_std=config.min_log_std,
            max_log_std=config.max_log_std,
        )
        self._actor = lehome_ppo_heads.load_actor_head(actor_head_path, head_config, device=self._device)
        self._value = lehome_ppo_heads.load_value_head(value_head_path, head_config, device=self._device)
        base_metadata = getattr(base_policy, "metadata", {}) or {}
        self._metadata = {
            **base_metadata,
            **(metadata or {}),
            "ppo_policy": {
                "enabled": True,
                "actor_head_path": actor_head_path,
                "value_head_path": value_head_path,
                "device": str(self._device),
                "gpu_mem_fraction": self._ppo_gpu_mem_fraction,
                "correction_scale": config.correction_scale,
                "delta_clip": config.delta_clip,
                "deterministic": config.deterministic,
            },
        }

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        base_outputs = self._base_policy.infer(obs)
        if "actions" not in base_outputs:
            raise ValueError("Base policy response is missing 'actions'.")
        if "policy_latent" not in base_outputs:
            raise ValueError(
                "Base policy response is missing 'policy_latent'. "
                "Create the base policy with return_policy_latent=True."
            )

        start_time = time.monotonic()
        base_action_np = np.asarray(base_outputs["actions"], dtype=np.float32)
        policy_latent_np = np.asarray(base_outputs["policy_latent"], dtype=np.float32)
        state_np = _extract_state(obs, base_outputs, self._config.state_dim)

        if base_action_np.shape != (self._config.action_horizon, self._config.action_dim):
            raise ValueError(
                "Expected base actions with shape "
                f"({self._config.action_horizon}, {self._config.action_dim}), got {base_action_np.shape}"
            )
        if policy_latent_np.shape != (self._config.action_horizon, self._config.latent_dim):
            raise ValueError(
                "Expected policy_latent with shape "
                f"({self._config.action_horizon}, {self._config.latent_dim}), got {policy_latent_np.shape}"
            )

        policy_latent = torch.as_tensor(policy_latent_np, dtype=torch.float32, device=self._device)
        base_action = torch.as_tensor(base_action_np, dtype=torch.float32, device=self._device)
        state = torch.as_tensor(state_np, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            delta_mean, delta_std = self._actor(policy_latent, base_action, state)
            value = self._value(policy_latent, base_action, state)
            distribution = torch.distributions.Normal(delta_mean, delta_std)
            if self._config.deterministic:
                sampled_delta = delta_mean
            else:
                sampled_delta = delta_mean + delta_std * torch.randn(
                    delta_mean.shape,
                    generator=self._generator,
                    device=self._device,
                    dtype=delta_mean.dtype,
                )
            delta_action = torch.clamp(sampled_delta, -self._config.delta_clip, self._config.delta_clip)
            old_log_prob = distribution.log_prob(delta_action).sum(dim=(-1, -2))
            entropy = distribution.entropy().sum(dim=(-1, -2))
            scaled_delta = float(self._config.correction_scale) * delta_action
            final_action = base_action + scaled_delta.squeeze(0)
            delta_l2 = torch.mean(torch.square(delta_action))

        ppo_time = time.monotonic() - start_time
        result = dict(base_outputs)
        result["base_action"] = base_action_np
        result["delta_action"] = np.asarray(delta_action.squeeze(0).detach().cpu(), dtype=np.float32)
        result["scaled_delta_action"] = np.asarray(scaled_delta.squeeze(0).detach().cpu(), dtype=np.float32)
        result["delta_mean"] = np.asarray(delta_mean.squeeze(0).detach().cpu(), dtype=np.float32)
        result["delta_std"] = np.asarray(delta_std.squeeze(0).detach().cpu(), dtype=np.float32)
        result["old_log_prob"] = lehome_ppo_heads.to_numpy_scalar(old_log_prob)
        result["old_value"] = lehome_ppo_heads.to_numpy_scalar(value)
        result["entropy"] = lehome_ppo_heads.to_numpy_scalar(entropy)
        result["correction_scale"] = _as_scalar(self._config.correction_scale)
        result["value_coef"] = _as_scalar(self._config.value_coef)
        result["entropy_coef"] = _as_scalar(self._config.entropy_coef)
        result["delta_coef"] = _as_scalar(self._config.delta_coef)
        result["delta_l2"] = lehome_ppo_heads.to_numpy_scalar(delta_l2)
        result["A_final"] = np.asarray(final_action.detach().cpu(), dtype=np.float32)
        result["actions"] = result["A_final"]
        result["ppo_timing"] = {"infer_ms": ppo_time * 1000}
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
