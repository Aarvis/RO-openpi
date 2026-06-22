from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PPOHeadConfig:
    action_horizon: int = 10
    action_dim: int = 12
    latent_dim: int = 1024
    state_dim: int = 12
    token_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    log_std_init: float = -0.5
    min_log_std: float = -5.0
    max_log_std: float = 1.0


def _build_encoder(config: PPOHeadConfig) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=config.token_dim,
        nhead=config.num_heads,
        dim_feedforward=int(config.token_dim * config.mlp_ratio),
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=False,
    )
    return nn.TransformerEncoder(layer, num_layers=config.num_layers)


class _TemporalBackbone(nn.Module):
    def __init__(self, config: PPOHeadConfig):
        super().__init__()
        self.config = config
        input_dim = config.latent_dim + config.action_dim + config.state_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.token_dim),
            nn.GELU(),
            nn.Linear(config.token_dim, config.token_dim),
        )
        self.time_embedding = nn.Parameter(torch.zeros(1, config.action_horizon, config.token_dim))
        self.encoder = _build_encoder(config)
        self.output_norm = nn.LayerNorm(config.token_dim)

    def forward(self, policy_latent: torch.Tensor, base_action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if policy_latent.ndim == 2:
            policy_latent = policy_latent.unsqueeze(0)
        if base_action.ndim == 2:
            base_action = base_action.unsqueeze(0)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        batch, horizon, _ = policy_latent.shape
        if horizon != self.config.action_horizon:
            raise ValueError(f"Expected policy_latent horizon={self.config.action_horizon}, got {horizon}")
        if base_action.shape[:2] != (batch, horizon):
            raise ValueError(
                "Expected base_action shape to match policy_latent batch/horizon, "
                f"got {tuple(base_action.shape)} vs {tuple(policy_latent.shape)}"
            )

        state_tokens = state[:, None, :].expand(batch, horizon, self.config.state_dim)
        tokens = torch.cat([policy_latent, base_action, state_tokens], dim=-1)
        tokens = self.input_proj(tokens) + self.time_embedding[:, :horizon, :]
        tokens = self.encoder(tokens)
        return self.output_norm(tokens)


class TemporalActorHead(nn.Module):
    def __init__(self, config: PPOHeadConfig):
        super().__init__()
        self.config = config
        self.backbone = _TemporalBackbone(config)
        self.mean_head = nn.Sequential(
            nn.Linear(config.token_dim, config.token_dim),
            nn.GELU(),
            nn.Linear(config.token_dim, config.action_dim),
        )
        self.log_std = nn.Parameter(torch.full((config.action_horizon, config.action_dim), config.log_std_init))
        self.reset_output_to_zero()

    def reset_output_to_zero(self) -> None:
        final = self.mean_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, policy_latent: torch.Tensor, base_action: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.backbone(policy_latent, base_action, state)
        mean = self.mean_head(tokens)
        log_std = torch.clamp(self.log_std, self.config.min_log_std, self.config.max_log_std)
        log_std = log_std.unsqueeze(0).expand_as(mean)
        return mean, torch.exp(log_std)


class TemporalValueHead(nn.Module):
    def __init__(self, config: PPOHeadConfig):
        super().__init__()
        self.config = config
        self.backbone = _TemporalBackbone(config)
        self.value_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(config.action_horizon * config.token_dim),
            nn.Linear(config.action_horizon * config.token_dim, config.token_dim),
            nn.GELU(),
            nn.Linear(config.token_dim, 1),
        )
        self.reset_output_to_zero()

    def reset_output_to_zero(self) -> None:
        final = self.value_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, policy_latent: torch.Tensor, base_action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(policy_latent, base_action, state)
        return self.value_head(tokens).squeeze(-1)


def _resolve_checkpoint_path(path: str | Path | None, candidates: tuple[str, ...]) -> Path | None:
    if path is None:
        return None
    ckpt_path = Path(path).expanduser()
    if ckpt_path.is_dir():
        for candidate in candidates:
            candidate_path = ckpt_path / candidate
            if candidate_path.exists():
                return candidate_path
    return ckpt_path


def _extract_state_dict(payload: Any, preferred_keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in payload.keys()):
            tensor_values = [v for v in payload.values() if torch.is_tensor(v)]
            if tensor_values:
                return payload
    raise ValueError("Checkpoint does not contain a loadable PyTorch state_dict.")


def load_actor_head(path: str | Path | None, config: PPOHeadConfig, *, device: torch.device) -> TemporalActorHead:
    actor = TemporalActorHead(config).to(device)
    ckpt_path = _resolve_checkpoint_path(path, ("actor_head.pt", "ppo_actor_head.pt", "ppo_heads.pt"))
    if ckpt_path is not None:
        payload = torch.load(ckpt_path, map_location=device)
        actor.load_state_dict(_extract_state_dict(payload, ("actor", "actor_head", "state_dict", "model_state_dict")))
        logger.info("Loaded LeHome PPO actor head from %s on %s", ckpt_path, device)
    else:
        logger.info("Initialized LeHome PPO actor head from scratch on %s", device)
    actor.eval()
    return actor


def load_value_head(path: str | Path | None, config: PPOHeadConfig, *, device: torch.device) -> TemporalValueHead:
    value = TemporalValueHead(config).to(device)
    ckpt_path = _resolve_checkpoint_path(path, ("value_head.pt", "ppo_value_head.pt", "ppo_heads.pt"))
    if ckpt_path is not None:
        payload = torch.load(ckpt_path, map_location=device)
        value.load_state_dict(_extract_state_dict(payload, ("value", "value_head", "state_dict", "model_state_dict")))
        logger.info("Loaded LeHome PPO value head from %s on %s", ckpt_path, device)
    else:
        logger.info("Initialized LeHome PPO value head from scratch on %s", device)
    value.eval()
    return value


def to_numpy_scalar(value: torch.Tensor) -> np.ndarray:
    return np.asarray(value.detach().cpu(), dtype=np.float32).reshape(())
