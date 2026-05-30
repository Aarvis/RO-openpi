from __future__ import annotations

import torch
from torch import nn


class StateTokenizer(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        latent_dim: int,
        num_state_tokens: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_state_tokens = num_state_tokens
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_state_tokens * latent_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        tokens = self.net(state.float())
        return tokens.reshape(state.shape[0], self.num_state_tokens, self.latent_dim)


class FutureLatentPredictor(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        num_cameras: int = 3,
        latent_tokens: int = 24,
        latent_dim: int = 512,
        num_state_tokens: int = 4,
        state_hidden_dim: int = 1024,
        transformer_layers: int = 6,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        predict_residual: bool = True,
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.latent_tokens = latent_tokens
        self.latent_dim = latent_dim
        self.num_state_tokens = num_state_tokens
        self.predict_residual = predict_residual

        self.state_tokenizer = StateTokenizer(
            state_dim=state_dim,
            latent_dim=latent_dim,
            num_state_tokens=num_state_tokens,
            hidden_dim=state_hidden_dim,
            dropout=dropout,
        )
        self.horizon_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.camera_pos = nn.Parameter(torch.zeros(1, num_cameras, 1, latent_dim))
        self.token_pos = nn.Parameter(torch.zeros(1, 1, latent_tokens, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=int(latent_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.output_norm = nn.LayerNorm(latent_dim)
        self.delta_head = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, current_latents: torch.Tensor, state: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = current_latents.shape[0]
        latent_tokens = current_latents + self.camera_pos + self.token_pos
        latent_sequence = latent_tokens.reshape(batch_size, self.num_cameras * self.latent_tokens, self.latent_dim)

        state_tokens = self.state_tokenizer(state)
        horizon_tokens = self.horizon_token.expand(batch_size, -1, -1)
        sequence = torch.cat([horizon_tokens, state_tokens, latent_sequence], dim=1)
        transformed = self.transformer(sequence)

        latent_start = 1 + self.num_state_tokens
        latent_output = transformed[:, latent_start:]
        delta = self.delta_head(self.output_norm(latent_output))
        delta = delta.reshape(batch_size, self.num_cameras, self.latent_tokens, self.latent_dim)
        prediction = current_latents + delta if self.predict_residual else delta
        return {
            "prediction": prediction,
            "delta": delta,
        }


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())

