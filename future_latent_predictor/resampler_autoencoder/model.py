from __future__ import annotations

import torch
from torch import nn


CAMERAS = ("top", "right_wrist", "left_wrist")


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        mlp_dim = int(embed_dim * mlp_ratio)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        queries = queries + attn_out
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class CameraResamplerEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_tokens: int,
        input_dim: int,
        latent_tokens: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.input_pos = nn.Parameter(torch.zeros(1, input_tokens, latent_dim))
        self.latent_queries = nn.Parameter(torch.randn(1, latent_tokens, latent_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    embed_dim=latent_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        context = self.input_proj(self.input_norm(tokens)) + self.input_pos
        queries = self.latent_queries.expand(tokens.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, context)
        return self.output_norm(queries)


class CameraResamplerDecoder(nn.Module):
    def __init__(
        self,
        *,
        output_tokens: int,
        output_dim: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.output_queries = nn.Parameter(torch.randn(1, output_tokens, latent_dim) * 0.02)
        self.latent_pos = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    embed_dim=latent_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, output_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        context = latents + self.latent_pos
        queries = self.output_queries.expand(latents.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, context)
        return self.output_proj(self.output_norm(queries))


class ResamplerAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        cameras: tuple[str, ...] = CAMERAS,
        input_tokens: int = 256,
        input_dim: int = 2048,
        latent_tokens: int = 24,
        latent_dim: int = 512,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cameras = tuple(cameras)
        self.input_tokens = input_tokens
        self.input_dim = input_dim
        self.latent_tokens = latent_tokens
        self.latent_dim = latent_dim

        self.camera_encoders = nn.ModuleDict(
            {
                camera: CameraResamplerEncoder(
                    input_tokens=input_tokens,
                    input_dim=input_dim,
                    latent_tokens=latent_tokens,
                    latent_dim=latent_dim,
                    num_layers=encoder_layers,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for camera in self.cameras
            }
        )
        self.camera_decoders = nn.ModuleDict(
            {
                camera: CameraResamplerDecoder(
                    output_tokens=input_tokens,
                    output_dim=input_dim,
                    latent_dim=latent_dim,
                    num_layers=decoder_layers,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for camera in self.cameras
            }
        )

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        latents = []
        for camera_index, camera in enumerate(self.cameras):
            latents.append(self.camera_encoders[camera](embeddings[:, camera_index]))
        return torch.stack(latents, dim=1)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        reconstructions = []
        for camera_index, camera in enumerate(self.cameras):
            reconstructions.append(self.camera_decoders[camera](latents[:, camera_index]))
        return torch.stack(reconstructions, dim=1)

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        latents = self.encode(embeddings)
        reconstructions = self.decode(latents)
        return {
            "latents": latents,
            "reconstructions": reconstructions,
        }


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())

