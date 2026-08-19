from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


TACTILE_NORM_STATS_FILENAME = "origami_tactile_norm_stats.json"


@dataclasses.dataclass(frozen=True)
class OrigamiTactileNormStats:
    center: tuple[float, ...]
    scale: tuple[float, ...]
    quantile_low: tuple[float, ...]
    quantile_high: tuple[float, ...]
    quantile_low_value: float
    quantile_high_value: float
    min_scale: float


def save_tactile_norm_stats(directory: str | Path, stats: OrigamiTactileNormStats) -> Path:
    path = Path(directory) / TACTILE_NORM_STATS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "center": list(stats.center),
        "scale": list(stats.scale),
        "quantile_low": list(stats.quantile_low),
        "quantile_high": list(stats.quantile_high),
        "quantile_low_value": float(stats.quantile_low_value),
        "quantile_high_value": float(stats.quantile_high_value),
        "min_scale": float(stats.min_scale),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_tactile_norm_stats(path: str | Path) -> OrigamiTactileNormStats:
    payload = json.loads(Path(path).read_text())
    if "center" not in payload or "scale" not in payload:
        raise KeyError(f"Tactile stats file is missing required keys: {path}")
    return OrigamiTactileNormStats(
        center=tuple(float(x) for x in payload["center"]),
        scale=tuple(float(x) for x in payload["scale"]),
        quantile_low=tuple(float(x) for x in payload.get("quantile_low", payload["center"])),
        quantile_high=tuple(float(x) for x in payload.get("quantile_high", payload["center"])),
        quantile_low_value=float(payload.get("quantile_low_value", 0.005)),
        quantile_high_value=float(payload.get("quantile_high_value", 0.995)),
        min_scale=float(payload.get("min_scale", 1.0e-6)),
    )


class _SharedFingerEncoder(nn.Module):
    output_dim: int
    hidden_dims: tuple[int, ...] = (64, 128)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        y = x
        for index, hidden_dim in enumerate(self.hidden_dims):
            y = nn.Dense(hidden_dim, param_dtype=jnp.float32, name=f"dense_{index}")(y)
            y = nn.gelu(y)
        y = nn.Dense(self.output_dim, param_dtype=jnp.float32, name="dense_out")(y)
        return nn.LayerNorm(name="encoder_norm")(y)


class _TactileSelfAttentionBlock(nn.Module):
    model_dim: int
    num_heads: int
    ffn_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"tactile model_dim ({self.model_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        head_dim = self.model_dim // self.num_heads

        residual = x
        y = nn.LayerNorm(name="attn_norm")(x)
        qkv = nn.Dense(3 * self.model_dim, use_bias=False, param_dtype=jnp.float32, name="qkv")(y)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        batch_size, seq_len, _ = q.shape
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, head_dim)

        scale = head_dim**-0.5
        attn_logits = jnp.einsum("bnhd,bmhd->bhnm", q, k, precision=jax.lax.Precision.HIGHEST) * scale
        attn_weights = nn.softmax(attn_logits.astype(jnp.float32), axis=-1).astype(q.dtype)
        attn_out = jnp.einsum("bhnm,bmhd->bnhd", attn_weights, v, precision=jax.lax.Precision.HIGHEST)
        attn_out = attn_out.reshape(batch_size, seq_len, self.model_dim)
        attn_out = nn.Dense(self.model_dim, use_bias=False, param_dtype=jnp.float32, name="attn_out")(attn_out)
        x = residual + attn_out

        residual = x
        y = nn.LayerNorm(name="ffn_norm")(x)
        y = nn.Dense(self.ffn_dim, param_dtype=jnp.float32, name="ffn_in")(y)
        y = nn.gelu(y)
        y = nn.Dense(self.model_dim, param_dtype=jnp.float32, name="ffn_out")(y)
        return residual + y


class OrigamiTactilePrefixAdapter(nn.Module):
    tactile_dim: int = 60
    finger_count: int = 10
    channels_per_finger: int = 6
    token_dim: int = 256
    output_dim: int = 2048
    finger_hidden_dims: tuple[int, ...] = (64, 128)
    transformer_layers: int = 2
    attention_heads: int = 4
    ffn_dim: int = 512
    use_type_embeddings: bool = True
    tanh_scale: float = 5.0
    min_scale: float = 1.0e-6
    norm_center: tuple[float, ...] | None = None
    norm_scale: tuple[float, ...] | None = None

    def setup(self) -> None:
        expected_dim = self.finger_count * self.channels_per_finger
        if self.tactile_dim != expected_dim:
            raise ValueError(
                f"tactile_dim must equal finger_count * channels_per_finger ({expected_dim}), got {self.tactile_dim}"
            )

    def _normalize(self, tactile: jax.Array) -> jax.Array:
        if self.norm_center is None or self.norm_scale is None:
            raise ValueError("Origami tactile conditioning is enabled, but tactile normalization stats were not loaded.")
        center = jnp.asarray(self.norm_center, dtype=jnp.float32)
        scale = jnp.maximum(jnp.asarray(self.norm_scale, dtype=jnp.float32), jnp.asarray(self.min_scale, dtype=jnp.float32))
        z = (jnp.asarray(tactile, dtype=jnp.float32) - center[None, :]) / scale[None, :]
        return jnp.asarray(self.tanh_scale, dtype=jnp.float32) * jnp.tanh(z / self.tanh_scale)

    def _pool_query(self, tokens: jax.Array, *, query_name: str) -> jax.Array:
        query = self.param(query_name, nn.initializers.normal(stddev=0.02), (self.token_dim,))
        logits = jnp.einsum("d,bkd->bk", query, tokens, precision=jax.lax.Precision.HIGHEST)
        logits = logits * (self.token_dim**-0.5)
        weights = nn.softmax(logits.astype(jnp.float32), axis=-1).astype(tokens.dtype)
        return jnp.einsum("bk,bkd->bd", weights, tokens, precision=jax.lax.Precision.HIGHEST)

    @nn.compact
    def __call__(self, tactile: jax.Array, *, train: bool = False) -> jax.Array:
        del train
        if tactile.shape[-1] != self.tactile_dim:
            raise ValueError(f"Expected tactile dim {self.tactile_dim}, got {tactile.shape[-1]}")

        normalized = self._normalize(tactile)
        finger_wrenches = normalized.reshape(normalized.shape[0], self.finger_count, self.channels_per_finger)
        tokens = _SharedFingerEncoder(
            output_dim=self.token_dim,
            hidden_dims=self.finger_hidden_dims,
            name="finger_encoder",
        )(finger_wrenches)

        hand_embeddings = self.param(
            "hand_embeddings",
            nn.initializers.normal(stddev=0.02),
            (2, self.token_dim),
        )
        finger_type_embeddings = self.param(
            "finger_type_embeddings",
            nn.initializers.normal(stddev=0.02),
            (5, self.token_dim),
        )
        hand_indices = jnp.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=jnp.int32)
        finger_type_indices = jnp.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=jnp.int32)
        tokens = tokens + hand_embeddings[hand_indices][None, :, :] + finger_type_embeddings[finger_type_indices][None, :, :]
        tokens = nn.LayerNorm(name="input_norm")(tokens)

        for layer_index in range(self.transformer_layers):
            tokens = _TactileSelfAttentionBlock(
                model_dim=self.token_dim,
                num_heads=self.attention_heads,
                ffn_dim=self.ffn_dim,
                name=f"block_{layer_index}",
            )(tokens)

        tokens = nn.LayerNorm(name="context_norm")(tokens)
        left_token = self._pool_query(tokens[:, :5, :], query_name="left_query")
        right_token = self._pool_query(tokens[:, 5:, :], query_name="right_query")
        global_token = self._pool_query(tokens, query_name="global_query")
        pooled = jnp.stack([left_token, right_token, global_token], axis=1)

        pooled = nn.LayerNorm(name="project_in_norm")(pooled)
        pooled = nn.Dense(512, param_dtype=jnp.float32, name="project_in")(pooled)
        pooled = nn.gelu(pooled)
        pooled = nn.Dense(self.output_dim, param_dtype=jnp.float32, name="project_out")(pooled)

        if self.use_type_embeddings:
            type_embeddings = self.param(
                "tactile_token_type_embeddings",
                nn.initializers.normal(stddev=0.02),
                (3, self.output_dim),
            )
            pooled = pooled + type_embeddings[None, :, :]
        return nn.LayerNorm(name="tactile_token_norm")(pooled)
