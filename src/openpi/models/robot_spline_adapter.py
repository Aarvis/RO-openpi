from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def _apply_scalar_rope(
    x: jax.Array,
    positions: jax.Array,
    *,
    base: float,
) -> jax.Array:
    """Applies scalar RoPE positions [B, N] to x [B, N, H, D]."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE chunk dimension must be even, got {x.shape[-1]}")

    half_dim = x.shape[-1] // 2
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(half_dim, dtype=jnp.float32)
    timescale = float(base) ** freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return out.astype(x.dtype)


def _width_fourier_features(widths: jax.Array, num_bands: int) -> jax.Array:
    scales = (2.0 ** jnp.arange(num_bands, dtype=jnp.float32)) * jnp.pi
    phases = widths[..., None] * scales[None, None, :]
    return jnp.concatenate([widths[..., None], jnp.sin(phases), jnp.cos(phases)], axis=-1)


class RobotSplineSelfAttentionBlock(nn.Module):
    model_dim: int
    num_heads: int
    ffn_dim: int
    rope_base: float = 10_000.0

    @nn.compact
    def __call__(self, x: jax.Array, geometry: jax.Array) -> jax.Array:
        residual = x
        y = nn.LayerNorm()(x)

        qkv = nn.Dense(3 * self.model_dim, use_bias=False, param_dtype=jnp.float32, name="qkv")(y)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        batch_size, seq_len, _ = q.shape
        head_dim = self.model_dim // self.num_heads
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if head_dim % 4 != 0:
            raise ValueError(
                f"Per-head dimension ({head_dim}) must be divisible by 4 for 4D spline RoPE."
            )

        rope_chunk_dim = head_dim // 4
        if rope_chunk_dim % 2 != 0:
            raise ValueError(
                f"Per-geometry RoPE chunk dimension ({rope_chunk_dim}) must be even."
            )

        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, head_dim)

        q_chunks = list(jnp.split(q, 4, axis=-1))
        k_chunks = list(jnp.split(k, 4, axis=-1))
        for geometry_index in range(4):
            positions = geometry[..., geometry_index]
            q_chunks[geometry_index] = _apply_scalar_rope(
                q_chunks[geometry_index],
                positions,
                base=self.rope_base,
            )
            k_chunks[geometry_index] = _apply_scalar_rope(
                k_chunks[geometry_index],
                positions,
                base=self.rope_base,
            )
        q = jnp.concatenate(q_chunks, axis=-1)
        k = jnp.concatenate(k_chunks, axis=-1)

        scale = head_dim ** -0.5
        attn_logits = jnp.einsum("bnhd,bmhd->bhnm", q, k, precision=jax.lax.Precision.HIGHEST) * scale
        attn_weights = nn.softmax(attn_logits.astype(jnp.float32), axis=-1).astype(q.dtype)
        attn_out = jnp.einsum("bhnm,bmhd->bnhd", attn_weights, v, precision=jax.lax.Precision.HIGHEST)
        attn_out = attn_out.reshape(batch_size, seq_len, self.model_dim)
        attn_out = nn.Dense(self.model_dim, use_bias=False, param_dtype=jnp.float32, name="attn_out")(attn_out)
        x = residual + attn_out

        residual = x
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.ffn_dim, param_dtype=jnp.float32, name="ffn_in")(y)
        y = nn.gelu(y)
        y = nn.Dense(self.model_dim, param_dtype=jnp.float32, name="ffn_out")(y)
        return residual + y


class RobotSplinePrefixAdapter(nn.Module):
    control_point_dim: int
    control_count: int
    degree: int
    model_dim: int
    output_dim: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    width_fourier_bands: int = 8
    width_hidden_dim: int = 512
    rope_base: float = 10_000.0

    def setup(self) -> None:
        if self.degree != 3:
            raise ValueError(f"RobotSplinePrefixAdapter currently expects cubic splines, got degree={self.degree}")

    def _geometry(self, knots: jax.Array) -> tuple[jax.Array, jax.Array]:
        expected_knot_count = self.control_count + self.degree + 1
        if knots.shape[-1] != expected_knot_count:
            raise ValueError(
                f"Expected knot count {expected_knot_count}, got {knots.shape[-1]} "
                f"for control_count={self.control_count}, degree={self.degree}"
            )

        left = knots[:, : self.control_count]
        right = knots[:, self.degree + 1 : self.degree + 1 + self.control_count]
        midpoint = 0.5 * (left + right)
        greville = (
            knots[:, 1 : self.control_count + 1]
            + knots[:, 2 : self.control_count + 2]
            + knots[:, 3 : self.control_count + 3]
        ) / 3.0
        widths = jnp.clip(right - left, a_min=1e-6)
        geometry = jnp.stack([left, right, midpoint, greville], axis=-1)
        return geometry, widths

    @nn.compact
    def __call__(self, coefficients: jax.Array, knots: jax.Array, *, train: bool = False) -> jax.Array:
        del train

        if coefficients.shape[-2] != self.control_count:
            raise ValueError(
                f"Expected {self.control_count} spline control points, got {coefficients.shape[-2]}"
            )
        if coefficients.shape[-1] != self.control_point_dim:
            raise ValueError(
                f"Expected control-point dimension {self.control_point_dim}, got {coefficients.shape[-1]}"
            )

        geometry, widths = self._geometry(knots)

        content_tokens = nn.Dense(self.model_dim, param_dtype=jnp.float32, name="content_proj")(coefficients)

        width_features = _width_fourier_features(widths, self.width_fourier_bands)
        width_tokens = nn.Dense(self.width_hidden_dim, param_dtype=jnp.float32, name="width_in")(width_features)
        width_tokens = nn.gelu(width_tokens)
        width_tokens = nn.Dense(self.model_dim, param_dtype=jnp.float32, name="width_out")(width_tokens)

        modality_embedding = self.param(
            "spline_modality_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.model_dim,),
        )
        tokens = content_tokens + width_tokens + modality_embedding[None, None, :]
        tokens = nn.LayerNorm(name="input_norm")(tokens)

        for layer_index in range(self.num_layers):
            tokens = RobotSplineSelfAttentionBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                rope_base=self.rope_base,
                name=f"block_{layer_index}",
            )(tokens, geometry)

        tokens = nn.LayerNorm(name="output_norm")(tokens)
        return nn.Dense(self.output_dim, param_dtype=jnp.float32, name="prefix_proj")(tokens)
