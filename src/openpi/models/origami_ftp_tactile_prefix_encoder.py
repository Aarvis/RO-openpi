from __future__ import annotations

import dataclasses
import math
from typing import Literal

from flax import linen as nn
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class FtpTactilePrefixConfig:
    image_size: int = 224
    patch_size: int = 16
    ftp_width: int = 768
    ftp_depth: int = 3
    ftp_heads: int = 12
    ftp_mlp_ratio: int = 4
    backbone_micro_batch: int = 64
    freeze_backbone: bool = True
    include_cls_token: bool = True
    normalize_images: bool = True
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    adapter_dim: int = 512
    prefix_dim: int = 2048
    tokens_per_finger: int = 4
    hands: int = 2
    fingers_per_hand: int = 5
    resampler_layers: int = 2
    resampler_heads: int = 8
    resampler_ffn_dim: int = 2048
    cross_finger_layers: int = 4
    cross_finger_heads: int = 8
    cross_finger_ffn_dim: int = 2048
    dropout: float = 0.1
    rope_enabled: bool = True
    rope_base: float = 10000.0
    rope_scale: float = 1.0
    hand_values: tuple[float, ...] = (-1.0, 1.0)
    finger_values: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    branch: Literal["auto", "deform", "full"] = "auto"

    @property
    def token_count(self) -> int:
        count = (self.image_size // self.patch_size) ** 2
        return count + 1 if self.include_cls_token else count

    @property
    def finger_count(self) -> int:
        return self.hands * self.fingers_per_hand

    @property
    def prefix_tokens(self) -> int:
        return self.finger_count * self.tokens_per_finger


def _axis_dimensions(head_dim: int, axes: int) -> list[int]:
    pairs = head_dim // 2
    base, remainder = divmod(pairs, axes)
    return [2 * (base + (1 if axis < remainder else 0)) for axis in range(axes)]


def apply_nd_rope(
    tensor: jax.Array,
    coordinates: jax.Array,
    *,
    base: float,
    scale: float,
) -> jax.Array:
    if coordinates.ndim == 2:
        coordinates = jnp.broadcast_to(coordinates[None, :, :], (tensor.shape[0],) + coordinates.shape)
    coordinates = coordinates.astype(jnp.float32)
    parts = []
    offset = 0
    for axis, width in enumerate(_axis_dimensions(tensor.shape[-1], coordinates.shape[-1])):
        section = tensor[..., offset : offset + width]
        inv_freq = jnp.power(
            jnp.asarray(base, dtype=jnp.float32),
            -jnp.arange(0, width, 2, dtype=jnp.float32) / float(width),
        )
        angles = coordinates[:, None, :, axis, None] * float(scale) * inv_freq
        cos = jnp.cos(angles).astype(tensor.dtype)
        sin = jnp.sin(angles).astype(tensor.dtype)
        even = section[..., 0::2]
        odd = section[..., 1::2]
        rotated = jnp.stack([even * cos - odd * sin, even * sin + odd * cos], axis=-1)
        parts.append(rotated.reshape(section.shape))
        offset += width
    if offset < tensor.shape[-1]:
        parts.append(tensor[..., offset:])
    return jnp.concatenate(parts, axis=-1)


def _attention_logits(q: jax.Array, k: jax.Array) -> jax.Array:
    batch, heads, query_tokens, head_dim = q.shape
    key_tokens = k.shape[2]
    q_flat = q.reshape(batch * heads, query_tokens, head_dim)
    k_flat = jnp.swapaxes(k, -1, -2).reshape(batch * heads, head_dim, key_tokens)
    return jnp.matmul(q_flat, k_flat).reshape(batch, heads, query_tokens, key_tokens)


def _attention_values(weights: jax.Array, v: jax.Array) -> jax.Array:
    batch, heads, query_tokens, key_tokens = weights.shape
    head_dim = v.shape[-1]
    weights_flat = weights.reshape(batch * heads, query_tokens, key_tokens)
    v_flat = v.reshape(batch * heads, key_tokens, head_dim)
    return jnp.matmul(weights_flat, v_flat).reshape(batch, heads, query_tokens, head_dim)


class FeedForward(nn.Module):
    width: int
    hidden_width: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, value: jax.Array, *, train: bool = False) -> jax.Array:
        value = nn.Dense(self.hidden_width, name="dense_in")(value)
        value = nn.gelu(value, approximate=False)
        value = nn.Dropout(self.dropout, name="dropout_in")(value, deterministic=not train)
        value = nn.Dense(self.width, name="dense_out")(value)
        return nn.Dropout(self.dropout, name="dropout_out")(value, deterministic=not train)


class RotaryMultiheadAttention(nn.Module):
    width: int
    heads: int
    rope_axes: int = 0
    rope_base: float = 10000.0
    rope_scale: float = 1.0
    dropout: float = 0.0

    def _split_heads(self, value: jax.Array) -> jax.Array:
        batch, tokens, _ = value.shape
        return value.reshape(batch, tokens, self.heads, self.width // self.heads).transpose(0, 2, 1, 3)

    def _merge_heads(self, value: jax.Array) -> jax.Array:
        batch, _, tokens, _ = value.shape
        return value.transpose(0, 2, 1, 3).reshape(batch, tokens, self.width)

    @nn.compact
    def __call__(
        self,
        query: jax.Array,
        memory: jax.Array | None = None,
        *,
        query_coordinates: jax.Array | None = None,
        key_coordinates: jax.Array | None = None,
        key_padding_mask: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        memory = query if memory is None else memory
        head_dim = self.width // self.heads
        q = self._split_heads(nn.Dense(self.width, name="q_proj")(query))
        k = self._split_heads(nn.Dense(self.width, name="k_proj")(memory))
        v = self._split_heads(nn.Dense(self.width, name="v_proj")(memory))
        if self.rope_axes:
            if query_coordinates is None:
                raise ValueError("RoPE attention requires query coordinates")
            if key_coordinates is None:
                key_coordinates = query_coordinates
            q = apply_nd_rope(q, query_coordinates, base=self.rope_base, scale=self.rope_scale)
            k = apply_nd_rope(k, key_coordinates, base=self.rope_base, scale=self.rope_scale)
        logits = _attention_logits(q, k).astype(jnp.float32) * (head_dim**-0.5)
        if key_padding_mask is not None:
            logits = jnp.where(key_padding_mask[:, None, None, :], jnp.asarray(-1.0e30, logits.dtype), logits)
        weights = nn.softmax(logits, axis=-1).astype(q.dtype)
        weights = nn.Dropout(self.dropout, name="attn_drop")(weights, deterministic=not train)
        output = _attention_values(weights, v)
        return nn.Dense(self.width, name="out_proj")(self._merge_heads(output))


class TransformerBlock(nn.Module):
    width: int
    heads: int
    ffn_width: int
    dropout: float
    rope_axes: int = 0
    rope_base: float = 10000.0
    rope_scale: float = 1.0

    @nn.compact
    def __call__(
        self,
        value: jax.Array,
        *,
        coordinates: jax.Array | None = None,
        key_padding_mask: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        normalized = nn.LayerNorm(epsilon=1e-5, name="norm_attn")(value)
        value = value + RotaryMultiheadAttention(
            self.width,
            self.heads,
            rope_axes=self.rope_axes,
            rope_base=self.rope_base,
            rope_scale=self.rope_scale,
            dropout=self.dropout,
            name="attn",
        )(
            normalized,
            query_coordinates=coordinates,
            key_padding_mask=key_padding_mask,
            train=train,
        )
        return value + FeedForward(self.width, self.ffn_width, self.dropout, name="ffn")(
            nn.LayerNorm(epsilon=1e-5, name="norm_ffn")(value),
            train=train,
        )


class CrossAttentionBlock(nn.Module):
    width: int
    heads: int
    ffn_width: int
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        query: jax.Array,
        memory: jax.Array,
        *,
        key_padding_mask: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        normalized_query = nn.LayerNorm(epsilon=1e-5, name="norm_query")(query)
        normalized_memory = nn.LayerNorm(epsilon=1e-5, name="norm_memory")(memory)
        query = query + RotaryMultiheadAttention(
            self.width,
            self.heads,
            dropout=self.dropout,
            name="cross_attn",
        )(
            normalized_query,
            normalized_memory,
            key_padding_mask=key_padding_mask,
            train=train,
        )
        normalized_self = nn.LayerNorm(epsilon=1e-5, name="norm_self")(query)
        query = query + RotaryMultiheadAttention(
            self.width,
            self.heads,
            dropout=self.dropout,
            name="self_attn",
        )(
            normalized_self,
            train=train,
        )
        return query + FeedForward(self.width, self.ffn_width, self.dropout, name="ffn")(
            nn.LayerNorm(epsilon=1e-5, name="norm_ffn")(query),
            train=train,
        )


class NativePatchEmbed(nn.Module):
    image_size: int
    patch_size: int
    width: int

    @nn.compact
    def __call__(self, images: jax.Array) -> jax.Array:
        tokens = nn.Conv(
            features=self.width,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="proj",
        )(images)
        return tokens.reshape(tokens.shape[0], -1, self.width)


class NativeMlp(nn.Module):
    width: int
    hidden_width: int

    @nn.compact
    def __call__(self, value: jax.Array) -> jax.Array:
        value = nn.Dense(self.hidden_width, name="fc1")(value)
        value = nn.gelu(value, approximate=False)
        return nn.Dense(self.width, name="fc2")(value)


class NativeViTAttention(nn.Module):
    width: int
    heads: int

    @nn.compact
    def __call__(self, value: jax.Array) -> jax.Array:
        batch, tokens, width = value.shape
        head_dim = width // self.heads
        qkv = nn.Dense(width * 3, name="qkv")(value)
        qkv = qkv.reshape(batch, tokens, 3, self.heads, head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = _attention_logits(q, k).astype(jnp.float32) * (head_dim**-0.5)
        attn = nn.softmax(logits, axis=-1).astype(q.dtype)
        value = _attention_values(attn, v)
        value = value.transpose(0, 2, 1, 3).reshape(batch, tokens, width)
        return nn.Dense(width, name="proj")(value)


class NativeViTBlock(nn.Module):
    width: int
    heads: int
    mlp_ratio: int

    @nn.compact
    def __call__(self, value: jax.Array) -> jax.Array:
        value = value + NativeViTAttention(self.width, self.heads, name="attn")(
            nn.LayerNorm(epsilon=1e-6, name="norm1")(value)
        )
        return value + NativeMlp(self.width, self.width * self.mlp_ratio, name="mlp")(
            nn.LayerNorm(epsilon=1e-6, name="norm2")(value)
        )


class SharpaWaveViTEncoder(nn.Module):
    cfg: FtpTactilePrefixConfig

    def setup(self) -> None:
        if not self.cfg.include_cls_token:
            raise ValueError("FTP SharpaWave encoder expects include_cls_token=True")
        self.patch_embed = NativePatchEmbed(
            self.cfg.image_size,
            self.cfg.patch_size,
            self.cfg.ftp_width,
            name="patch_embed",
        )

    @nn.compact
    def __call__(self, images: jax.Array) -> jax.Array:
        tokens = self.patch_embed(images)
        cls = self.param(
            "cls_token",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, 1, self.cfg.ftp_width),
        )
        pos = self.param(
            "pos_embed",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, self.cfg.token_count, self.cfg.ftp_width),
        )
        tokens = jnp.concatenate([jnp.broadcast_to(cls, (tokens.shape[0], 1, self.cfg.ftp_width)), tokens], axis=1)
        tokens = tokens + pos
        for index in range(self.cfg.ftp_depth):
            tokens = NativeViTBlock(
                self.cfg.ftp_width,
                self.cfg.ftp_heads,
                self.cfg.ftp_mlp_ratio,
                name=f"blocks_{index}",
            )(tokens)
        return tokens


class FrozenSharpaWaveBackbone(nn.Module):
    cfg: FtpTactilePrefixConfig

    def setup(self) -> None:
        self.encoder = SharpaWaveViTEncoder(self.cfg, name="encoder")

    def __call__(self, images: jax.Array) -> jax.Array:
        flat = images.reshape((-1,) + images.shape[-3:])
        flat = jnp.transpose(flat, (0, 2, 3, 1)).astype(jnp.float32)
        flat = jnp.where(jnp.max(flat) > 2.0, flat / 255.0, flat)
        if self.cfg.normalize_images:
            mean = jnp.asarray(self.cfg.image_mean, dtype=jnp.float32).reshape(1, 1, 1, 3)
            std = jnp.asarray(self.cfg.image_std, dtype=jnp.float32).reshape(1, 1, 1, 3)
            flat = (flat - mean) / std

        micro_batch = int(self.cfg.backbone_micro_batch)
        if micro_batch > 0 and flat.shape[0] > micro_batch:
            original_count = flat.shape[0]
            chunks = math.ceil(original_count / micro_batch)
            padded_count = chunks * micro_batch
            pad_count = padded_count - original_count
            flat = jnp.pad(flat, ((0, pad_count), (0, 0), (0, 0), (0, 0)))
            flat_chunks = flat.reshape((chunks, micro_batch) + flat.shape[1:])
            tokens = jax.lax.map(lambda chunk: self.encoder(chunk), flat_chunks)
            tokens = tokens.reshape((padded_count,) + tokens.shape[2:])[:original_count]
        else:
            tokens = self.encoder(flat)

        if self.cfg.freeze_backbone:
            tokens = jax.lax.stop_gradient(tokens)
        return tokens.reshape(images.shape[:-3] + tokens.shape[-2:])


class FeatureAdapter(nn.Module):
    output_dim: int
    dropout: float

    @nn.compact
    def __call__(self, value: jax.Array, *, train: bool = False) -> jax.Array:
        value = nn.LayerNorm(epsilon=1e-5, name="norm")(value)
        value = nn.Dense(self.output_dim, name="dense_in")(value)
        value = nn.gelu(value, approximate=False)
        value = nn.Dropout(self.dropout, name="dropout")(value, deterministic=not train)
        return nn.Dense(self.output_dim, name="dense_out")(value)


def canonical_layout(cfg: FtpTactilePrefixConfig) -> tuple[jax.Array, jax.Array, jax.Array]:
    if len(cfg.hand_values) != cfg.hands or len(cfg.finger_values) != cfg.fingers_per_hand:
        raise ValueError("canonical coordinates must match hand/finger counts")
    hand_ids = []
    finger_ids = []
    coordinates = []
    for hand in range(cfg.hands):
        for finger in range(cfg.fingers_per_hand):
            hand_ids.append(hand)
            finger_ids.append(finger)
            for _ in range(cfg.tokens_per_finger):
                coordinates.append([cfg.hand_values[hand], cfg.finger_values[finger]])
    return (
        jnp.asarray(hand_ids, dtype=jnp.int32),
        jnp.asarray(finger_ids, dtype=jnp.int32),
        jnp.asarray(coordinates, dtype=jnp.float32),
    )


class PerFingerResampler(nn.Module):
    cfg: FtpTactilePrefixConfig

    @nn.compact
    def __call__(self, memory: jax.Array, key_padding_mask: jax.Array, *, train: bool = False) -> jax.Array:
        batch, fingers, _, width = memory.shape
        slot_queries = self.param(
            "slot_queries",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, self.cfg.tokens_per_finger, width),
        )
        hand_ids, finger_ids, _ = canonical_layout(self.cfg)
        hand_bias = nn.Embed(self.cfg.hands, width, name="hand_embedding")(hand_ids)
        finger_bias = nn.Embed(self.cfg.fingers_per_hand, width, name="finger_embedding")(finger_ids)
        queries = jnp.broadcast_to(slot_queries, (batch, fingers, self.cfg.tokens_per_finger, width))
        queries = queries + hand_bias[None, :, None, :] + finger_bias[None, :, None, :]
        queries = queries.reshape(batch * fingers, self.cfg.tokens_per_finger, width)
        flat_memory = memory.reshape(batch * fingers, memory.shape[2], width)
        flat_mask = key_padding_mask.reshape(batch * fingers, key_padding_mask.shape[-1])
        for index in range(self.cfg.resampler_layers):
            queries = CrossAttentionBlock(
                width,
                self.cfg.resampler_heads,
                self.cfg.resampler_ffn_dim,
                self.cfg.dropout,
                name=f"layers_{index}",
            )(
                queries,
                flat_memory,
                key_padding_mask=flat_mask,
                train=train,
            )
        return queries.reshape(batch, fingers, self.cfg.tokens_per_finger, width)


class CrossFingerTransformer(nn.Module):
    cfg: FtpTactilePrefixConfig

    @nn.compact
    def __call__(self, tokens: jax.Array, coordinates: jax.Array, *, train: bool = False) -> jax.Array:
        rope_axes = 2 if self.cfg.rope_enabled else 0
        for index in range(self.cfg.cross_finger_layers):
            tokens = TransformerBlock(
                self.cfg.adapter_dim,
                self.cfg.cross_finger_heads,
                self.cfg.cross_finger_ffn_dim,
                dropout=self.cfg.dropout,
                rope_axes=rope_axes,
                rope_base=self.cfg.rope_base,
                rope_scale=self.cfg.rope_scale,
                name=f"layers_{index}",
            )(
                tokens,
                coordinates=coordinates,
                train=train,
            )
        return nn.LayerNorm(epsilon=1e-5, name="norm")(tokens)


class PrefixProjection(nn.Module):
    cfg: FtpTactilePrefixConfig

    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        value = nn.LayerNorm(epsilon=1e-5, name="norm_in")(tokens)
        value = nn.Dense(self.cfg.prefix_dim, name="dense_in")(value)
        value = nn.gelu(value, approximate=False)
        value = nn.Dense(self.cfg.prefix_dim, name="dense_out")(value)
        value = nn.LayerNorm(epsilon=1e-5, name="norm_out")(value)
        token_type = self.param(
            "token_type",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, 1, self.cfg.prefix_dim),
        )
        return value + token_type


def _availability(mask: jax.Array | None, batch: int) -> jax.Array:
    if mask is None:
        return jnp.ones((batch,), dtype=bool)
    mask = jnp.asarray(mask, dtype=bool)
    if mask.ndim == 0:
        mask = jnp.broadcast_to(mask[None], (batch,))
    elif mask.ndim == 2 and mask.shape[-1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 1:
        raise ValueError(f"Expected raw availability mask shape [B] or [B, 1], got {mask.shape}")
    return mask


class OrigamiFtpTactilePrefixEncoder(nn.Module):
    cfg: FtpTactilePrefixConfig

    def setup(self) -> None:
        self.backbone = FrozenSharpaWaveBackbone(self.cfg, name="backbone")
        self.deform_adapter = FeatureAdapter(self.cfg.adapter_dim, self.cfg.dropout, name="deform_adapter")
        self.raw_adapter = FeatureAdapter(self.cfg.adapter_dim, self.cfg.dropout, name="raw_adapter")
        self.resampler = PerFingerResampler(self.cfg, name="resampler")
        self.cross_finger = CrossFingerTransformer(self.cfg, name="cross_finger")
        self.prefix_projection = PrefixProjection(self.cfg, name="prefix_projection")

    def _encode_view(
        self,
        deform_features: jax.Array,
        raw_features: jax.Array,
        raw_input_available: jax.Array,
        *,
        use_raw: bool,
        train: bool,
    ) -> jax.Array:
        batch, fingers, token_count, _ = deform_features.shape
        deform_memory = self.deform_adapter(deform_features, train=train)
        raw_memory = self.raw_adapter(raw_features, train=train)
        memory = jnp.concatenate([deform_memory, raw_memory], axis=2)
        deform_mask = jnp.zeros((batch, fingers, token_count), dtype=bool)
        if use_raw:
            raw_mask = jnp.broadcast_to(~raw_input_available[:, None, None], (batch, fingers, token_count))
        else:
            raw_mask = jnp.ones((batch, fingers, token_count), dtype=bool)
        key_padding_mask = jnp.concatenate([deform_mask, raw_mask], axis=2)
        _, _, coords = canonical_layout(self.cfg)
        coords = jnp.broadcast_to(coords[None, :, :], (batch, self.cfg.prefix_tokens, 2))
        finger_tokens = self.resampler(memory, key_padding_mask, train=train)
        content = finger_tokens.reshape(batch, self.cfg.prefix_tokens, self.cfg.adapter_dim)
        content = self.cross_finger(content, coords, train=train)
        return self.prefix_projection(content)

    def __call__(
        self,
        deform_images: jax.Array,
        raw_images: jax.Array | None = None,
        raw_available: jax.Array | None = None,
        *,
        train: bool = False,
    ) -> jax.Array:
        if deform_images.shape[-4] != self.cfg.finger_count:
            raise ValueError(f"Expected {self.cfg.finger_count} tactile crops, got shape {deform_images.shape}")
        if deform_images.shape[-3:] != (3, self.cfg.image_size, self.cfg.image_size):
            raise ValueError(
                "Expected tactile crops with shape [..., 3, "
                f"{self.cfg.image_size}, {self.cfg.image_size}], got {deform_images.shape}"
            )
        batch = deform_images.shape[0]
        if raw_images is None:
            raw_images = jnp.zeros_like(deform_images)
            raw_available = jnp.zeros((batch,), dtype=bool)
        raw_input_available = _availability(raw_available, batch)

        deform_features = self.backbone(deform_images)
        raw_features = self.backbone(raw_images)
        deform_prefix = self._encode_view(
            deform_features,
            raw_features,
            raw_input_available,
            use_raw=False,
            train=train,
        )
        if self.cfg.branch == "deform":
            return deform_prefix

        full_prefix = self._encode_view(
            deform_features,
            raw_features,
            raw_input_available,
            use_raw=True,
            train=train,
        )
        if self.cfg.branch == "full":
            return full_prefix
        return jnp.where(raw_input_available[:, None, None], full_prefix, deform_prefix)
