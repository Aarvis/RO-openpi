import logging
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.origami_planner_adapter as _origami_planner_adapter
import openpi.models.origami_ftp_tactile_prefix_encoder as _origami_ftp_tactile_prefix_encoder
import openpi.models.origami_spline_losses as _origami_spline_losses
import openpi.models.origami_tactile_adapter as _origami_tactile_adapter
import openpi.models.robot_spline_adapter as _robot_spline_adapter
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize

logger = logging.getLogger("openpi")


def _safe_cumsum(values: at.Array, *, axis: int) -> at.Array:
    """Compute cumulative sums without XLA's reduce-window cumsum lowering."""
    values = jnp.moveaxis(values, axis, 0)
    init = jnp.zeros(values.shape[1:], dtype=values.dtype)

    def step(carry, value):
        carry = carry + value
        return carry, carry

    _, out = jax.lax.scan(step, init, values)
    return jnp.moveaxis(out, 0, axis)


def _positions_from_input_mask(input_mask: at.Array) -> at.Array:
    return _safe_cumsum(jnp.asarray(input_mask, dtype=jnp.int32), axis=1) - 1


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    input_mask = jnp.asarray(input_mask, dtype=jnp.bool_)
    mask_ar = jnp.asarray(mask_ar, dtype=jnp.bool_)
    if mask_ar.ndim == 1:
        cumsum = _safe_cumsum(mask_ar.astype(jnp.int32), axis=0)
        attn_mask = cumsum[None, None, :] <= cumsum[None, :, None]
    else:
        mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
        cumsum = _safe_cumsum(mask_ar.astype(jnp.int32), axis=1)
        attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = jnp.logical_and(input_mask[:, None, :], input_mask[:, :, None])
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class FutureLatentPolicyAdapter(nnx.Module):
    """Projects compact future image latents into the pi0.5 prefix token space."""

    def __init__(self, *, latent_dim: int, hidden_dim: int, output_dim: int, rngs: nnx.Rngs):
        self.input_proj = nnx.Linear(latent_dim, hidden_dim, rngs=rngs)
        self.output_proj = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, latents: at.Array, valid_mask: at.Array) -> tuple[at.Array, at.Array]:
        batch_size, num_cameras, latent_tokens, latent_dim = latents.shape
        tokens = jnp.reshape(latents, (batch_size, num_cameras * latent_tokens, latent_dim))
        tokens = self.input_proj(tokens)
        tokens = nnx.swish(tokens)
        tokens = self.output_proj(tokens)
        token_mask = einops.repeat(valid_mask, "b c -> b (c t)", t=latent_tokens)
        return tokens, token_mask


def _load_origami_action_stats(
    stats_dir: str | None,
) -> _origami_spline_losses.PackedActionNormStats | None:
    if not stats_dir:
        return None
    try:
        loaded = _normalize.load(_download.maybe_download(stats_dir))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Origami VLA action normalization stats were not found under {stats_dir}. "
            "Run scripts/compute_origami_vla_norm_stats.py before training."
        ) from exc
    required_keys = ("actions_control_points", "actions_span_widths")
    missing = [key for key in required_keys if key not in loaded]
    if missing:
        raise KeyError(
            f"Missing Origami packed-action normalization stats {missing} under {stats_dir}. "
            "Re-run scripts/compute_origami_vla_norm_stats.py to generate separate control-point and span stats."
        )

    def _convert(stats: _normalize.NormStats) -> _origami_spline_losses.ActionNormStats:
        def _freeze(values: np.ndarray | None) -> tuple[float, ...] | None:
            if values is None:
                return None
            return tuple(np.asarray(values, dtype=np.float32).reshape(-1).tolist())

        return _origami_spline_losses.ActionNormStats(
            # Keep these as immutable host-side constants. Static JAX arrays traced during
            # jitted model init can leak out through NNX graph metadata.
            mean=_freeze(stats.mean),
            std=_freeze(stats.std),
            q01=_freeze(stats.q01),
            q99=_freeze(stats.q99),
        )

    logger.info("Loaded Origami packed action normalization stats from %s", stats_dir)
    return _origami_spline_losses.PackedActionNormStats(
        control_points=_convert(loaded["actions_control_points"]),
        span_widths=_convert(loaded["actions_span_widths"]),
    )


def _load_origami_tactile_stats(
    stats_dir: str | None,
) -> _origami_tactile_adapter.OrigamiTactileNormStats | None:
    if not stats_dir:
        return None
    resolved_dir = _download.maybe_download(stats_dir)
    path = resolved_dir / _origami_tactile_adapter.TACTILE_NORM_STATS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Origami tactile normalization stats were not found at {path}. "
            "Run scripts/compute_origami_vla_norm_stats.py before training."
        )
    logger.info("Loaded Origami tactile normalization stats from %s", path)
    return _origami_tactile_adapter.load_tactile_norm_stats(path)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.future_latent_config = config.future_latent
        self.robot_spline_config = config.robot_spline
        self.origami_vla_config = config.origami_vla
        self.origami_action_stats = (
            _load_origami_action_stats(config.origami_vla.action_norm_stats_dir)
            if config.origami_vla.enabled
            and config.origami_vla.action_mode == "spline"
            and not config.origami_vla.disable_auxiliary_losses
            else None
        )
        self.origami_tactile_stats = (
            _load_origami_tactile_stats(config.origami_vla.action_norm_stats_dir)
            if config.origami_vla.enabled and config.origami_vla.tactile_enabled
            else None
        )
        self.image_keys = config.image_keys if not (config.robot_spline.enabled and not config.robot_spline.use_image_prefix) else ()
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(jnp.zeros((1, *_model.IMAGE_RESOLUTION, 3), dtype=jnp.float32), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.future_latent.enabled:
            self.future_latent_adapter = FutureLatentPolicyAdapter(
                latent_dim=config.future_latent.latent_dim,
                hidden_dim=config.future_latent.adapter_hidden_dim,
                output_dim=paligemma_config.width,
                rngs=rngs,
            )
        if config.robot_spline.enabled:
            self.robot_spline_adapter = nnx_bridge.ToNNX(
                _robot_spline_adapter.RobotSplinePrefixAdapter(
                    control_point_dim=config.robot_spline.control_point_dim,
                    control_count=config.robot_spline.control_count,
                    degree=config.robot_spline.degree,
                    model_dim=config.robot_spline.model_dim,
                    output_dim=paligemma_config.width,
                    num_layers=config.robot_spline.num_layers,
                    num_heads=config.robot_spline.num_heads,
                    ffn_dim=config.robot_spline.ffn_dim,
                    width_fourier_bands=config.robot_spline.width_fourier_bands,
                    width_hidden_dim=config.robot_spline.width_hidden_dim,
                    rope_base=config.robot_spline.rope_base,
                )
            )
            self.robot_spline_adapter.lazy_init(
                jnp.zeros(
                    (1, config.robot_spline.control_count, config.robot_spline.control_point_dim),
                    dtype=jnp.float32,
                ),
                jnp.zeros((1, config.robot_spline.knot_count), dtype=jnp.float32),
                train=False,
                rngs=rngs,
            )
        if config.origami_vla.enabled:
            self.origami_planner_adapter = nnx_bridge.ToNNX(
                _origami_planner_adapter.OrigamiPlannerPrefixAdapter(
                    belief_dim=config.origami_vla.belief_dim,
                    history_dim=config.origami_vla.history_dim,
                    output_dim=paligemma_config.width,
                    belief_hidden_dims=config.origami_vla.planner_belief_hidden_dims,
                    progress_hidden_dims=config.origami_vla.planner_progress_hidden_dims,
                    uncertainty_hidden_dims=config.origami_vla.planner_uncertainty_hidden_dims,
                    history_hidden_dims=config.origami_vla.planner_history_hidden_dims,
                    use_type_embeddings=config.origami_vla.planner_use_type_embeddings,
                )
            )
            self.origami_planner_adapter.lazy_init(
                jnp.zeros((1, config.origami_vla.belief_dim), dtype=jnp.float32),
                jnp.zeros((1, 2), dtype=jnp.float32),
                jnp.zeros((1, 3), dtype=jnp.float32),
                jnp.zeros((1, config.origami_vla.history_dim), dtype=jnp.float32),
                train=False,
                rngs=rngs,
            )
            if config.origami_vla.ftp_tactile_enabled:
                if config.origami_vla.ftp_tactile_prefix_dim != paligemma_config.width:
                    raise ValueError(
                        "ftp_tactile_prefix_dim must match the PaliGemma prefix width, "
                        f"got {config.origami_vla.ftp_tactile_prefix_dim} and {paligemma_config.width}."
                    )
                ftp_tactile_cfg = _origami_ftp_tactile_prefix_encoder.FtpTactilePrefixConfig(
                    image_size=config.origami_vla.ftp_tactile_image_size,
                    patch_size=config.origami_vla.ftp_tactile_patch_size,
                    ftp_width=config.origami_vla.ftp_tactile_width,
                    ftp_depth=config.origami_vla.ftp_tactile_depth,
                    ftp_heads=config.origami_vla.ftp_tactile_heads,
                    ftp_mlp_ratio=config.origami_vla.ftp_tactile_mlp_ratio,
                    backbone_micro_batch=config.origami_vla.ftp_tactile_backbone_micro_batch,
                    freeze_backbone=config.origami_vla.ftp_tactile_freeze_backbone,
                    include_cls_token=config.origami_vla.ftp_tactile_include_cls_token,
                    normalize_images=config.origami_vla.ftp_tactile_normalize_images,
                    image_mean=config.origami_vla.ftp_tactile_image_mean,
                    image_std=config.origami_vla.ftp_tactile_image_std,
                    adapter_dim=config.origami_vla.ftp_tactile_adapter_dim,
                    prefix_dim=config.origami_vla.ftp_tactile_prefix_dim,
                    tokens_per_finger=config.origami_vla.ftp_tactile_tokens_per_finger,
                    hands=config.origami_vla.ftp_tactile_hands,
                    fingers_per_hand=config.origami_vla.ftp_tactile_fingers_per_hand,
                    resampler_layers=config.origami_vla.ftp_tactile_resampler_layers,
                    resampler_heads=config.origami_vla.ftp_tactile_resampler_heads,
                    resampler_ffn_dim=config.origami_vla.ftp_tactile_resampler_ffn_dim,
                    cross_finger_layers=config.origami_vla.ftp_tactile_cross_finger_layers,
                    cross_finger_heads=config.origami_vla.ftp_tactile_cross_finger_heads,
                    cross_finger_ffn_dim=config.origami_vla.ftp_tactile_cross_finger_ffn_dim,
                    dropout=config.origami_vla.ftp_tactile_dropout,
                    rope_enabled=config.origami_vla.ftp_tactile_rope_enabled,
                    rope_base=config.origami_vla.ftp_tactile_rope_base,
                    rope_scale=config.origami_vla.ftp_tactile_rope_scale,
                    hand_values=config.origami_vla.ftp_tactile_hand_values,
                    finger_values=config.origami_vla.ftp_tactile_finger_values,
                    branch=config.origami_vla.ftp_tactile_branch,
                )
                self.origami_ftp_tactile_prefix_encoder = nnx_bridge.ToNNX(
                    _origami_ftp_tactile_prefix_encoder.OrigamiFtpTactilePrefixEncoder(
                        ftp_tactile_cfg,
                    )
                )
                tactile_images = jnp.zeros(
                    (
                        1,
                        ftp_tactile_cfg.finger_count,
                        3,
                        ftp_tactile_cfg.image_size,
                        ftp_tactile_cfg.image_size,
                    ),
                    dtype=jnp.uint8,
                )
                self.origami_ftp_tactile_prefix_encoder.lazy_init(
                    tactile_images,
                    tactile_images,
                    jnp.ones((1,), dtype=jnp.bool_),
                    train=False,
                    rngs=rngs,
                )
            if config.origami_vla.tactile_enabled:
                if self.origami_tactile_stats is None:
                    raise FileNotFoundError(
                        "Origami tactile conditioning is enabled, but tactile normalization stats were not loaded."
                    )
                self.origami_tactile_adapter = nnx_bridge.ToNNX(
                    _origami_tactile_adapter.OrigamiTactilePrefixAdapter(
                        tactile_dim=config.origami_vla.tactile_dim,
                        finger_count=config.origami_vla.tactile_finger_count,
                        channels_per_finger=config.origami_vla.tactile_channels_per_finger,
                        token_dim=config.origami_vla.tactile_token_dim,
                        output_dim=paligemma_config.width,
                        finger_hidden_dims=config.origami_vla.tactile_finger_hidden_dims,
                        transformer_layers=config.origami_vla.tactile_transformer_layers,
                        attention_heads=config.origami_vla.tactile_attention_heads,
                        ffn_dim=config.origami_vla.tactile_ffn_dim,
                        use_type_embeddings=config.origami_vla.tactile_use_type_embeddings,
                        tanh_scale=config.origami_vla.tactile_soft_clip_scale,
                        min_scale=config.origami_vla.tactile_min_scale,
                        norm_center=self.origami_tactile_stats.center,
                        norm_scale=self.origami_tactile_stats.scale,
                    )
                )
                self.origami_tactile_adapter.lazy_init(
                    jnp.zeros((1, config.origami_vla.tactile_dim), dtype=jnp.float32),
                    train=False,
                    rngs=rngs,
                )
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _select_future_latents(
        self,
        obs: _model.Observation,
        *,
        rng: at.KeyArrayLike | None,
        train: bool,
    ) -> tuple[at.Float[at.Array, "b c t d"], at.Bool[at.Array, "b c"]] | None:
        if not self.future_latent_config.enabled or obs.future_latent_pred is None:
            return None

        pred = jnp.asarray(obs.future_latent_pred)
        true = jnp.asarray(obs.future_latent_true) if obs.future_latent_true is not None else pred
        valid = (
            jnp.asarray(obs.future_latent_valid_mask, dtype=jnp.bool_)
            if obs.future_latent_valid_mask is not None
            else jnp.ones(pred.shape[:2], dtype=jnp.bool_)
        )
        batch_size = pred.shape[0]

        if train and rng is not None:
            probs = self.future_latent_config
            choice = jax.random.uniform(rng, (batch_size,))
            use_pred = choice < probs.predicted_latent_prob
            use_true = choice < (probs.predicted_latent_prob + probs.true_latent_prob)
            selected = jnp.where(use_pred[:, None, None, None], pred, true)
            selected = jnp.where(use_true[:, None, None, None], selected, jnp.zeros_like(selected))
            selected_valid = jnp.where(use_true[:, None], valid, jnp.zeros_like(valid))
            return selected, selected_valid

        return pred, valid

    def encode_future_latent_image_embeddings(self, obs: _model.Observation) -> dict[str, at.Array]:
        """Return current image encoder tokens used by the PyTorch future-latent stack at inference."""
        obs = _model.preprocess_observation(None, obs, train=False, image_keys=tuple(obs.images.keys()))
        embeddings = {}
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            embeddings[name] = image_tokens
        return embeddings

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation, *, rng: at.KeyArrayLike | None = None, train: bool = False
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in self.image_keys:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]

        if self.robot_spline_config.enabled:
            if obs.robot_spline_coefficients is None or obs.robot_spline_knots is None:
                raise ValueError("robot_spline_coefficients and robot_spline_knots are required when robot_spline is enabled.")
            spline_tokens = self.robot_spline_adapter(
                jnp.asarray(obs.robot_spline_coefficients),
                jnp.asarray(obs.robot_spline_knots),
                train=train,
            )
            if tokens:
                spline_tokens = spline_tokens.astype(tokens[0].dtype)
            tokens.append(spline_tokens)
            input_mask.append(jnp.ones(spline_tokens.shape[:2], dtype=jnp.bool_))
            ar_mask += [False] * spline_tokens.shape[1]

        future_latents = self._select_future_latents(obs, rng=rng, train=train)
        if future_latents is not None:
            future_values, future_valid = future_latents
            future_tokens, future_mask = self.future_latent_adapter(future_values, future_valid)
            if tokens:
                future_tokens = future_tokens.astype(tokens[0].dtype)
            tokens.append(future_tokens)
            input_mask.append(future_mask)
            ar_mask += [False] * future_tokens.shape[1]

        if self.origami_vla_config.enabled:
            required = (
                obs.planner_state_belief,
                obs.planner_progress_transition,
                obs.planner_uncertainty,
                obs.planner_history_latent,
            )
            if any(value is None for value in required):
                raise ValueError("Origami planner conditioning is enabled, but planner rollout features are missing.")
            planner_tokens = self.origami_planner_adapter(
                jnp.asarray(obs.planner_state_belief),
                jnp.asarray(obs.planner_progress_transition),
                jnp.asarray(obs.planner_uncertainty),
                jnp.asarray(obs.planner_history_latent),
                train=train,
            )
            if tokens:
                planner_tokens = planner_tokens.astype(tokens[0].dtype)
            tokens.append(planner_tokens)
            if obs.planner_available is None:
                planner_mask = jnp.ones(planner_tokens.shape[:-1], dtype=jnp.bool_)
            else:
                planner_available = jnp.asarray(obs.planner_available, dtype=jnp.bool_)
                planner_mask = jnp.broadcast_to(planner_available[..., None], planner_tokens.shape[:-1])
            input_mask.append(planner_mask)
            ar_mask += [False] * planner_tokens.shape[1]
            if self.origami_vla_config.tactile_enabled:
                if obs.tactile is None:
                    raise ValueError("Origami tactile conditioning is enabled, but observation.tactile is missing.")
                tactile_tokens = self.origami_tactile_adapter(
                    jnp.asarray(obs.tactile),
                    train=train,
                )
                if tokens:
                    tactile_tokens = tactile_tokens.astype(tokens[0].dtype)
                tokens.append(tactile_tokens)
                input_mask.append(jnp.ones(tactile_tokens.shape[:2], dtype=jnp.bool_))
                ar_mask += [False] * tactile_tokens.shape[1]
            if self.origami_vla_config.ftp_tactile_enabled:
                if obs.tactile_deform_images is None:
                    raise ValueError(
                        "Origami FTP tactile prefix conditioning is enabled, but tactile_deform_images is missing."
                    )
                ftp_tactile_tokens = self.origami_ftp_tactile_prefix_encoder(
                    jnp.asarray(obs.tactile_deform_images),
                    None if obs.tactile_raw_images is None else jnp.asarray(obs.tactile_raw_images),
                    None if obs.tactile_raw_available is None else jnp.asarray(obs.tactile_raw_available),
                    # Keep the pretrained tactile prefix path deterministic inside Pi0.5.
                    # The non-frozen tactile weights still receive gradients from the VLA loss.
                    train=False,
                )
                if tokens:
                    ftp_tactile_tokens = ftp_tactile_tokens.astype(tokens[0].dtype)
                tokens.append(ftp_tactile_tokens)
                input_mask.append(jnp.ones(ftp_tactile_tokens.shape[:2], dtype=jnp.bool_))
                ar_mask += [False] * ftp_tactile_tokens.shape[1]

        if not tokens:
            raise ValueError("Prefix construction produced no tokens. Provide at least language/state tokens or another prefix modality.")
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss_and_metrics(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        preprocess_rng, future_rng, noise_rng, time_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            image_keys=self.image_keys,
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation, rng=future_rng, train=train)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = _positions_from_input_mask(input_mask)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        sq_error = jnp.square(v_t - u_t)
        if observation.action_mask is not None:
            action_mask = jnp.asarray(observation.action_mask, dtype=sq_error.dtype)
            action_mask = jnp.broadcast_to(action_mask, sq_error.shape)
            denom = jnp.clip(jnp.sum(action_mask, axis=-1), 1.0)
            base_loss = jnp.sum(sq_error * action_mask, axis=-1) / denom
        else:
            action_mask = jnp.ones_like(sq_error, dtype=jnp.float32)
            base_loss = jnp.mean(sq_error, axis=-1)

        sample_weight = (
            observation.sample_weight
            if (
                self.origami_vla_config.enabled
                and self.origami_vla_config.episode_execution_speed_preference
                and self.origami_vla_config.use_speed_efficiency_weight
            )
            else None
        )

        def reduce_metric(values: at.Array) -> at.Array:
            return _model.reduce_batch_metric(
                values,
                sample_weight,
                normalize=self.origami_vla_config.normalize_speed_efficiency_weighted_loss,
                eps=self.origami_vla_config.speed_efficiency_weight_eps,
            )

        base_loss_mean = reduce_metric(base_loss)
        if not self.origami_vla_config.enabled:
            return base_loss, {
                "loss": base_loss_mean,
                "loss_total": base_loss_mean,
                "loss_base_flow": base_loss_mean,
            }
        if self.origami_vla_config.disable_auxiliary_losses or self.origami_vla_config.action_mode != "spline":
            zero = jnp.asarray(0.0, dtype=base_loss.dtype)
            return base_loss, {
                "loss": base_loss_mean,
                "loss_total": base_loss_mean,
                "loss_base_flow": base_loss_mean,
                "loss_aux_total": zero,
                "loss_curve_raw": zero,
                "loss_start_raw": zero,
                "loss_end_raw": zero,
                "loss_width_raw": zero,
                "loss_curve_weighted": zero,
                "loss_start_weighted": zero,
                "loss_end_weighted": zero,
                "loss_width_weighted": zero,
            }

        pred_actions_for_aux = x_t - time_expanded * v_t
        aux_loss, aux_terms = _origami_spline_losses.compute_auxiliary_losses(
            pred_actions_for_aux,
            actions,
            action_mask,
            stats=self.origami_action_stats,
            use_quantiles=self.origami_vla_config.use_quantile_norm,
            degree=self.origami_vla_config.degree,
            max_control_points=self.origami_vla_config.max_control_points,
            max_span_count=self.origami_vla_config.max_span_count,
            sample_count=self.origami_vla_config.curve_sample_count,
            smooth_l1_beta=self.origami_vla_config.smooth_l1_beta,
            width_min=self.origami_vla_config.width_min,
            curve_weight=self.origami_vla_config.curve_loss_weight,
            start_weight=self.origami_vla_config.start_loss_weight,
            end_weight=self.origami_vla_config.end_loss_weight,
            width_weight=self.origami_vla_config.width_loss_weight,
        )
        total_loss = base_loss + aux_loss[:, None]

        curve_raw = reduce_metric(aux_terms["curve"])
        start_raw = reduce_metric(aux_terms["start"])
        end_raw = reduce_metric(aux_terms["end"])
        width_raw = reduce_metric(aux_terms["width"])
        aux_total = reduce_metric(aux_loss)
        total_mean = reduce_metric(total_loss)

        return total_loss, {
            "loss": total_mean,
            "loss_total": total_mean,
            "loss_base_flow": base_loss_mean,
            "loss_aux_total": aux_total,
            "loss_curve_raw": curve_raw,
            "loss_start_raw": start_raw,
            "loss_end_raw": end_raw,
            "loss_width_raw": width_raw,
            "loss_curve_weighted": self.origami_vla_config.curve_loss_weight * curve_raw,
            "loss_start_weighted": self.origami_vla_config.start_loss_weight * start_raw,
            "loss_end_weighted": self.origami_vla_config.end_loss_weight * end_raw,
            "loss_width_weighted": self.origami_vla_config.width_loss_weight * width_raw,
        }

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        chunked_loss, _ = self.compute_loss_and_metrics(rng, observation, actions, train=train)
        return chunked_loss

    def _compute_prefix_cache(
        self,
        observation: _model.Observation,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        Any,
    ]:
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = _positions_from_input_mask(prefix_mask)
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return prefix_tokens, prefix_mask, kv_cache

    def _compute_suffix_out(
        self,
        observation: _model.Observation,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        kv_cache: Any,
        x_t: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> at.Float[at.Array, "b ah emb"]:
        batch_size = observation.state.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, timestep)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + _safe_cumsum(
            jnp.asarray(suffix_mask, dtype=jnp.int32), axis=-1
        ) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return suffix_out[:, -self.action_horizon :]

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=self.image_keys)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, kv_cache = self._compute_prefix_cache(observation)

        def step(carry):
            x_t, time = carry
            suffix_out = self._compute_suffix_out(
                observation,
                prefix_tokens,
                prefix_mask,
                kv_cache,
                x_t,
                jnp.broadcast_to(time, batch_size),
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions_with_policy_latent(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> dict[str, at.Array]:
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=self.image_keys)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, kv_cache = self._compute_prefix_cache(observation)

        def step(carry):
            x_t, time = carry
            suffix_out = self._compute_suffix_out(
                observation,
                prefix_tokens,
                prefix_mask,
                kv_cache,
                x_t,
                jnp.broadcast_to(time, batch_size),
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        policy_latent = self._compute_suffix_out(
            observation,
            prefix_tokens,
            prefix_mask,
            kv_cache,
            x_0,
            jnp.zeros((batch_size,), dtype=x_0.dtype),
        )
        return {
            "actions": x_0,
            "policy_latent": policy_latent,
        }
