import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.future_latent_order as _future_latent_order
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class FutureLatentConfig:
    enabled: bool = False
    num_cameras: int = 3
    latent_tokens: int = 24
    latent_dim: int = 512
    adapter_hidden_dim: int = 1024
    output_dim: int = 2048
    predicted_latent_prob: float = 0.50
    true_latent_prob: float = 0.30
    dropped_latent_prob: float = 0.20
    sidecar_root: str | None = None
    resampler_checkpoint_path: str | None = None
    future_predictor_checkpoint_path: str | None = None
    policy_camera_order: tuple[str, ...] = _future_latent_order.POLICY_FUTURE_LATENT_CAMERA_ORDER
    freeze_image_encoder: bool = True
    freeze_resampler: bool = True
    freeze_future_predictor: bool = True

    def __post_init__(self) -> None:
        total = self.predicted_latent_prob + self.true_latent_prob + self.dropped_latent_prob
        if self.enabled and abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Future latent mixture probabilities must sum to 1.0, got "
                f"{self.predicted_latent_prob} + {self.true_latent_prob} + {self.dropped_latent_prob} = {total}"
            )
        if self.enabled and len(self.policy_camera_order) != self.num_cameras:
            raise ValueError(
                "Future latent policy_camera_order must have num_cameras entries, got "
                f"{self.policy_camera_order} for num_cameras={self.num_cameras}"
            )


@dataclasses.dataclass(frozen=True)
class RobotSplineConfig:
    enabled: bool = False
    use_image_prefix: bool = True
    control_count: int = 13
    degree: int = 3
    control_point_dim: int = 2048
    model_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    ffn_dim: int = 2048
    width_fourier_bands: int = 8
    width_hidden_dim: int = 512
    rope_base: float = 10_000.0

    @property
    def knot_count(self) -> int:
        return self.control_count + self.degree + 1

    def __post_init__(self) -> None:
        if self.control_count <= 0:
            raise ValueError(f"control_count must be positive, got {self.control_count}")
        if self.degree != 3:
            raise ValueError(f"Only cubic robot spline conditioning is currently supported, got degree={self.degree}")
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"robot spline model_dim ({self.model_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        head_dim = self.model_dim // self.num_heads
        if head_dim % 4 != 0:
            raise ValueError(
                f"robot spline per-head dimension ({head_dim}) must be divisible by 4 for 4D RoPE"
            )
        if (head_dim // 4) % 2 != 0:
            raise ValueError(
                f"robot spline per-geometry RoPE chunk ({head_dim // 4}) must be even"
            )


@dataclasses.dataclass(frozen=True)
class OrigamiVlaConfig:
    enabled: bool = False
    action_mode: Literal["spline", "action_chunk", "bspline_points"] = "spline"
    # User-facing switch: when enabled, the VLA loss uses per-sample speed-efficiency
    # weights so faster semantic checkpoint executions contribute more strongly.
    episode_execution_speed_preference: bool = False
    belief_dim: int = 39
    history_dim: int = 512
    planner_belief_hidden_dims: tuple[int, ...] = (256, 512)
    planner_progress_hidden_dims: tuple[int, ...] = (128, 512)
    planner_uncertainty_hidden_dims: tuple[int, ...] = (128, 512)
    planner_history_hidden_dims: tuple[int, ...] = (1024,)
    planner_use_type_embeddings: bool = True
    degree: int = 3
    max_control_points: int = 18
    max_span_count: int = 15
    spline_span_representation: Literal["physical_widths", "logits"] = "physical_widths"
    bspline_control_point_count: int = 13
    bspline_point_count: int = 17
    bspline_width_logit_count: int = 10
    bspline_curve_sample_intervals: int = 120
    bspline_softmax_clip: float | None = 30.0
    bspline_denominator_eps: float = 1.0e-6
    bspline_aux_metrics_enabled: bool = True
    curve_sample_count: int = 120
    smooth_l1_beta: float = 0.05
    curve_loss_weight: float = 0.5
    start_loss_weight: float = 0.1
    end_loss_weight: float = 0.25
    width_loss_weight: float = 0.05
    width_min: float = 1e-4
    curve_fm_enabled: bool = False
    curve_fm_backprop: bool = False
    curve_fm_loss_weight: float = 0.0
    curve_fm_sample_intervals: int = 10
    curve_fm_include_endpoints: bool = True
    curve_fm_width_min: float = 1e-4
    curve_fm_denominator_eps: float = 1e-6
    curve_fm_softmax_clip: float | None = 30.0
    curve_fm_loss_clip: float | None = None
    curve_fm_separate_velocity_metrics: bool = False
    mask_action_noise: bool = True
    action_norm_stats_dir: str | None = None
    use_quantile_norm: bool = True
    compute_auxiliary_metrics: bool = True
    backprop_auxiliary_losses: bool = True
    disable_auxiliary_losses: bool = False
    tactile_prompt_input: bool = False
    prompt_discrete_clip: bool = False
    tactile_enabled: bool = False
    tactile_dim: int = 60
    tactile_finger_count: int = 10
    tactile_channels_per_finger: int = 6
    tactile_token_dim: int = 256
    tactile_finger_hidden_dims: tuple[int, ...] = (64, 128)
    tactile_transformer_layers: int = 2
    tactile_attention_heads: int = 4
    tactile_ffn_dim: int = 512
    tactile_use_type_embeddings: bool = True
    tactile_quantile_low: float = 0.005
    tactile_quantile_high: float = 0.995
    tactile_min_scale: float = 1.0e-6
    tactile_soft_clip_scale: float = 5.0
    ftp_tactile_enabled: bool = False
    ftp_tactile_branch: Literal["auto", "deform", "full"] = "auto"
    ftp_tactile_image_size: int = 224
    ftp_tactile_patch_size: int = 16
    ftp_tactile_width: int = 768
    ftp_tactile_depth: int = 3
    ftp_tactile_heads: int = 12
    ftp_tactile_mlp_ratio: int = 4
    ftp_tactile_backbone_micro_batch: int = 64
    ftp_tactile_freeze_backbone: bool = True
    ftp_tactile_include_cls_token: bool = True
    ftp_tactile_normalize_images: bool = True
    ftp_tactile_image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    ftp_tactile_image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    ftp_tactile_adapter_dim: int = 512
    ftp_tactile_prefix_dim: int = 2048
    ftp_tactile_tokens_per_finger: int = 4
    ftp_tactile_hands: int = 2
    ftp_tactile_fingers_per_hand: int = 5
    ftp_tactile_resampler_layers: int = 2
    ftp_tactile_resampler_heads: int = 8
    ftp_tactile_resampler_ffn_dim: int = 2048
    ftp_tactile_cross_finger_layers: int = 4
    ftp_tactile_cross_finger_heads: int = 8
    ftp_tactile_cross_finger_ffn_dim: int = 2048
    ftp_tactile_dropout: float = 0.1
    ftp_tactile_rope_enabled: bool = True
    ftp_tactile_rope_base: float = 10000.0
    ftp_tactile_rope_scale: float = 1.0
    ftp_tactile_hand_values: tuple[float, ...] = (-1.0, 1.0)
    ftp_tactile_finger_values: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    # Internal loss-weighting toggle. Keep this enabled and drive behavior with
    # episode_execution_speed_preference unless you need lower-level debugging.
    use_speed_efficiency_weight: bool = True
    normalize_speed_efficiency_weighted_loss: bool = True
    speed_efficiency_weight_eps: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.action_mode not in ("spline", "action_chunk", "bspline_points"):
            raise ValueError(f"Unsupported Origami action_mode: {self.action_mode!r}")
        if self.spline_span_representation not in ("physical_widths", "logits"):
            raise ValueError(
                "Origami spline_span_representation must be 'physical_widths' or 'logits', "
                f"got {self.spline_span_representation!r}"
            )
        if self.action_mode != "spline" and self.spline_span_representation != "physical_widths":
            raise ValueError("spline_span_representation='logits' is only valid for action_mode='spline'.")
        if self.action_mode == "bspline_points":
            if self.bspline_control_point_count <= 0:
                raise ValueError("bspline_control_point_count must be positive.")
            if self.bspline_point_count <= 0:
                raise ValueError("bspline_point_count must be positive.")
            if self.bspline_width_logit_count <= 0:
                raise ValueError("bspline_width_logit_count must be positive.")
            if self.bspline_curve_sample_intervals <= 0:
                raise ValueError("bspline_curve_sample_intervals must be positive.")
            if self.bspline_denominator_eps <= 0.0:
                raise ValueError("bspline_denominator_eps must be > 0.")
            if self.bspline_softmax_clip is not None and self.bspline_softmax_clip <= 0.0:
                raise ValueError("bspline_softmax_clip must be positive when set.")
        if self.curve_fm_backprop and not self.curve_fm_enabled:
            raise ValueError("curve_fm_backprop=True requires curve_fm_enabled=True.")
        if self.curve_fm_backprop and self.curve_fm_loss_weight <= 0.0:
            raise ValueError("curve_fm_backprop=True requires curve_fm_loss_weight > 0.")
        if self.curve_fm_enabled and self.action_mode != "spline":
            raise ValueError("curve_fm_enabled=True is only valid for action_mode='spline'.")
        if self.curve_fm_sample_intervals <= 0:
            raise ValueError("curve_fm_sample_intervals must be positive.")
        if self.curve_fm_width_min < 0.0:
            raise ValueError("curve_fm_width_min must be >= 0.")
        if self.curve_fm_denominator_eps <= 0.0:
            raise ValueError("curve_fm_denominator_eps must be > 0.")
        if self.curve_fm_softmax_clip is not None and self.curve_fm_softmax_clip <= 0.0:
            raise ValueError("curve_fm_softmax_clip must be positive when set.")
        if self.curve_fm_loss_clip is not None and self.curve_fm_loss_clip <= 0.0:
            raise ValueError("curve_fm_loss_clip must be positive when set.")
        if self.speed_efficiency_weight_eps <= 0.0:
            raise ValueError("speed_efficiency_weight_eps must be > 0")
        if self.tactile_enabled or self.tactile_prompt_input:
            expected_dim = self.tactile_finger_count * self.tactile_channels_per_finger
            if self.tactile_dim != expected_dim:
                raise ValueError(
                    "Origami tactile_dim must equal tactile_finger_count * tactile_channels_per_finger, "
                    f"got tactile_dim={self.tactile_dim} and expected {expected_dim}."
                )
            if not (0.0 <= self.tactile_quantile_low < self.tactile_quantile_high <= 1.0):
                raise ValueError(
                    "Origami tactile quantiles must satisfy 0 <= low < high <= 1, got "
                    f"{self.tactile_quantile_low} and {self.tactile_quantile_high}."
                )
            if self.tactile_min_scale <= 0.0:
                raise ValueError("Origami tactile_min_scale must be > 0.")
            if self.tactile_soft_clip_scale <= 0.0:
                raise ValueError("Origami tactile_soft_clip_scale must be > 0.")
        if self.ftp_tactile_enabled:
            if self.ftp_tactile_branch not in ("auto", "deform", "full"):
                raise ValueError(f"Unsupported ftp_tactile_branch: {self.ftp_tactile_branch!r}")
            if self.ftp_tactile_image_size <= 0:
                raise ValueError("ftp_tactile_image_size must be positive.")
            if self.ftp_tactile_patch_size <= 0:
                raise ValueError("ftp_tactile_patch_size must be positive.")
            if self.ftp_tactile_image_size % self.ftp_tactile_patch_size:
                raise ValueError("ftp_tactile_image_size must be divisible by ftp_tactile_patch_size.")
            if not self.ftp_tactile_include_cls_token:
                raise ValueError("FTP SharpaWave prefix support currently expects ftp_tactile_include_cls_token=True.")
            if self.ftp_tactile_adapter_dim <= 0 or self.ftp_tactile_prefix_dim <= 0:
                raise ValueError("FTP tactile adapter and prefix dimensions must be positive.")
            if self.ftp_tactile_adapter_dim % self.ftp_tactile_resampler_heads:
                raise ValueError("ftp_tactile_adapter_dim must be divisible by ftp_tactile_resampler_heads.")
            if self.ftp_tactile_adapter_dim % self.ftp_tactile_cross_finger_heads:
                raise ValueError("ftp_tactile_adapter_dim must be divisible by ftp_tactile_cross_finger_heads.")
            if self.ftp_tactile_width % self.ftp_tactile_heads:
                raise ValueError("ftp_tactile_width must be divisible by ftp_tactile_heads.")
            if self.ftp_tactile_tokens_per_finger <= 0:
                raise ValueError("ftp_tactile_tokens_per_finger must be positive.")
            if self.ftp_tactile_hands <= 0 or self.ftp_tactile_fingers_per_hand <= 0:
                raise ValueError("FTP tactile hands and fingers_per_hand must be positive.")
            if len(self.ftp_tactile_hand_values) != self.ftp_tactile_hands:
                raise ValueError("ftp_tactile_hand_values must match ftp_tactile_hands.")
            if len(self.ftp_tactile_finger_values) != self.ftp_tactile_fingers_per_hand:
                raise ValueError("ftp_tactile_finger_values must match ftp_tactile_fingers_per_hand.")
            if not 0.0 <= self.ftp_tactile_dropout < 1.0:
                raise ValueError("ftp_tactile_dropout must satisfy 0 <= dropout < 1.")


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    state_dim: int | None = None
    image_keys: tuple[str, ...] = _model.IMAGE_KEYS
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    future_latent: FutureLatentConfig = dataclasses.field(default_factory=FutureLatentConfig)
    robot_spline: RobotSplineConfig = dataclasses.field(default_factory=RobotSplineConfig)
    origami_vla: OrigamiVlaConfig = dataclasses.field(default_factory=OrigamiVlaConfig)

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.state_dim is None:
            object.__setattr__(self, "state_dim", self.action_dim)
        if not self.image_keys:
            raise ValueError("image_keys must contain at least one image key when using the image prefix.")
        if self.origami_vla.enabled and not self.pi05:
            raise ValueError("Origami VLA support is currently implemented only for pi0.5.")
        if (
            self.origami_vla.enabled
            and self.origami_vla.action_mode == "spline"
            and self.action_horizon != self.origami_vla.max_control_points + 1
        ):
            raise ValueError(
                "Origami VLA actions are packed as control-point tokens plus one span-width token, "
                f"so action_horizon must equal {self.origami_vla.max_control_points + 1}, got {self.action_horizon}."
            )
        if (
            self.origami_vla.enabled
            and self.origami_vla.action_mode == "spline"
            and self.action_dim < self.origami_vla.max_span_count
        ):
            raise ValueError(
                f"Origami VLA action_dim ({self.action_dim}) must be >= max_span_count "
                f"({self.origami_vla.max_span_count})."
            )
        if (
            self.origami_vla.enabled
            and self.origami_vla.action_mode == "bspline_points"
            and self.action_horizon != self.origami_vla.bspline_point_count + 1
        ):
            raise ValueError(
                "Origami B-spline point actions are packed as point tokens plus one width-logit token, "
                f"so action_horizon must equal {self.origami_vla.bspline_point_count + 1}, "
                f"got {self.action_horizon}."
            )
        if (
            self.origami_vla.enabled
            and self.origami_vla.action_mode == "bspline_points"
            and self.action_dim < self.origami_vla.bspline_width_logit_count
        ):
            raise ValueError(
                f"Origami B-spline point action_dim ({self.action_dim}) must be >= "
                f"bspline_width_logit_count ({self.origami_vla.bspline_width_logit_count})."
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        include_images = not (self.robot_spline.enabled and not self.robot_spline.use_image_prefix)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images=(
                    {
                        key: image_spec for key in self.image_keys
                    }
                    if include_images
                    else {}
                ),
                image_masks=(
                    {
                        key: image_mask_spec for key in self.image_keys
                    }
                    if include_images
                    else {}
                ),
                state=jax.ShapeDtypeStruct([batch_size, self.state_dim], jnp.float32),
                sample_weight=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                future_latent_pred=(
                    jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.future_latent.num_cameras,
                            self.future_latent.latent_tokens,
                            self.future_latent.latent_dim,
                        ],
                        jnp.float32,
                    )
                    if self.future_latent.enabled
                    else None
                ),
                future_latent_true=(
                    jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.future_latent.num_cameras,
                            self.future_latent.latent_tokens,
                            self.future_latent.latent_dim,
                        ],
                        jnp.float32,
                    )
                    if self.future_latent.enabled
                    else None
                ),
                future_latent_valid_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.future_latent.num_cameras], jnp.bool_)
                    if self.future_latent.enabled
                    else None
                ),
                robot_spline_coefficients=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.robot_spline.control_count, self.robot_spline.control_point_dim],
                        jnp.float32,
                    )
                    if self.robot_spline.enabled
                    else None
                ),
                robot_spline_knots=(
                    jax.ShapeDtypeStruct([batch_size, self.robot_spline.knot_count], jnp.float32)
                    if self.robot_spline.enabled
                    else None
                ),
                planner_state_belief=(
                    jax.ShapeDtypeStruct([batch_size, self.origami_vla.belief_dim], jnp.float32)
                    if self.origami_vla.enabled
                    else None
                ),
                planner_progress_transition=(
                    jax.ShapeDtypeStruct([batch_size, 2], jnp.float32)
                    if self.origami_vla.enabled
                    else None
                ),
                planner_uncertainty=(
                    jax.ShapeDtypeStruct([batch_size, 3], jnp.float32)
                    if self.origami_vla.enabled
                    else None
                ),
                planner_history_latent=(
                    jax.ShapeDtypeStruct([batch_size, self.origami_vla.history_dim], jnp.float32)
                    if self.origami_vla.enabled
                    else None
                ),
                planner_available=(
                    jax.ShapeDtypeStruct([batch_size], jnp.bool_)
                    if self.origami_vla.enabled
                    else None
                ),
                tactile=(
                    jax.ShapeDtypeStruct([batch_size, self.origami_vla.tactile_dim], jnp.float32)
                    if self.origami_vla.enabled and self.origami_vla.tactile_enabled
                    else None
                ),
                tactile_deform_images=(
                    jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.origami_vla.ftp_tactile_hands * self.origami_vla.ftp_tactile_fingers_per_hand,
                            3,
                            self.origami_vla.ftp_tactile_image_size,
                            self.origami_vla.ftp_tactile_image_size,
                        ],
                        jnp.uint8,
                    )
                    if self.origami_vla.enabled and self.origami_vla.ftp_tactile_enabled
                    else None
                ),
                tactile_deform_available=(
                    jax.ShapeDtypeStruct([batch_size], jnp.bool_)
                    if self.origami_vla.enabled and self.origami_vla.ftp_tactile_enabled
                    else None
                ),
                tactile_raw_images=(
                    jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.origami_vla.ftp_tactile_hands * self.origami_vla.ftp_tactile_fingers_per_hand,
                            3,
                            self.origami_vla.ftp_tactile_image_size,
                            self.origami_vla.ftp_tactile_image_size,
                        ],
                        jnp.uint8,
                    )
                    if self.origami_vla.enabled and self.origami_vla.ftp_tactile_enabled
                    else None
                ),
                tactile_raw_available=(
                    jax.ShapeDtypeStruct([batch_size], jnp.bool_)
                    if self.origami_vla.enabled and self.origami_vla.ftp_tactile_enabled
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if (
            self.origami_vla.enabled
            and self.origami_vla.ftp_tactile_enabled
            and self.origami_vla.ftp_tactile_freeze_backbone
        ):
            filters.append(nnx_utils.PathRegex(".*origami_ftp_tactile_prefix_encoder/backbone/.*"))
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
