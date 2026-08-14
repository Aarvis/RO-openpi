import dataclasses
from typing import TYPE_CHECKING

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
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    future_latent: FutureLatentConfig = dataclasses.field(default_factory=FutureLatentConfig)
    robot_spline: RobotSplineConfig = dataclasses.field(default_factory=RobotSplineConfig)

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

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
                        "base_0_rgb": image_spec,
                        "left_wrist_0_rgb": image_spec,
                        "right_wrist_0_rgb": image_spec,
                    }
                    if include_images
                    else {}
                ),
                image_masks=(
                    {
                        "base_0_rgb": image_mask_spec,
                        "left_wrist_0_rgb": image_mask_spec,
                        "right_wrist_0_rgb": image_mask_spec,
                    }
                    if include_images
                    else {}
                ),
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
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
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
