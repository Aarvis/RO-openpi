import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


def _orbax_metadata_tree(metadata):
    """Return the PyTree metadata across Orbax metadata API versions."""
    if isinstance(metadata, dict):
        return metadata

    tree = getattr(metadata, "tree", None)
    if tree is not None:
        return tree

    item_metadata = getattr(metadata, "item_metadata", None)
    if item_metadata is not None:
        if isinstance(item_metadata, dict):
            return item_metadata
        tree = getattr(item_metadata, "tree", None)
        if tree is not None:
            return tree

    raise TypeError(
        "Unsupported Orbax metadata object returned by PyTreeCheckpointer.metadata(): "
        f"{type(metadata)!r}. Expected dict-like metadata or an object with a 'tree' attribute."
    )


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "action_mask": bool[*b, ah, ad],  # Optional, loss mask for continuous action dimensions
#     "sample_weight": float32[*b],  # Optional, per-sample training weight
#     "tactile": float32[*b, 60],  # Optional, low-dimensional tactile wrench vector
#     "planner_available": bool[*b],  # Optional, masks checkpoint-planner prefix tokens when absent
#     "tactile_deform_images": uint8|float32[*b, 10, 3, 224, 224],  # Optional tactile image crops
#     "tactile_deform_available": bool[*b],  # Optional deform tactile image availability mask
#     "tactile_raw_images": uint8|float32[*b, 10, 3, 224, 224],  # Optional tactile image crops
#     "tactile_raw_available": bool[*b],  # Optional raw tactile image availability mask
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]
    # Optional mask for continuous action loss.
    action_mask: at.Bool[ArrayT, "*b ah ad"] | None = None
    # Optional per-sample training weight.
    sample_weight: at.Float[ArrayT, "*b"] | None = None

    # Optional compact future visual latents used by future-latent-conditioned policies.
    # Shape convention: [*b, cameras, latent_tokens, latent_dim].
    future_latent_pred: at.Float[ArrayT, "*b c t d"] | None = None
    future_latent_true: at.Float[ArrayT, "*b c t d"] | None = None
    future_latent_valid_mask: at.Bool[ArrayT, "*b c"] | None = None

    # Optional predicted robot spline sidecar used by spline-conditioned pi0.5 variants.
    robot_spline_coefficients: at.Float[ArrayT, "*b n d"] | None = None
    robot_spline_knots: at.Float[ArrayT, "*b k"] | None = None

    # Optional checkpoint-planner rollout features used by Origami VLA variants.
    planner_state_belief: at.Float[ArrayT, "*b belief"] | None = None
    planner_progress_transition: at.Float[ArrayT, "*b p"] | None = None
    planner_uncertainty: at.Float[ArrayT, "*b u"] | None = None
    planner_history_latent: at.Float[ArrayT, "*b hist"] | None = None
    planner_available: at.Bool[ArrayT, "*b"] | None = None
    # Optional tactile wrench observation used by Origami VLA variants.
    tactile: at.Float[ArrayT, "*b tactile"] | None = None
    # Optional tactile image crops used by FTP/SharpaWave tactile prefix variants.
    tactile_deform_images: at.Array | None = None
    tactile_deform_available: at.Bool[ArrayT, "*b"] | None = None
    tactile_raw_images: at.Array | None = None
    tactile_raw_available: at.Bool[ArrayT, "*b"] | None = None

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        image_dict = data.get("image", {})
        image_mask_dict = data.get("image_mask", {})
        # If images are uint8, convert them to [-1, 1] float32.
        for key in image_dict:
            if image_dict[key].dtype == np.uint8:
                image_dict[key] = image_dict[key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(image_dict[key], "dtype") and image_dict[key].dtype == torch.uint8:
                image_dict[key] = image_dict[key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(
            images=image_dict,
            image_masks=image_mask_dict,
            state=data["state"],
            action_mask=data.get("action_mask"),
            sample_weight=data.get("sample_weight"),
            future_latent_pred=data.get("future_latent_pred"),
            future_latent_true=data.get("future_latent_true"),
            future_latent_valid_mask=data.get("future_latent_valid_mask"),
            robot_spline_coefficients=data.get("robot_spline_coefficients"),
            robot_spline_knots=data.get("robot_spline_knots"),
            planner_state_belief=data.get("planner_state_belief"),
            planner_progress_transition=data.get("planner_progress_transition"),
            planner_uncertainty=data.get("planner_uncertainty"),
            planner_history_latent=data.get("planner_history_latent"),
            planner_available=data.get("planner_available"),
            tactile=data.get("tactile"),
            tactile_deform_images=data.get("tactile_deform_images"),
            tactile_deform_available=data.get("tactile_deform_available"),
            tactile_raw_images=data.get("tactile_raw_images"),
            tactile_raw_available=data.get("tactile_raw_available"),
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        action_mask=observation.action_mask,
        sample_weight=observation.sample_weight,
        future_latent_pred=observation.future_latent_pred,
        future_latent_true=observation.future_latent_true,
        future_latent_valid_mask=observation.future_latent_valid_mask,
        robot_spline_coefficients=observation.robot_spline_coefficients,
        robot_spline_knots=observation.robot_spline_knots,
        planner_state_belief=observation.planner_state_belief,
        planner_progress_transition=observation.planner_progress_transition,
        planner_uncertainty=observation.planner_uncertainty,
        planner_history_latent=observation.planner_history_latent,
        planner_available=observation.planner_available,
        tactile=observation.tactile,
        tactile_deform_images=observation.tactile_deform_images,
        tactile_deform_available=observation.tactile_deform_available,
        tactile_raw_images=observation.tactile_raw_images,
        tactile_raw_available=observation.tactile_raw_available,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


def reduce_batch_metric(
    values: at.Array,
    sample_weight: at.Array | None,
    *,
    normalize: bool = True,
    eps: float = 1.0e-6,
) -> at.Array:
    """Reduce a batch metric, optionally applying per-sample weights.

    `values` is expected to have shape `[batch..., ...]`, where the leading dimensions
    match `sample_weight.shape`. Any remaining trailing dimensions are averaged within
    each sample before applying the weighted batch reduction.
    """

    values = jnp.asarray(values)
    if sample_weight is None:
        return jnp.mean(values)

    weights = jnp.asarray(sample_weight, dtype=values.dtype)
    batch_ndim = weights.ndim
    if values.ndim < batch_ndim:
        raise ValueError(
            f"values.ndim ({values.ndim}) must be >= sample_weight.ndim ({batch_ndim})"
        )
    if tuple(values.shape[:batch_ndim]) != tuple(weights.shape):
        raise ValueError(
            f"values leading shape {values.shape[:batch_ndim]} does not match sample_weight shape {weights.shape}"
        )

    per_sample = values
    if values.ndim > batch_ndim:
        per_sample = jnp.mean(values, axis=tuple(range(batch_ndim, values.ndim)))

    flat_values = jnp.reshape(per_sample, (-1,))
    flat_weights = jnp.reshape(jnp.maximum(weights, 0.0), (-1,))
    if normalize:
        denom = jnp.maximum(jnp.sum(flat_weights), jnp.asarray(eps, dtype=flat_weights.dtype))
        return jnp.sum(flat_values * flat_weights) / denom
    return jnp.mean(flat_values * flat_weights)


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    def compute_loss_and_metrics(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        chunked_loss = self.compute_loss(rng, observation, actions, train=train)
        loss = jnp.mean(chunked_loss)
        return chunked_loss, {
            "loss": loss,
            "loss_total": loss,
        }

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        metadata_tree = _orbax_metadata_tree(metadata)
        if "params" not in metadata_tree:
            raise KeyError(
                f"Checkpoint metadata at {params_path} does not contain a top-level 'params' item. "
                f"Available keys: {sorted(str(key) for key in metadata_tree.keys())}"
            )
        item = {"params": metadata_tree["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
