from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats
OrigamiSplineSpanRepresentation: TypeAlias = str


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class SetPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def _span_widths_to_centered_logits_np(widths: np.ndarray, *, eps: float = 1.0e-6) -> np.ndarray:
    widths = np.asarray(widths, dtype=np.float32)
    logits = np.log(np.clip(widths, eps, None))
    return logits - np.mean(logits, axis=-1, keepdims=True)


@dataclasses.dataclass(frozen=True)
class OrigamiSplineNormalize(DataTransformFn):
    """Normalize Origami VLA packed spline actions with separate control-point and span stats."""

    state_stats: NormStats | None
    tactile_prompt_stats: NormStats | None
    control_point_stats: NormStats | None
    span_stats: NormStats | None
    max_control_points: int
    max_span_count: int
    span_representation: OrigamiSplineSpanRepresentation = "physical_widths"
    use_quantiles: bool = False
    strict: bool = False

    def __post_init__(self):
        if self.span_representation not in ("physical_widths", "logits"):
            raise ValueError(f"Unsupported Origami spline span representation: {self.span_representation!r}")
        span_stats_name = "actions_span_logits" if self.span_representation == "logits" else "actions_span_widths"
        if self.use_quantiles:
            for name, stats in (
                ("state", self.state_stats),
                ("tactile_prompt", self.tactile_prompt_stats),
                ("actions_control_points", self.control_point_stats),
                (span_stats_name, self.span_stats),
            ):
                if stats is None:
                    continue
                if stats.q01 is None or stats.q99 is None:
                    raise ValueError(f"Quantile stats required for {name} when use_quantiles=True.")

    def __call__(self, data: DataDict) -> DataDict:
        if "state" in data:
            if self.state_stats is None:
                if self.strict:
                    raise ValueError("OrigamiSplineNormalize expected state stats but none were provided.")
            else:
                data["state"] = self._apply(np.asarray(data["state"]), self.state_stats)
        elif self.strict and self.state_stats is not None:
            raise ValueError("OrigamiSplineNormalize expected a 'state' key in the data.")

        if "tactile_prompt" in data:
            if self.tactile_prompt_stats is None:
                if self.strict:
                    raise ValueError("OrigamiSplineNormalize expected tactile_prompt stats but none were provided.")
            else:
                data["tactile_prompt"] = self._apply(np.asarray(data["tactile_prompt"]), self.tactile_prompt_stats)
        elif self.strict and self.tactile_prompt_stats is not None:
            raise ValueError("OrigamiSplineNormalize expected a 'tactile_prompt' key in the data.")

        if "actions" in data:
            if self.control_point_stats is None or self.span_stats is None:
                if self.strict:
                    raise ValueError(
                        "OrigamiSplineNormalize expected both control-point and span stats for packed actions."
                    )
                return data
            data["actions"] = self._normalize_actions(np.asarray(data["actions"]))
        elif self.strict and (self.control_point_stats is not None or self.span_stats is not None):
            raise ValueError("OrigamiSplineNormalize expected an 'actions' key in the data.")
        return data

    def _apply(self, x: np.ndarray, stats: NormStats) -> np.ndarray:
        if self.use_quantiles:
            assert stats.q01 is not None
            assert stats.q99 is not None
            q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        normalized = np.array(actions, copy=True)
        normalized[..., : self.max_control_points, :] = self._apply(
            normalized[..., : self.max_control_points, :],
            self.control_point_stats,
        )
        if normalized.shape[-2] <= self.max_control_points:
            raise ValueError(
                f"Packed Origami actions must have at least {self.max_control_points + 1} rows, "
                f"got shape {normalized.shape}."
            )
        spans = normalized[..., self.max_control_points, : self.max_span_count]
        if self.span_representation == "logits":
            spans = _span_widths_to_centered_logits_np(spans)
        normalized[..., self.max_control_points, : self.max_span_count] = self._apply(spans, self.span_stats)
        return normalized


@dataclasses.dataclass(frozen=True)
class OrigamiSplineUnnormalize(DataTransformFn):
    """Inverse of OrigamiSplineNormalize for packed spline actions."""

    state_stats: NormStats | None
    tactile_prompt_stats: NormStats | None
    control_point_stats: NormStats | None
    span_stats: NormStats | None
    max_control_points: int
    max_span_count: int
    span_representation: OrigamiSplineSpanRepresentation = "physical_widths"
    use_quantiles: bool = False

    def __post_init__(self):
        if self.span_representation not in ("physical_widths", "logits"):
            raise ValueError(f"Unsupported Origami spline span representation: {self.span_representation!r}")
        span_stats_name = "actions_span_logits" if self.span_representation == "logits" else "actions_span_widths"
        if self.use_quantiles:
            for name, stats in (
                ("state", self.state_stats),
                ("tactile_prompt", self.tactile_prompt_stats),
                ("actions_control_points", self.control_point_stats),
                (span_stats_name, self.span_stats),
            ):
                if stats is None:
                    continue
                if stats.q01 is None or stats.q99 is None:
                    raise ValueError(f"Quantile stats required for {name} when use_quantiles=True.")

    def __call__(self, data: DataDict) -> DataDict:
        if "state" in data and self.state_stats is not None:
            data["state"] = self._apply(np.asarray(data["state"]), self.state_stats)
        if "tactile_prompt" in data and self.tactile_prompt_stats is not None:
            data["tactile_prompt"] = self._apply(np.asarray(data["tactile_prompt"]), self.tactile_prompt_stats)
        if "actions" in data and self.control_point_stats is not None and self.span_stats is not None:
            data["actions"] = self._unnormalize_actions(np.asarray(data["actions"]))
        return data

    def _apply(self, x: np.ndarray, stats: NormStats) -> np.ndarray:
        if self.use_quantiles:
            assert stats.q01 is not None
            assert stats.q99 is not None
            q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return x * (std + 1e-6) + mean

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        unnormalized = np.array(actions, copy=True)
        unnormalized[..., : self.max_control_points, :] = self._apply(
            unnormalized[..., : self.max_control_points, :],
            self.control_point_stats,
        )
        if unnormalized.shape[-2] <= self.max_control_points:
            raise ValueError(
                f"Packed Origami actions must have at least {self.max_control_points + 1} rows, "
                f"got shape {unnormalized.shape}."
            )
        unnormalized[..., self.max_control_points, : self.max_span_count] = self._apply(
            unnormalized[..., self.max_control_points, : self.max_span_count],
            self.span_stats,
        )
        return unnormalized


@dataclasses.dataclass(frozen=True)
class OrigamiBsplinePointsNormalize(DataTransformFn):
    """Normalize packed B-spline point actions with separate point and width-logit stats."""

    state_stats: NormStats | None
    tactile_prompt_stats: NormStats | None
    point_stats: NormStats | None
    width_logit_stats: NormStats | None
    point_count: int
    width_logit_count: int
    use_quantiles: bool = False
    strict: bool = False

    def __post_init__(self):
        if self.use_quantiles:
            for name, stats in (
                ("state", self.state_stats),
                ("tactile_prompt", self.tactile_prompt_stats),
                ("actions_bspline_points", self.point_stats),
                ("actions_bspline_width_logits", self.width_logit_stats),
            ):
                if stats is None:
                    continue
                if stats.q01 is None or stats.q99 is None:
                    raise ValueError(f"Quantile stats required for {name} when use_quantiles=True.")

    def __call__(self, data: DataDict) -> DataDict:
        if "state" in data:
            if self.state_stats is None:
                if self.strict:
                    raise ValueError("OrigamiBsplinePointsNormalize expected state stats but none were provided.")
            else:
                data["state"] = self._apply(np.asarray(data["state"]), self.state_stats)
        elif self.strict and self.state_stats is not None:
            raise ValueError("OrigamiBsplinePointsNormalize expected a 'state' key in the data.")

        if "tactile_prompt" in data:
            if self.tactile_prompt_stats is None:
                if self.strict:
                    raise ValueError(
                        "OrigamiBsplinePointsNormalize expected tactile_prompt stats but none were provided."
                    )
            else:
                data["tactile_prompt"] = self._apply(np.asarray(data["tactile_prompt"]), self.tactile_prompt_stats)
        elif self.strict and self.tactile_prompt_stats is not None:
            raise ValueError("OrigamiBsplinePointsNormalize expected a 'tactile_prompt' key in the data.")

        if "actions" in data:
            if self.point_stats is None or self.width_logit_stats is None:
                if self.strict:
                    raise ValueError(
                        "OrigamiBsplinePointsNormalize expected both point and width-logit stats for packed actions."
                    )
                return data
            data["actions"] = self._normalize_actions(np.asarray(data["actions"]))
        elif self.strict and (self.point_stats is not None or self.width_logit_stats is not None):
            raise ValueError("OrigamiBsplinePointsNormalize expected an 'actions' key in the data.")
        return data

    def _apply(self, x: np.ndarray, stats: NormStats) -> np.ndarray:
        if self.use_quantiles:
            assert stats.q01 is not None
            assert stats.q99 is not None
            q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        normalized = np.array(actions, copy=True)
        if normalized.shape[-2] <= self.point_count:
            raise ValueError(
                f"Packed B-spline point actions must have at least {self.point_count + 1} rows, "
                f"got shape {normalized.shape}."
            )
        normalized[..., : self.point_count, :] = self._apply(
            normalized[..., : self.point_count, :],
            self.point_stats,
        )
        normalized[..., self.point_count, : self.width_logit_count] = self._apply(
            normalized[..., self.point_count, : self.width_logit_count],
            self.width_logit_stats,
        )
        return normalized


@dataclasses.dataclass(frozen=True)
class OrigamiBsplinePointsUnnormalize(DataTransformFn):
    """Inverse of OrigamiBsplinePointsNormalize."""

    state_stats: NormStats | None
    tactile_prompt_stats: NormStats | None
    point_stats: NormStats | None
    width_logit_stats: NormStats | None
    point_count: int
    width_logit_count: int
    use_quantiles: bool = False

    def __post_init__(self):
        if self.use_quantiles:
            for name, stats in (
                ("state", self.state_stats),
                ("tactile_prompt", self.tactile_prompt_stats),
                ("actions_bspline_points", self.point_stats),
                ("actions_bspline_width_logits", self.width_logit_stats),
            ):
                if stats is None:
                    continue
                if stats.q01 is None or stats.q99 is None:
                    raise ValueError(f"Quantile stats required for {name} when use_quantiles=True.")

    def __call__(self, data: DataDict) -> DataDict:
        if "state" in data and self.state_stats is not None:
            data["state"] = self._apply(np.asarray(data["state"]), self.state_stats)
        if "tactile_prompt" in data and self.tactile_prompt_stats is not None:
            data["tactile_prompt"] = self._apply(np.asarray(data["tactile_prompt"]), self.tactile_prompt_stats)
        if "actions" in data and self.point_stats is not None and self.width_logit_stats is not None:
            data["actions"] = self._unnormalize_actions(np.asarray(data["actions"]))
        return data

    def _apply(self, x: np.ndarray, stats: NormStats) -> np.ndarray:
        if self.use_quantiles:
            assert stats.q01 is not None
            assert stats.q99 is not None
            q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return x * (std + 1e-6) + mean

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        unnormalized = np.array(actions, copy=True)
        if unnormalized.shape[-2] <= self.point_count:
            raise ValueError(
                f"Packed B-spline point actions must have at least {self.point_count + 1} rows, "
                f"got shape {unnormalized.shape}."
            )
        unnormalized[..., : self.point_count, :] = self._apply(
            unnormalized[..., : self.point_count, :],
            self.point_stats,
        )
        unnormalized[..., self.point_count, : self.width_logit_count] = self._apply(
            unnormalized[..., self.point_count, : self.width_logit_count],
            self.width_logit_stats,
        )
        return unnormalized


def make_normalize_transform(
    norm_stats: at.PyTree[NormStats] | None,
    *,
    use_quantiles: bool = False,
    strict: bool = False,
    origami_max_control_points: int | None = None,
    origami_max_span_count: int | None = None,
    origami_action_mode: str | None = None,
    origami_spline_span_representation: OrigamiSplineSpanRepresentation = "physical_widths",
) -> DataTransformFn:
    if origami_action_mode == "spline" and (
        origami_max_control_points is not None or origami_max_span_count is not None
    ):
        if origami_max_control_points is None or origami_max_span_count is None:
            raise ValueError("Both origami_max_control_points and origami_max_span_count must be provided together.")
        stats_dict = norm_stats or {}
        span_key = (
            "actions_span_logits" if origami_spline_span_representation == "logits" else "actions_span_widths"
        )
        return OrigamiSplineNormalize(
            state_stats=stats_dict.get("state"),
            tactile_prompt_stats=stats_dict.get("tactile_prompt"),
            control_point_stats=stats_dict.get("actions_control_points"),
            span_stats=stats_dict.get(span_key),
            max_control_points=origami_max_control_points,
            max_span_count=origami_max_span_count,
            span_representation=origami_spline_span_representation,
            use_quantiles=use_quantiles,
            strict=strict,
        )
    if origami_action_mode == "bspline_points" and (
        origami_max_control_points is not None or origami_max_span_count is not None
    ):
        if origami_max_control_points is None or origami_max_span_count is None:
            raise ValueError("Both origami_max_control_points and origami_max_span_count must be provided together.")
        stats_dict = norm_stats or {}
        return OrigamiBsplinePointsNormalize(
            state_stats=stats_dict.get("state"),
            tactile_prompt_stats=stats_dict.get("tactile_prompt"),
            point_stats=stats_dict.get("actions_bspline_points"),
            width_logit_stats=stats_dict.get("actions_bspline_width_logits"),
            point_count=origami_max_control_points,
            width_logit_count=origami_max_span_count,
            use_quantiles=use_quantiles,
            strict=strict,
        )
    return Normalize(norm_stats, use_quantiles=use_quantiles, strict=strict)


def make_unnormalize_transform(
    norm_stats: at.PyTree[NormStats] | None,
    *,
    use_quantiles: bool = False,
    origami_max_control_points: int | None = None,
    origami_max_span_count: int | None = None,
    origami_action_mode: str | None = None,
    origami_spline_span_representation: OrigamiSplineSpanRepresentation = "physical_widths",
) -> DataTransformFn:
    if origami_action_mode == "spline" and (
        origami_max_control_points is not None or origami_max_span_count is not None
    ):
        if origami_max_control_points is None or origami_max_span_count is None:
            raise ValueError("Both origami_max_control_points and origami_max_span_count must be provided together.")
        stats_dict = norm_stats or {}
        span_key = (
            "actions_span_logits" if origami_spline_span_representation == "logits" else "actions_span_widths"
        )
        return OrigamiSplineUnnormalize(
            state_stats=stats_dict.get("state"),
            tactile_prompt_stats=stats_dict.get("tactile_prompt"),
            control_point_stats=stats_dict.get("actions_control_points"),
            span_stats=stats_dict.get(span_key),
            max_control_points=origami_max_control_points,
            max_span_count=origami_max_span_count,
            span_representation=origami_spline_span_representation,
            use_quantiles=use_quantiles,
        )
    if origami_action_mode == "bspline_points" and (
        origami_max_control_points is not None or origami_max_span_count is not None
    ):
        if origami_max_control_points is None or origami_max_span_count is None:
            raise ValueError("Both origami_max_control_points and origami_max_span_count must be provided together.")
        stats_dict = norm_stats or {}
        return OrigamiBsplinePointsUnnormalize(
            state_stats=stats_dict.get("state"),
            tactile_prompt_stats=stats_dict.get("tactile_prompt"),
            point_stats=stats_dict.get("actions_bspline_points"),
            width_logit_stats=stats_dict.get("actions_bspline_width_logits"),
            point_count=origami_max_control_points,
            width_logit_count=origami_max_span_count,
            use_quantiles=use_quantiles,
        )
    return Unnormalize(norm_stats, use_quantiles=use_quantiles)


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False
    discrete_tactile_input: bool = False
    tactile_key: str = "tactile_prompt"
    tactile_mask_key: str = "tactile_prompt_mask"
    clip_discrete_inputs: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
            state_mask = data.get("state_mask")
        else:
            state = None
            state_mask = None

        if self.discrete_tactile_input:
            if (tactile := data.get(self.tactile_key, None)) is None:
                raise ValueError(f"Tactile prompt input is required at key {self.tactile_key!r}.")
            tactile_mask = data.get(self.tactile_mask_key)
        else:
            tactile = None
            tactile_mask = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(
            prompt,
            state,
            state_mask,
            tactile,
            tactile_mask,
            clip_discrete_inputs=self.clip_discrete_inputs,
        )
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        if "action_mask" in data:
            data["action_mask"] = pad_to_dim(data["action_mask"], self.model_action_dim, axis=-1, value=False)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
