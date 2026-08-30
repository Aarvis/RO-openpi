import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CompositeWeightLoader(WeightLoader):
    """Applies multiple weight loaders in sequence."""

    loaders: tuple[WeightLoader, ...]

    def load(self, params: at.Params) -> at.Params:
        loaded = params
        for loader in self.loaders:
            loaded = loader.load(loaded)
        return loaded


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    missing_regex: str = ".*lora.*"

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add explicitly allowed missing weights from the freshly initialized reference tree.
        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)


@dataclasses.dataclass(frozen=True)
class NpzSubsetWeightLoader(WeightLoader):
    """Loads named parameters from a flat Flax/Linen `.npz` file into an initialized tree."""

    params_path: str
    key_prefix: str | None = None
    strict: bool = True
    min_matched: int = 1

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(self.params_path)
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))

        if self.key_prefix:
            prefix = self.key_prefix.strip("/")
            flat_params = {f"{prefix}/{key}": value for key, value in flat_params.items()}

        loaded_params = flax.traverse_util.unflatten_dict(flat_params, sep="/")
        return _merge_subset_params(
            loaded_params,
            params,
            strict=self.strict,
            min_matched=self.min_matched,
            source=str(path),
        )


@dataclasses.dataclass(frozen=True)
class OrbaxSubsetWeightLoader(WeightLoader):
    """Loads a nested OpenPI/Orbax params tree into an initialized model subtree."""

    params_path: str
    key_prefix: str | None = None
    strict: bool = True
    min_matched: int = 1

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        if self.key_prefix:
            prefix = self.key_prefix.strip("/")
            flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
            loaded_params = flax.traverse_util.unflatten_dict(
                {f"{prefix}/{key}": value for key, value in flat_loaded.items()},
                sep="/",
            )
        return _merge_subset_params(
            loaded_params,
            params,
            strict=self.strict,
            min_matched=self.min_matched,
            source=str(self.params_path),
        )


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    pattern = re.compile(missing_regex)

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref_value = flat_ref[k]
            loaded_shape = tuple(v.shape) if hasattr(v, "shape") else None
            ref_shape = tuple(ref_value.shape) if hasattr(ref_value, "shape") else None

            if loaded_shape is not None and ref_shape is not None and loaded_shape != ref_shape:
                if pattern.fullmatch(k):
                    logger.info(
                        "Skipping checkpoint parameter %s due to shape mismatch: checkpoint=%s model=%s; "
                        "using fresh initialization from missing_regex.",
                        k,
                        loaded_shape,
                        ref_shape,
                    )
                    continue
                raise ValueError(
                    f"Checkpoint parameter shape mismatch for {k}: checkpoint={loaded_shape}, model={ref_shape}"
                )

            result[k] = v.astype(ref_value.dtype) if v.dtype != ref_value.dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_subset_params(
    loaded_params: at.Params,
    params: at.Params,
    *,
    strict: bool,
    min_matched: int,
    source: str,
) -> at.Params:
    """Overwrites a strict subset of a reference tree with loaded arrays."""
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    result = dict(flat_ref)
    unexpected: list[str] = []
    matched = 0

    for key, value in flat_loaded.items():
        if key not in flat_ref:
            unexpected.append(key)
            continue

        ref_value = flat_ref[key]
        loaded_shape = tuple(value.shape) if hasattr(value, "shape") else None
        ref_shape = tuple(ref_value.shape) if hasattr(ref_value, "shape") else None
        if loaded_shape is not None and ref_shape is not None and loaded_shape != ref_shape:
            raise ValueError(
                f"Loaded parameter shape mismatch for {key}: source={loaded_shape} model={ref_shape}"
            )

        ref_dtype = getattr(ref_value, "dtype", None)
        if ref_dtype is not None and hasattr(value, "dtype") and value.dtype != ref_dtype:
            value = value.astype(ref_dtype)
        result[key] = value
        matched += 1

    if strict and unexpected:
        preview = ", ".join(unexpected[:20])
        suffix = "" if len(unexpected) <= 20 else f", ... and {len(unexpected) - 20} more"
        raise KeyError(f"{source} contains parameters that are not present in the model: {preview}{suffix}")
    if matched < min_matched:
        raise RuntimeError(f"Loaded only {matched} matching parameters from {source}; expected at least {min_matched}.")

    logger.info("Loaded %d parameter arrays from %s", matched, source)
    return flax.traverse_util.unflatten_dict(result, sep="/")
