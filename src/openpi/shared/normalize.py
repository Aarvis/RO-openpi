import json
import pathlib

import numpy as np
import numpydantic
import pydantic


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = None
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    @staticmethod
    def _broadcast_weights(weights: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
        if weights.shape == original_shape:
            return weights
        # Treat lower-rank weights as applying across leading batch dimensions and
        # broadcast them across any remaining batch axes plus the final feature axis.
        if weights.ndim <= len(original_shape):
            expanded_shape = weights.shape + (1,) * (len(original_shape) - weights.ndim)
            return np.broadcast_to(weights.reshape(expanded_shape), original_shape)
        return np.broadcast_to(weights, original_shape)

    def update(self, batch: np.ndarray, mask: np.ndarray | None = None, weights: np.ndarray | None = None) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): An array where all dimensions except the last are batch dimensions.
        """
        original_shape = batch.shape
        batch = batch.reshape(-1, batch.shape[-1])
        vector_length = batch.shape[1]
        if mask is None:
            mask = np.ones(batch.shape, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != original_shape:
                mask = np.broadcast_to(mask, original_shape)
            mask = mask.reshape(batch.shape)
        if weights is None:
            weights_arr = np.ones(batch.shape, dtype=np.float64)
        else:
            weights_arr = np.asarray(weights, dtype=np.float64)
            if weights_arr.shape != original_shape:
                weights_arr = self._broadcast_weights(weights_arr, original_shape)
            weights_arr = weights_arr.reshape(batch.shape)

        if self._count is None:
            self._count = np.zeros(vector_length, dtype=np.float64)
            self._mean = np.zeros(vector_length, dtype=np.float64)
            self._mean_of_squares = np.zeros(vector_length, dtype=np.float64)
            self._min = np.zeros(vector_length, dtype=np.float64)
            self._max = np.zeros(vector_length, dtype=np.float64)
            self._histograms = [None for _ in range(vector_length)]
            self._bin_edges = [None for _ in range(vector_length)]
        elif vector_length != self._mean.size:
            raise ValueError("The length of new vectors does not match the initialized vector length.")

        for i in range(vector_length):
            values = batch[mask[:, i], i]
            value_weights = weights_arr[mask[:, i], i]
            positive = value_weights > 0.0
            values = values[positive]
            value_weights = value_weights[positive]
            if values.size == 0:
                continue
            values = np.asarray(values, dtype=np.float64)
            value_weights = np.asarray(value_weights, dtype=np.float64)
            weight_sum = float(np.sum(value_weights))
            if weight_sum <= 0.0:
                continue
            if self._count[i] == 0:
                self._mean[i] = np.average(values, weights=value_weights)
                self._mean_of_squares[i] = np.average(values**2, weights=value_weights)
                self._min[i] = np.min(values)
                self._max[i] = np.max(values)
                self._histograms[i] = np.zeros(self._num_quantile_bins)
                self._bin_edges[i] = np.linspace(
                    self._min[i] - 1e-10,
                    self._max[i] + 1e-10,
                    self._num_quantile_bins + 1,
                )
                self._count[i] = weight_sum
                self._update_histogram_dim(i, values, value_weights)
                continue

            new_max = np.max(values)
            new_min = np.min(values)
            if new_max > self._max[i] or new_min < self._min[i]:
                self._max[i] = max(self._max[i], new_max)
                self._min[i] = min(self._min[i], new_min)
                self._adjust_histogram_dim(i)

            old_count = self._count[i]
            new_count = old_count + weight_sum
            batch_mean = np.average(values, weights=value_weights)
            batch_mean_of_squares = np.average(values**2, weights=value_weights)
            self._mean[i] += (batch_mean - self._mean[i]) * (weight_sum / new_count)
            self._mean_of_squares[i] += (batch_mean_of_squares - self._mean_of_squares[i]) * (weight_sum / new_count)
            self._count[i] = new_count
            self._update_histogram_dim(i, values, value_weights)

    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count is None or not np.any(self._count >= 2):
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        mean = np.zeros_like(self._mean)
        stddev = np.ones_like(self._mean)
        valid = self._count > 0
        mean[valid] = self._mean[valid]
        variance = self._mean_of_squares[valid] - self._mean[valid] ** 2
        stddev[valid] = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histogram_dim(self, dim: int):
        """Adjust a histogram when min or max changes for a single dimension."""
        old_edges = self._bin_edges[dim]
        if old_edges is None or self._histograms[dim] is None:
            return
        new_edges = np.linspace(self._min[dim] - 1e-10, self._max[dim] + 1e-10, self._num_quantile_bins + 1)
        new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[dim])
        self._histograms[dim] = new_hist
        self._bin_edges[dim] = new_edges

    def _update_histogram_dim(self, dim: int, values: np.ndarray, weights: np.ndarray | None = None) -> None:
        """Update the histogram for a single dimension."""
        if self._bin_edges[dim] is None or self._histograms[dim] is None:
            return
        hist, _ = np.histogram(values, bins=self._bin_edges[dim], weights=weights)
        self._histograms[dim] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        if self._count is None:
            raise ValueError("No statistics have been accumulated.")

        results = [np.full_like(self._mean, -1.0 if q <= 0.5 else 1.0, dtype=np.float64) for q in quantiles]
        for dim, (count, hist, edges) in enumerate(zip(self._count, self._histograms, self._bin_edges, strict=True)):
            if count <= 0 or hist is None or edges is None:
                continue
            cumsum = np.cumsum(hist)
            for result, q in zip(results, quantiles, strict=True):
                target_count = q * count
                idx = int(np.searchsorted(cumsum, target_count, side="left"))
                idx = min(max(idx, 0), len(edges) - 1)
                result[dim] = edges[idx]
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
