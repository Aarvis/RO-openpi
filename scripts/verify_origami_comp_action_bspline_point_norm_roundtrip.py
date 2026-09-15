from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import BSpline
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.origami_vla_dataset as _origami_vla_dataset


PERCENTILES = (0, 1, 2, 5, 10, 15, 25, 50, 75, 80, 85, 90, 95, 97, 98, 99, 99.9, 99.99, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify B-spline point targets after normalization: generated points/logits -> normalize -> "
            "unnormalize -> solve control points -> compare dense curve against the source local delta spline."
        )
    )
    parser.add_argument("--config-name", type=str, default="pi05_origami_comp_action_bspline_points")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--norm-stats-dir", type=Path, default=None)
    parser.add_argument("--source-local-target-npz-name", type=str, default="local_delta_action_cubic_knotspans10.npz")
    parser.add_argument("--split", choices=("train", "val", "all"), default="train")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--dense-sample-intervals", type=int, default=120)
    parser.add_argument("--roundtrip-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--progress-update-rows", type=int, default=1024)
    return parser.parse_args()


def _resolve(args: argparse.Namespace):
    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.origami_vla is None:
        raise RuntimeError(f"Config {args.config_name!r} did not produce Origami dataset settings.")
    settings = data_config.origami_vla
    if settings.action_source != "bspline_points":
        raise ValueError(
            f"Config {args.config_name!r} uses action_source={settings.action_source!r}; "
            "expected 'bspline_points'."
        )
    if args.dataset_root is not None:
        settings = dataclasses.replace(settings, dataset_root=str(args.dataset_root))
    if args.manifest_root is not None:
        settings = dataclasses.replace(settings, manifest_root=str(args.manifest_root))
    if args.max_rows is not None:
        settings = dataclasses.replace(settings, max_rows=int(args.max_rows))

    norm_stats_dir = args.norm_stats_dir
    if norm_stats_dir is None:
        stats_dir = config.model.origami_vla.action_norm_stats_dir
        if not stats_dir:
            asset_id = config.data.assets.asset_id or config.data.repo_id
            if asset_id is None:
                raise ValueError("Could not infer norm stats dir; pass --norm-stats-dir.")
            stats_dir = str((Path(config.assets_base_dir) / config.name / str(asset_id)).resolve())
        norm_stats_dir = Path(stats_dir)
    return config, settings, Path(norm_stats_dir)


def _load_norm_stats(path: Path, *, use_quantiles: bool):
    loaded = _normalize.load(_download.maybe_download(str(path)))
    required = ("actions_bspline_points", "actions_bspline_width_logits")
    missing = [key for key in required if key not in loaded]
    if missing:
        raise KeyError(f"Missing norm stats {missing} under {path}.")
    if use_quantiles:
        for key in required:
            stats = loaded[key]
            if stats.q01 is None or stats.q99 is None:
                raise ValueError(f"Quantile norm is enabled, but {key} lacks q01/q99.")
    return loaded


def _apply_norm(x: np.ndarray, stats: _normalize.NormStats, *, use_quantiles: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if use_quantiles:
        q01 = np.asarray(stats.q01, dtype=np.float32)[..., : x.shape[-1]]
        q99 = np.asarray(stats.q99, dtype=np.float32)[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1.0e-6) * 2.0 - 1.0
    mean = np.asarray(stats.mean, dtype=np.float32)[..., : x.shape[-1]]
    std = np.asarray(stats.std, dtype=np.float32)[..., : x.shape[-1]]
    return (x - mean) / (std + 1.0e-6)


def _apply_unnorm(x: np.ndarray, stats: _normalize.NormStats, *, use_quantiles: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if use_quantiles:
        q01 = np.asarray(stats.q01, dtype=np.float32)[..., : x.shape[-1]]
        q99 = np.asarray(stats.q99, dtype=np.float32)[..., : x.shape[-1]]
        return (x + 1.0) * 0.5 * (q99 - q01 + 1.0e-6) + q01
    mean = np.asarray(stats.mean, dtype=np.float32)[..., : x.shape[-1]]
    std = np.asarray(stats.std, dtype=np.float32)[..., : x.shape[-1]]
    return x * (std + 1.0e-6) + mean


def _softmax_widths(logits: np.ndarray, *, clip: float = 30.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = np.clip(logits, -clip, clip)
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / max(float(np.sum(exp_logits)), 1.0e-12)


def _clamped_knots_from_widths(widths: np.ndarray, degree: int) -> np.ndarray:
    widths = np.asarray(widths, dtype=np.float64)
    boundaries = np.concatenate([[0.0], np.cumsum(widths)])
    knots = np.ones(widths.size + 2 * degree + 1, dtype=np.float64)
    knots[: degree + 1] = 0.0
    knots[-degree - 1 :] = 1.0
    if widths.size > 1:
        knots[degree + 1 : degree + widths.size] = boundaries[1:-1]
    return knots


def _basis_matrix(knots: np.ndarray, degree: int, u_values: np.ndarray) -> np.ndarray:
    control_count = int(knots.size - degree - 1)
    identity = np.eye(control_count, dtype=np.float64)
    return np.asarray(BSpline(knots, identity, degree)(u_values), dtype=np.float64)


def _source_sample(source_archive, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
    if "coefficients" in source_archive:
        offsets = np.asarray(source_archive["coefficient_offsets"], dtype=np.int64)
        knot_offsets = np.asarray(source_archive["local_knot_offsets"], dtype=np.int64)
        coefficients = np.asarray(source_archive["coefficients"], dtype=np.float64)
        knots = np.asarray(source_archive["local_knots"], dtype=np.float64)
    else:
        offsets = np.asarray(source_archive["control_point_offsets"], dtype=np.int64)
        knot_offsets = np.asarray(source_archive["local_knot_offsets"], dtype=np.int64)
        coefficients = np.asarray(source_archive["local_delta_control_points"], dtype=np.float64)
        knots = np.asarray(source_archive["local_knots"], dtype=np.float64)
    start, end = offsets[sample_index : sample_index + 2]
    knot_start, knot_end = knot_offsets[sample_index : sample_index + 2]
    return coefficients[start:end], knots[knot_start:knot_end]


def _summarize(name: str, values: list[float]) -> None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        print(f"{name}: no values")
        return
    parts = [
        f"count={array.size}",
        f"mean={array.mean():.9g}",
        f"std={array.std():.9g}",
    ]
    for p in PERCENTILES:
        label = "max" if p == 100 else f"p{str(p).replace('.', '_')}"
        parts.append(f"{label}={np.percentile(array, p):.9g}")
    print(f"{name}: " + ", ".join(parts))


def main() -> int:
    args = parse_args()
    config, settings, norm_stats_dir = _resolve(args)
    rows_df = _origami_vla_dataset.load_manifest_rows(settings, args.split)
    if rows_df.empty:
        raise RuntimeError(f"No rows found for split {args.split!r} under {settings.manifest_root}.")
    rows_df = rows_df.drop_duplicates(
        subset=["episode_uid", "frame_position", "local_target_npz_sample_index"]
    ).reset_index(drop=True)

    norm_stats = _load_norm_stats(norm_stats_dir, use_quantiles=bool(config.model.origami_vla.use_quantile_norm))
    point_stats = norm_stats["actions_bspline_points"]
    width_stats = norm_stats["actions_bspline_width_logits"]

    grouped_rows: dict[str, list[dict[str, int]]] = defaultdict(list)
    for row in rows_df.to_dict(orient="records"):
        grouped_rows[str(row["episode_uid"])].append(
            {
                "frame_position": int(row["frame_position"]),
                "sample_index": int(row["local_target_npz_sample_index"]),
            }
        )

    dataset_root = Path(settings.dataset_root)
    degree = int(settings.degree)
    dense_u = np.linspace(0.0, 1.0, int(args.dense_sample_intervals) + 1, dtype=np.float64)
    use_quantiles = bool(config.model.origami_vla.use_quantile_norm)

    print("Origami B-spline point norm round-trip verification")
    print(f"  config_name       : {args.config_name}")
    print(f"  dataset_root      : {dataset_root}")
    print(f"  manifest_root     : {settings.manifest_root}")
    print(f"  target npz        : {settings.local_target_npz_name}")
    print(f"  source npz        : {args.source_local_target_npz_name}")
    print(f"  norm_stats_dir    : {norm_stats_dir}")
    print(f"  split             : {args.split}")
    print(f"  rows              : {len(rows_df):,}")
    print(f"  dense samples     : {dense_u.size}")
    print(f"  use_quantile_norm : {use_quantiles}")

    curve_mae: list[float] = []
    curve_max_abs: list[float] = []
    point_roundtrip_max_abs: list[float] = []
    width_roundtrip_max_abs: list[float] = []
    width_sum_abs: list[float] = []
    failures = 0

    progress = tqdm(total=len(rows_df), desc="Verify B-spline point roundtrip", unit="row", dynamic_ncols=True)
    for episode_uid in sorted(grouped_rows):
        arrays_root = dataset_root / "episodes" / episode_uid / "arrays"
        target_archive = np.load(arrays_root / settings.local_target_npz_name, allow_pickle=False)
        source_archive = np.load(arrays_root / args.source_local_target_npz_name, allow_pickle=False)
        try:
            for row in grouped_rows[episode_uid]:
                sample_index = int(row["sample_index"])
                points = np.asarray(target_archive["points"][sample_index], dtype=np.float32)
                width_logits = np.asarray(target_archive["width_logits"][sample_index], dtype=np.float32)
                u_values = np.asarray(target_archive["u_values"][sample_index], dtype=np.float64)

                points_rt = _apply_unnorm(
                    _apply_norm(points, point_stats, use_quantiles=use_quantiles),
                    point_stats,
                    use_quantiles=use_quantiles,
                )
                width_logits_rt = _apply_unnorm(
                    _apply_norm(width_logits, width_stats, use_quantiles=use_quantiles),
                    width_stats,
                    use_quantiles=use_quantiles,
                )
                point_roundtrip_max_abs.append(float(np.max(np.abs(points_rt - points))))
                width_roundtrip_max_abs.append(float(np.max(np.abs(width_logits_rt - width_logits))))

                widths = _softmax_widths(width_logits_rt, clip=30.0)
                width_sum_abs.append(float(abs(np.sum(widths) - 1.0)))
                knots = _clamped_knots_from_widths(widths, degree)
                basis = _basis_matrix(knots, degree, u_values)
                coefficients, *_ = np.linalg.lstsq(basis, np.asarray(points_rt, dtype=np.float64), rcond=None)
                reconstructed_curve = BSpline(knots, coefficients, degree)(dense_u)

                source_coefficients, source_knots = _source_sample(source_archive, sample_index)
                source_curve = BSpline(source_knots, source_coefficients, degree)(dense_u)
                diff = np.asarray(reconstructed_curve - source_curve, dtype=np.float64)
                sample_mae = float(np.mean(np.abs(diff)))
                sample_max = float(np.max(np.abs(diff)))
                curve_mae.append(sample_mae)
                curve_max_abs.append(sample_max)
                if sample_max > float(args.roundtrip_tolerance):
                    failures += 1

                progress.update(1)
                if len(curve_mae) % int(args.progress_update_rows) == 0:
                    progress.set_postfix(mae=f"{sample_mae:.3g}", max=f"{sample_max:.3g}")
        finally:
            target_archive.close()
            source_archive.close()

    progress.close()
    _summarize("curve_mae_65d", curve_mae)
    _summarize("curve_max_abs_dim", curve_max_abs)
    _summarize("point_norm_roundtrip_max_abs", point_roundtrip_max_abs)
    _summarize("width_logit_norm_roundtrip_max_abs", width_roundtrip_max_abs)
    _summarize("width_sum_abs_error", width_sum_abs)
    print(f"Rows above tolerance {args.roundtrip_tolerance:g}: {failures}")
    if failures:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
