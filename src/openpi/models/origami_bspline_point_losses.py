from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

import openpi.models.origami_spline_losses as _spline


@dataclasses.dataclass(frozen=True)
class PackedBsplinePointNormStats:
    points: _spline.ActionNormStats
    width_logits: _spline.ActionNormStats


def denormalize_packed_bspline_point_actions(
    actions: jax.Array,
    *,
    stats: PackedBsplinePointNormStats | None,
    use_quantiles: bool,
    point_count: int,
    width_logit_count: int,
) -> jax.Array:
    if stats is None:
        return actions
    denormalized = actions
    points = _spline.denormalize_actions(
        actions[:, :point_count, :],
        stats=stats.points,
        use_quantiles=use_quantiles,
    )
    width_logits = _spline.denormalize_actions(
        actions[:, point_count, :width_logit_count],
        stats=stats.width_logits,
        use_quantiles=use_quantiles,
    )
    denormalized = denormalized.at[:, :point_count, :].set(points)
    denormalized = denormalized.at[:, point_count, :width_logit_count].set(width_logits)
    return denormalized


def _batched_basis_matrix(
    u_values: jax.Array,
    knots: jax.Array,
    *,
    degree: int,
    control_point_count: int,
    num_ctrl: jax.Array,
    denominator_eps: float,
) -> jax.Array:
    eps = denominator_eps
    left = knots[:, :-1]
    right = knots[:, 1:]
    u_expanded = u_values[:, :, None]
    base = ((u_expanded >= left[:, None, :]) & (u_expanded < right[:, None, :])).astype(u_values.dtype)
    basis = base[:, :, :control_point_count]

    last_index = jnp.clip(num_ctrl - 1, a_min=0, a_max=control_point_count - 1)
    is_one = jnp.isclose(u_values, 1.0, atol=eps)
    one_hot = jax.nn.one_hot(last_index, control_point_count, dtype=u_values.dtype)
    basis = jnp.where(is_one[:, :, None], one_hot[:, None, :], basis)

    for current_degree in range(1, degree + 1):
        next_basis = []
        for index in range(control_point_count):
            denom_left = knots[:, index + current_degree] - knots[:, index]
            safe_denom_left = jnp.where(denom_left > eps, denom_left, 1.0)
            left_coeff = (u_values - knots[:, index][:, None]) / safe_denom_left[:, None]
            left_coeff = jnp.where(denom_left[:, None] > eps, left_coeff, 0.0)
            left_term = left_coeff * basis[:, :, index]
            if index + 1 < control_point_count:
                denom_right = knots[:, index + current_degree + 1] - knots[:, index + 1]
                safe_denom_right = jnp.where(denom_right > eps, denom_right, 1.0)
                right_coeff = (knots[:, index + current_degree + 1][:, None] - u_values) / safe_denom_right[:, None]
                right_coeff = jnp.where(denom_right[:, None] > eps, right_coeff, 0.0)
                right_term = right_coeff * basis[:, :, index + 1]
            else:
                right_term = 0.0
            next_basis.append(left_term + right_term)
        basis = jnp.stack(next_basis, axis=-1)
    return basis


def _phase_points_from_widths(
    widths: jax.Array,
    span_mask: jax.Array,
    *,
    degree: int,
    control_point_count: int,
    width_logit_count: int,
    midpoint_spans_1based: tuple[int, ...] = (1, 4, 7, 10),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    knots, num_ctrl = _spline._build_clamped_knots(
        widths,
        span_mask,
        degree=degree,
        max_span_count=width_logit_count,
    )
    greville = []
    for index in range(control_point_count):
        greville.append(jnp.mean(knots[:, index + 1 : index + degree + 1], axis=-1))
    greville_u = jnp.stack(greville, axis=-1)

    boundaries = jnp.concatenate(
        [
            jnp.zeros((widths.shape[0], 1), dtype=widths.dtype),
            jnp.cumsum(widths, axis=-1),
        ],
        axis=-1,
    )
    midpoint_values = []
    for span_1based in midpoint_spans_1based:
        left = boundaries[:, int(span_1based) - 1]
        right = boundaries[:, int(span_1based)]
        midpoint_values.append(0.5 * (left + right))
    midpoint_u = jnp.stack(midpoint_values, axis=-1)
    u_values = jnp.concatenate([greville_u, midpoint_u], axis=-1)
    order = jnp.argsort(u_values, axis=-1)
    sorted_u = jnp.take_along_axis(u_values, order, axis=-1)
    return sorted_u, knots, num_ctrl


def _solve_control_points_from_points(
    points: jax.Array,
    widths: jax.Array,
    span_mask: jax.Array,
    *,
    degree: int,
    control_point_count: int,
    width_logit_count: int,
    denominator_eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    u_values, knots, num_ctrl = _phase_points_from_widths(
        widths,
        span_mask,
        degree=degree,
        control_point_count=control_point_count,
        width_logit_count=width_logit_count,
    )
    basis = _batched_basis_matrix(
        u_values,
        knots,
        degree=degree,
        control_point_count=control_point_count,
        num_ctrl=num_ctrl,
        denominator_eps=denominator_eps,
    )
    pinv = jnp.linalg.pinv(basis, rcond=denominator_eps)
    control_points = jnp.einsum("bcs,bsd->bcd", pinv, points, precision=jax.lax.Precision.HIGHEST)
    singular_values = jnp.linalg.svd(basis, compute_uv=False)
    return control_points, basis, singular_values


def compute_auxiliary_metrics(
    pred_actions_norm: jax.Array,
    target_actions_norm: jax.Array,
    action_mask: jax.Array,
    *,
    stats: PackedBsplinePointNormStats | None,
    use_quantiles: bool,
    degree: int,
    point_count: int,
    control_point_count: int,
    width_logit_count: int,
    sample_intervals: int,
    smooth_l1_beta: float,
    width_min: float,
    denominator_eps: float,
    softmax_clip: float | None,
) -> dict[str, jax.Array]:
    pred_actions_norm = jnp.asarray(pred_actions_norm, dtype=jnp.float32)
    target_actions_norm = jnp.asarray(target_actions_norm, dtype=jnp.float32)
    action_mask = jnp.asarray(action_mask, dtype=jnp.bool_)

    pred_actions = denormalize_packed_bspline_point_actions(
        pred_actions_norm,
        stats=stats,
        use_quantiles=use_quantiles,
        point_count=point_count,
        width_logit_count=width_logit_count,
    )
    target_actions = denormalize_packed_bspline_point_actions(
        target_actions_norm,
        stats=stats,
        use_quantiles=use_quantiles,
        point_count=point_count,
        width_logit_count=width_logit_count,
    )

    point_mask = action_mask[:, :point_count, :]
    span_mask = action_mask[:, point_count, :width_logit_count]
    pred_points = jnp.where(point_mask, pred_actions[:, :point_count, :], 0.0)
    target_points = jnp.where(point_mask, target_actions[:, :point_count, :], 0.0)
    pred_widths = _spline._masked_softmax_widths(
        pred_actions[:, point_count, :width_logit_count],
        span_mask,
        min_width=width_min,
        softmax_clip=softmax_clip,
    )
    target_widths = _spline._masked_softmax_widths(
        target_actions[:, point_count, :width_logit_count],
        span_mask,
        min_width=width_min,
        softmax_clip=softmax_clip,
    )

    pred_control, pred_basis, pred_singular_values = _solve_control_points_from_points(
        pred_points,
        pred_widths,
        span_mask,
        degree=degree,
        control_point_count=control_point_count,
        width_logit_count=width_logit_count,
        denominator_eps=denominator_eps,
    )
    target_control, target_basis, target_singular_values = _solve_control_points_from_points(
        target_points,
        target_widths,
        span_mask,
        degree=degree,
        control_point_count=control_point_count,
        width_logit_count=width_logit_count,
        denominator_eps=denominator_eps,
    )
    del pred_basis

    u = jnp.linspace(0.0, 1.0, int(sample_intervals) + 1, dtype=jnp.float32)
    pred_curve = _spline.evaluate_batch_splines(
        pred_control,
        pred_widths,
        span_mask,
        degree=degree,
        max_control_points=control_point_count,
        max_span_count=width_logit_count,
        u=u,
        denominator_eps=denominator_eps,
    )
    target_curve = _spline.evaluate_batch_splines(
        target_control,
        target_widths,
        span_mask,
        degree=degree,
        max_control_points=control_point_count,
        max_span_count=width_logit_count,
        u=u,
        denominator_eps=denominator_eps,
    )

    curve_diff = pred_curve - target_curve
    start_diff = pred_curve[:, 0, :] - target_curve[:, 0, :]
    end_diff = pred_curve[:, -1, :] - target_curve[:, -1, :]
    width_diff = pred_widths - target_widths
    span_mask_f32 = jnp.asarray(span_mask, dtype=jnp.float32)
    span_denom = jnp.clip(jnp.sum(span_mask_f32, axis=-1), a_min=1.0)

    singular_max = jnp.max(target_singular_values, axis=-1)
    singular_min = jnp.min(target_singular_values, axis=-1)
    pred_singular_min = jnp.min(pred_singular_values, axis=-1)

    return {
        "curve": jnp.mean(_spline._smooth_l1(curve_diff, smooth_l1_beta), axis=(1, 2)),
        "start": jnp.mean(_spline._smooth_l1(start_diff, smooth_l1_beta), axis=-1),
        "end": jnp.mean(_spline._smooth_l1(end_diff, smooth_l1_beta), axis=-1),
        "width": jnp.sum(_spline._smooth_l1(width_diff, smooth_l1_beta) * span_mask_f32, axis=-1) / span_denom,
        "curve_mae_rad": jnp.mean(jnp.abs(curve_diff), axis=(1, 2)),
        "start_mae_rad": jnp.mean(jnp.abs(start_diff), axis=-1),
        "end_mae_rad": jnp.mean(jnp.abs(end_diff), axis=-1),
        "width_mae": jnp.sum(jnp.abs(width_diff) * span_mask_f32, axis=-1) / span_denom,
        "condition_number": singular_max / jnp.clip(singular_min, a_min=denominator_eps),
        "sigma_min": singular_min,
        "pred_sigma_min": pred_singular_min,
        "finite": (
            jnp.all(jnp.isfinite(curve_diff), axis=(1, 2))
            & jnp.all(jnp.isfinite(target_basis), axis=(1, 2))
        ).astype(jnp.float32),
    }
