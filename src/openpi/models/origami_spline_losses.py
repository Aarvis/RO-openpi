from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class ActionNormStats:
    mean: jax.Array
    std: jax.Array
    q01: jax.Array | None = None
    q99: jax.Array | None = None


@dataclasses.dataclass(frozen=True)
class PackedActionNormStats:
    control_points: ActionNormStats
    span_widths: ActionNormStats


def denormalize_actions(
    actions: jax.Array,
    *,
    stats: ActionNormStats | None,
    use_quantiles: bool,
) -> jax.Array:
    if stats is None:
        return actions
    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile action denormalization requested, but q01/q99 are missing.")
        return (actions + 1.0) * 0.5 * (stats.q99 + 1e-6 - stats.q01) + stats.q01
    return actions * (stats.std + 1e-6) + stats.mean


def denormalize_packed_actions(
    actions: jax.Array,
    *,
    stats: PackedActionNormStats | None,
    use_quantiles: bool,
    max_control_points: int,
    max_span_count: int,
) -> jax.Array:
    if stats is None:
        return actions
    denormalized = actions
    control_points = denormalize_actions(
        actions[:, :max_control_points, :],
        stats=stats.control_points,
        use_quantiles=use_quantiles,
    )
    span_widths = denormalize_actions(
        actions[:, max_control_points, :max_span_count],
        stats=stats.span_widths,
        use_quantiles=use_quantiles,
    )
    denormalized = denormalized.at[:, :max_control_points, :].set(control_points)
    denormalized = denormalized.at[:, max_control_points, :max_span_count].set(span_widths)
    return denormalized


def _smooth_l1(diff: jax.Array, beta: float) -> jax.Array:
    abs_diff = jnp.abs(diff)
    if beta <= 0.0:
        return abs_diff
    quadratic = 0.5 * jnp.square(diff) / beta
    linear = abs_diff - 0.5 * beta
    return jnp.where(abs_diff < beta, quadratic, linear)


def _positive_normalized_widths(raw_widths: jax.Array, mask: jax.Array, *, min_width: float) -> jax.Array:
    positive = jax.nn.softplus(raw_widths) + min_width
    positive = positive * mask
    denom = jnp.clip(jnp.sum(positive, axis=-1, keepdims=True), a_min=min_width)
    return positive / denom


def _build_clamped_knots(
    widths: jax.Array,
    span_mask: jax.Array,
    *,
    degree: int,
    max_span_count: int,
) -> tuple[jax.Array, jax.Array]:
    boundaries = jnp.concatenate(
        [
            jnp.zeros((widths.shape[0], 1), dtype=widths.dtype),
            jnp.cumsum(widths, axis=-1),
        ],
        axis=-1,
    )
    full_len = max_span_count + 2 * degree + 1
    knots = jnp.ones((widths.shape[0], full_len), dtype=widths.dtype)
    knots = knots.at[:, : degree + 1].set(0.0)
    knots = knots.at[:, -degree - 1 :].set(1.0)
    internal_count = max_span_count - 1
    if internal_count > 0:
        knots = knots.at[:, degree + 1 : degree + 1 + internal_count].set(boundaries[:, 1:-1])
    num_spans = jnp.sum(span_mask, axis=-1).astype(jnp.int32)
    num_ctrl = num_spans + degree
    return knots, num_ctrl


def _basis_matrix(
    u: jax.Array,
    knots: jax.Array,
    *,
    degree: int,
    max_control_points: int,
    num_ctrl: jax.Array,
) -> jax.Array:
    eps = 1e-6
    left = knots[:, :-1]
    right = knots[:, 1:]
    u_expanded = u[None, :, None]
    base = ((u_expanded >= left[:, None, :]) & (u_expanded < right[:, None, :])).astype(jnp.float32)
    basis = base[:, :, :max_control_points]

    last_index = jnp.clip(num_ctrl - 1, a_min=0, a_max=max_control_points - 1)
    is_one = jnp.isclose(u, 1.0, atol=eps)
    one_hot = jax.nn.one_hot(last_index, max_control_points, dtype=jnp.float32)
    basis = jnp.where(is_one[None, :, None], one_hot[:, None, :], basis)

    for current_degree in range(1, degree + 1):
        next_basis = []
        for index in range(max_control_points):
            denom_left = knots[:, index + current_degree] - knots[:, index]
            left_term = jnp.where(
                denom_left[:, None] > eps,
                ((u[None, :] - knots[:, index][:, None]) / denom_left[:, None]) * basis[:, :, index],
                0.0,
            )
            if index + 1 < max_control_points:
                denom_right = knots[:, index + current_degree + 1] - knots[:, index + 1]
                right_term = jnp.where(
                    denom_right[:, None] > eps,
                    ((knots[:, index + current_degree + 1][:, None] - u[None, :]) / denom_right[:, None])
                    * basis[:, :, index + 1],
                    0.0,
                )
            else:
                right_term = 0.0
            next_basis.append(left_term + right_term)
        basis = jnp.stack(next_basis, axis=-1)
    return basis


def evaluate_batch_splines(
    control_points: jax.Array,
    widths: jax.Array,
    span_mask: jax.Array,
    *,
    degree: int,
    max_control_points: int,
    max_span_count: int,
    u: jax.Array,
) -> jax.Array:
    knots, num_ctrl = _build_clamped_knots(widths, span_mask, degree=degree, max_span_count=max_span_count)
    basis = _basis_matrix(
        u,
        knots,
        degree=degree,
        max_control_points=max_control_points,
        num_ctrl=num_ctrl,
    )
    return jnp.einsum("bsc,bcd->bsd", basis, control_points, precision=jax.lax.Precision.HIGHEST)


def compute_auxiliary_losses(
    pred_actions_norm: jax.Array,
    target_actions_norm: jax.Array,
    action_mask: jax.Array,
    *,
    stats: PackedActionNormStats | None,
    use_quantiles: bool,
    degree: int,
    max_control_points: int,
    max_span_count: int,
    sample_count: int,
    smooth_l1_beta: float,
    width_min: float,
    curve_weight: float,
    start_weight: float,
    end_weight: float,
    width_weight: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if all(weight <= 0.0 for weight in (curve_weight, start_weight, end_weight, width_weight)):
        zeros = jnp.zeros((pred_actions_norm.shape[0],), dtype=pred_actions_norm.dtype)
        return zeros, {
            "curve": zeros,
            "start": zeros,
            "end": zeros,
            "width": zeros,
        }

    pred_actions = denormalize_packed_actions(
        pred_actions_norm,
        stats=stats,
        use_quantiles=use_quantiles,
        max_control_points=max_control_points,
        max_span_count=max_span_count,
    )
    target_actions = denormalize_packed_actions(
        target_actions_norm,
        stats=stats,
        use_quantiles=use_quantiles,
        max_control_points=max_control_points,
        max_span_count=max_span_count,
    )

    control_mask = jnp.asarray(action_mask[:, :max_control_points, :], dtype=jnp.bool_)
    span_mask = jnp.asarray(action_mask[:, max_control_points, :max_span_count], dtype=jnp.bool_)

    pred_control = pred_actions[:, :max_control_points, :]
    target_control = target_actions[:, :max_control_points, :]
    pred_control = jnp.where(control_mask, pred_control, 0.0)
    target_control = jnp.where(control_mask, target_control, 0.0)

    pred_widths = _positive_normalized_widths(
        pred_actions[:, max_control_points, :max_span_count],
        jnp.asarray(span_mask, dtype=pred_actions.dtype),
        min_width=width_min,
    )
    target_widths = jnp.where(span_mask, target_actions[:, max_control_points, :max_span_count], 0.0)
    target_widths = target_widths / jnp.clip(jnp.sum(target_widths, axis=-1, keepdims=True), a_min=width_min)

    u = jnp.linspace(0.0, 1.0, sample_count, dtype=pred_actions.dtype)
    pred_curve = evaluate_batch_splines(
        pred_control,
        pred_widths,
        span_mask,
        degree=degree,
        max_control_points=max_control_points,
        max_span_count=max_span_count,
        u=u,
    )
    target_curve = evaluate_batch_splines(
        target_control,
        target_widths,
        span_mask,
        degree=degree,
        max_control_points=max_control_points,
        max_span_count=max_span_count,
        u=u,
    )

    curve_loss = jnp.mean(_smooth_l1(pred_curve - target_curve, smooth_l1_beta), axis=(1, 2))
    start_loss = jnp.mean(_smooth_l1(pred_curve[:, 0, :] - target_curve[:, 0, :], smooth_l1_beta), axis=-1)
    end_loss = jnp.mean(_smooth_l1(pred_curve[:, -1, :] - target_curve[:, -1, :], smooth_l1_beta), axis=-1)
    width_loss = jnp.sum(
        _smooth_l1(pred_widths - target_widths, smooth_l1_beta) * jnp.asarray(span_mask, dtype=pred_widths.dtype),
        axis=-1,
    ) / jnp.clip(jnp.sum(jnp.asarray(span_mask, dtype=pred_widths.dtype), axis=-1), a_min=1.0)

    total = (
        curve_weight * curve_loss
        + start_weight * start_loss
        + end_weight * end_loss
        + width_weight * width_loss
    )
    return total, {
        "curve": curve_loss,
        "start": start_loss,
        "end": end_loss,
        "width": width_loss,
    }
