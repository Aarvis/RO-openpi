from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class ActionNormStats:
    mean: Any
    std: Any
    q01: Any | None = None
    q99: Any | None = None


@dataclasses.dataclass(frozen=True)
class PackedActionNormStats:
    control_points: ActionNormStats
    span_widths: ActionNormStats


def _as_jax_array(value: Any, *, dtype: jnp.dtype | None = None) -> jax.Array:
    if dtype is None:
        return jnp.asarray(value)
    return jnp.asarray(value, dtype=dtype)


def denormalize_actions(
    actions: jax.Array,
    *,
    stats: ActionNormStats | None,
    use_quantiles: bool,
) -> jax.Array:
    if stats is None:
        return actions
    mean = _as_jax_array(stats.mean, dtype=actions.dtype)
    std = _as_jax_array(stats.std, dtype=actions.dtype)
    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile action denormalization requested, but q01/q99 are missing.")
        q01 = _as_jax_array(stats.q01, dtype=actions.dtype)
        q99 = _as_jax_array(stats.q99, dtype=actions.dtype)
        return (actions + 1.0) * 0.5 * (q99 + 1e-6 - q01) + q01
    return actions * (std + 1e-6) + mean


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


def _masked_softmax_widths(
    logits: jax.Array,
    mask: jax.Array,
    *,
    min_width: float,
    softmax_clip: float | None = None,
    enforce_min_width: bool = False,
) -> jax.Array:
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    logits = jnp.asarray(logits, dtype=jnp.float32)
    if softmax_clip is not None:
        logits = jnp.clip(logits, -softmax_clip, softmax_clip)
    masked_logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
    max_logits = jnp.max(masked_logits, axis=-1, keepdims=True)
    max_logits = jnp.where(jnp.isfinite(max_logits), max_logits, 0.0)
    exp_logits = jnp.exp(masked_logits - max_logits) * jnp.asarray(mask, dtype=jnp.float32)
    denom = jnp.clip(jnp.sum(exp_logits, axis=-1, keepdims=True), a_min=min_width)
    widths = exp_logits / denom
    if not enforce_min_width or min_width <= 0.0:
        return widths
    valid = jnp.asarray(mask, dtype=jnp.float32)
    valid_count = jnp.sum(valid, axis=-1, keepdims=True)
    safe_min_width = jnp.minimum(min_width, 0.999 / jnp.clip(valid_count, a_min=1.0))
    scale = jnp.clip(1.0 - valid_count * safe_min_width, a_min=0.0)
    return jnp.where(mask, widths * scale + safe_min_width, 0.0)


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
    denominator_eps: float = 1e-6,
) -> jax.Array:
    eps = denominator_eps
    left = knots[:, :-1]
    right = knots[:, 1:]
    u_expanded = u[None, :, None]
    base = ((u_expanded >= left[:, None, :]) & (u_expanded < right[:, None, :])).astype(u.dtype)
    basis = base[:, :, :max_control_points]

    last_index = jnp.clip(num_ctrl - 1, a_min=0, a_max=max_control_points - 1)
    is_one = jnp.isclose(u, 1.0, atol=eps)
    one_hot = jax.nn.one_hot(last_index, max_control_points, dtype=u.dtype)
    basis = jnp.where(is_one[None, :, None], one_hot[:, None, :], basis)

    for current_degree in range(1, degree + 1):
        next_basis = []
        for index in range(max_control_points):
            denom_left = knots[:, index + current_degree] - knots[:, index]
            safe_denom_left = jnp.where(denom_left > eps, denom_left, 1.0)
            left_coeff = (u[None, :] - knots[:, index][:, None]) / safe_denom_left[:, None]
            left_coeff = jnp.where(denom_left[:, None] > eps, left_coeff, 0.0)
            left_term = left_coeff * basis[:, :, index]
            if index + 1 < max_control_points:
                denom_right = knots[:, index + current_degree + 1] - knots[:, index + 1]
                safe_denom_right = jnp.where(denom_right > eps, denom_right, 1.0)
                right_coeff = (knots[:, index + current_degree + 1][:, None] - u[None, :]) / safe_denom_right[:, None]
                right_coeff = jnp.where(denom_right[:, None] > eps, right_coeff, 0.0)
                right_term = right_coeff * basis[:, :, index + 1]
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
    denominator_eps: float = 1e-6,
) -> jax.Array:
    knots, num_ctrl = _build_clamped_knots(widths, span_mask, degree=degree, max_span_count=max_span_count)
    basis = _basis_matrix(
        u,
        knots,
        degree=degree,
        max_control_points=max_control_points,
        num_ctrl=num_ctrl,
        denominator_eps=denominator_eps,
    )
    return jnp.einsum("bsc,bcd->bsd", basis, control_points, precision=jax.lax.Precision.HIGHEST)


def _sample_phases_from_intervals(intervals: int, *, include_endpoints: bool) -> jax.Array:
    if include_endpoints:
        return jnp.linspace(0.0, 1.0, intervals + 1, dtype=jnp.float32)
    if intervals <= 1:
        return jnp.asarray([0.5], dtype=jnp.float32)
    return jnp.arange(1, intervals, dtype=jnp.float32) / jnp.asarray(intervals, dtype=jnp.float32)


def decode_packed_spline_curve(
    actions_norm: jax.Array,
    action_mask: jax.Array,
    *,
    stats: PackedActionNormStats | None,
    use_quantiles: bool,
    degree: int,
    max_control_points: int,
    max_span_count: int,
    u: jax.Array,
    width_min: float,
    span_representation: str = "physical_widths",
    denominator_eps: float = 1e-6,
    softmax_clip: float | None = None,
    enforce_min_width: bool = False,
) -> tuple[jax.Array, jax.Array]:
    if span_representation not in ("physical_widths", "logits"):
        raise ValueError(f"Unsupported spline span representation: {span_representation!r}")

    actions_norm = jnp.asarray(actions_norm, dtype=jnp.float32)
    actions = denormalize_packed_actions(
        actions_norm,
        stats=stats,
        use_quantiles=use_quantiles,
        max_control_points=max_control_points,
        max_span_count=max_span_count,
    )
    control_mask = jnp.asarray(action_mask[:, :max_control_points, :], dtype=jnp.bool_)
    span_mask = jnp.asarray(action_mask[:, max_control_points, :max_span_count], dtype=jnp.bool_)
    control = jnp.where(control_mask, actions[:, :max_control_points, :], 0.0)

    if span_representation == "logits":
        widths = _masked_softmax_widths(
            actions[:, max_control_points, :max_span_count],
            span_mask,
            min_width=width_min,
            softmax_clip=softmax_clip,
            enforce_min_width=enforce_min_width,
        )
    else:
        widths = _positive_normalized_widths(
            actions[:, max_control_points, :max_span_count],
            jnp.asarray(span_mask, dtype=actions.dtype),
            min_width=width_min,
        )

    curve = evaluate_batch_splines(
        control,
        widths,
        span_mask,
        degree=degree,
        max_control_points=max_control_points,
        max_span_count=max_span_count,
        u=u,
        denominator_eps=denominator_eps,
    )
    return curve, widths


def compute_curve_flow_matching_loss(
    flow_state_norm: jax.Array,
    pred_velocity_norm: jax.Array,
    target_velocity_norm: jax.Array,
    action_mask: jax.Array,
    *,
    stats: PackedActionNormStats | None,
    use_quantiles: bool,
    degree: int,
    max_control_points: int,
    max_span_count: int,
    sample_intervals: int,
    include_endpoints: bool,
    width_min: float,
    denominator_eps: float,
    softmax_clip: float | None,
    loss_clip: float | None,
    span_representation: str = "physical_widths",
    separate_velocity_metrics: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    action_mask_f32 = jnp.asarray(action_mask, dtype=jnp.float32)
    flow_state_norm = jnp.asarray(flow_state_norm, dtype=jnp.float32) * action_mask_f32
    pred_velocity_norm = jnp.asarray(pred_velocity_norm, dtype=jnp.float32) * action_mask_f32
    target_velocity_norm = jnp.asarray(target_velocity_norm, dtype=jnp.float32) * action_mask_f32
    u = _sample_phases_from_intervals(sample_intervals, include_endpoints=include_endpoints)

    def decode_curve(values: jax.Array) -> jax.Array:
        curve, _ = decode_packed_spline_curve(
            values,
            action_mask,
            stats=stats,
            use_quantiles=use_quantiles,
            degree=degree,
            max_control_points=max_control_points,
            max_span_count=max_span_count,
            u=u,
            width_min=width_min,
            span_representation=span_representation,
            denominator_eps=denominator_eps,
            softmax_clip=softmax_clip,
            enforce_min_width=True,
        )
        return curve

    if separate_velocity_metrics:
        _, curve_velocity_pred = jax.jvp(decode_curve, (flow_state_norm,), (pred_velocity_norm,))
        _, curve_velocity_target = jax.jvp(decode_curve, (flow_state_norm,), (target_velocity_norm,))
        curve_velocity_error = curve_velocity_pred - curve_velocity_target
        curve_jvp_norm_pred = jnp.sqrt(jnp.sum(jnp.square(curve_velocity_pred), axis=(1, 2)))
        curve_jvp_norm_target = jnp.sqrt(jnp.sum(jnp.square(curve_velocity_target), axis=(1, 2)))
        max_abs_curve_velocity_pred = jnp.max(jnp.abs(curve_velocity_pred), axis=(1, 2))
        max_abs_curve_velocity_target = jnp.max(jnp.abs(curve_velocity_target), axis=(1, 2))
    else:
        velocity_error_norm = pred_velocity_norm - target_velocity_norm
        _, curve_velocity_error = jax.jvp(decode_curve, (flow_state_norm,), (velocity_error_norm,))
        zeros = jnp.zeros((flow_state_norm.shape[0],), dtype=jnp.float32)
        curve_jvp_norm_pred = zeros
        curve_jvp_norm_target = zeros
        max_abs_curve_velocity_pred = zeros
        max_abs_curve_velocity_target = zeros

    curve_velocity_error = jnp.asarray(curve_velocity_error, dtype=jnp.float32)
    sq_error = jnp.square(curve_velocity_error)
    loss = jnp.mean(sq_error, axis=(1, 2))
    if loss_clip is not None:
        loss = jnp.minimum(loss, loss_clip)

    _, widths = decode_packed_spline_curve(
        flow_state_norm,
        action_mask,
        stats=stats,
        use_quantiles=use_quantiles,
        degree=degree,
        max_control_points=max_control_points,
        max_span_count=max_span_count,
        u=u,
        width_min=width_min,
        span_representation=span_representation,
        denominator_eps=denominator_eps,
        softmax_clip=softmax_clip,
        enforce_min_width=True,
    )
    span_mask = jnp.asarray(action_mask[:, max_control_points, :max_span_count], dtype=jnp.bool_)
    valid_widths = jnp.where(span_mask, widths, jnp.inf)
    min_predicted_span_width = jnp.min(valid_widths, axis=-1)
    finite_mask = jnp.isfinite(loss) & jnp.all(jnp.isfinite(curve_velocity_error), axis=(1, 2))
    curve_jvp_error_norm = jnp.sqrt(jnp.sum(sq_error, axis=(1, 2)))

    return loss, {
        "curve_fm_velocity_mae": jnp.mean(jnp.abs(curve_velocity_error), axis=(1, 2)),
        "curve_fm_velocity_rms": jnp.sqrt(jnp.mean(sq_error, axis=(1, 2))),
        "curve_fm_jvp_error_norm": curve_jvp_error_norm,
        "curve_fm_jvp_norm_pred": curve_jvp_norm_pred,
        "curve_fm_jvp_norm_target": curve_jvp_norm_target,
        "curve_fm_max_abs_velocity_pred": max_abs_curve_velocity_pred,
        "curve_fm_max_abs_velocity_target": max_abs_curve_velocity_target,
        "curve_fm_min_predicted_span_width": min_predicted_span_width,
        "curve_fm_finite": finite_mask.astype(jnp.float32),
    }


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
    span_representation: str = "physical_widths",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if span_representation not in ("physical_widths", "logits"):
        raise ValueError(f"Unsupported spline span representation: {span_representation!r}")
    if all(weight <= 0.0 for weight in (curve_weight, start_weight, end_weight, width_weight)):
        zeros = jnp.zeros((pred_actions_norm.shape[0],), dtype=jnp.float32)
        return zeros, {
            "curve": zeros,
            "start": zeros,
            "end": zeros,
            "width": zeros,
            "curve_mae_rad": zeros,
            "start_mae_rad": zeros,
            "end_mae_rad": zeros,
            "width_mae": zeros,
        }

    # Keep the spline auxiliary path in float32 even when the policy itself runs in
    # bfloat16. This avoids precision loss in de-normalization, knot widths, basis
    # recursion, and curve sampling.
    pred_actions_norm = jnp.asarray(pred_actions_norm, dtype=jnp.float32)
    target_actions_norm = jnp.asarray(target_actions_norm, dtype=jnp.float32)

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

    if span_representation == "logits":
        pred_widths = _masked_softmax_widths(
            pred_actions[:, max_control_points, :max_span_count],
            span_mask,
            min_width=width_min,
        )
        target_widths = _masked_softmax_widths(
            target_actions[:, max_control_points, :max_span_count],
            span_mask,
            min_width=width_min,
        )
    else:
        pred_widths = _positive_normalized_widths(
            pred_actions[:, max_control_points, :max_span_count],
            jnp.asarray(span_mask, dtype=pred_actions.dtype),
            min_width=width_min,
        )
        target_widths = jnp.where(span_mask, target_actions[:, max_control_points, :max_span_count], 0.0)
        target_widths = target_widths / jnp.clip(jnp.sum(target_widths, axis=-1, keepdims=True), a_min=width_min)

    u = jnp.linspace(0.0, 1.0, sample_count, dtype=jnp.float32)
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

    curve_diff = pred_curve - target_curve
    start_diff = pred_curve[:, 0, :] - target_curve[:, 0, :]
    end_diff = pred_curve[:, -1, :] - target_curve[:, -1, :]
    width_diff = pred_widths - target_widths

    curve_loss = jnp.mean(_smooth_l1(curve_diff, smooth_l1_beta), axis=(1, 2))
    start_loss = jnp.mean(_smooth_l1(start_diff, smooth_l1_beta), axis=-1)
    end_loss = jnp.mean(_smooth_l1(end_diff, smooth_l1_beta), axis=-1)
    width_loss = jnp.sum(
        _smooth_l1(width_diff, smooth_l1_beta) * jnp.asarray(span_mask, dtype=pred_widths.dtype),
        axis=-1,
    ) / jnp.clip(jnp.sum(jnp.asarray(span_mask, dtype=pred_widths.dtype), axis=-1), a_min=1.0)
    curve_mae_rad = jnp.mean(jnp.abs(curve_diff), axis=(1, 2))
    start_mae_rad = jnp.mean(jnp.abs(start_diff), axis=-1)
    end_mae_rad = jnp.mean(jnp.abs(end_diff), axis=-1)
    width_mae = jnp.sum(
        jnp.abs(width_diff) * jnp.asarray(span_mask, dtype=pred_widths.dtype),
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
        "curve_mae_rad": curve_mae_rad,
        "start_mae_rad": start_mae_rad,
        "end_mae_rad": end_mae_rad,
        "width_mae": width_mae,
    }
