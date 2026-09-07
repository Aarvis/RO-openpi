import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import origami_spline_losses as losses


def _spline_mask(batch_size: int = 2) -> jax.Array:
    mask = jnp.zeros((batch_size, 14, 65), dtype=jnp.float32)
    mask = mask.at[:, :13, :].set(1.0)
    mask = mask.at[:, 13, :10].set(1.0)
    return mask


def _spline_state(batch_size: int = 2) -> jax.Array:
    cps = jnp.linspace(-0.2, 0.2, batch_size * 13 * 65, dtype=jnp.float32).reshape(batch_size, 13, 65)
    logits = jnp.linspace(-0.6, 0.6, batch_size * 10, dtype=jnp.float32).reshape(batch_size, 10)
    logits = logits - jnp.mean(logits, axis=-1, keepdims=True)
    state = jnp.zeros((batch_size, 14, 65), dtype=jnp.float32)
    state = state.at[:, :13, :].set(cps)
    state = state.at[:, 13, :10].set(logits)
    return state


def test_curve_flow_matching_loss_is_finite() -> None:
    state = _spline_state()
    mask = _spline_mask()
    pred_velocity = jnp.ones_like(state) * 0.03
    target_velocity = jnp.ones_like(state) * -0.02

    loss, terms = losses.compute_curve_flow_matching_loss(
        state,
        pred_velocity,
        target_velocity,
        mask,
        stats=None,
        use_quantiles=False,
        degree=3,
        max_control_points=13,
        max_span_count=10,
        sample_intervals=10,
        include_endpoints=True,
        width_min=1e-4,
        denominator_eps=1e-6,
        softmax_clip=30.0,
        loss_clip=None,
        span_representation="logits",
    )

    assert loss.shape == (2,)
    assert np.all(np.isfinite(np.asarray(loss)))
    assert np.all(np.asarray(terms["curve_fm_finite"]) == 1.0)


def test_curve_flow_matching_one_jvp_matches_two_jvp_subtraction() -> None:
    state = _spline_state()
    mask = _spline_mask()
    pred_velocity = jnp.sin(jnp.arange(state.size, dtype=jnp.float32)).reshape(state.shape) * 0.03
    target_velocity = jnp.cos(jnp.arange(state.size, dtype=jnp.float32)).reshape(state.shape) * 0.02

    one_jvp_loss, _ = losses.compute_curve_flow_matching_loss(
        state,
        pred_velocity,
        target_velocity,
        mask,
        stats=None,
        use_quantiles=False,
        degree=3,
        max_control_points=13,
        max_span_count=10,
        sample_intervals=10,
        include_endpoints=True,
        width_min=1e-4,
        denominator_eps=1e-6,
        softmax_clip=30.0,
        loss_clip=None,
        span_representation="logits",
        separate_velocity_metrics=False,
    )
    two_jvp_loss, _ = losses.compute_curve_flow_matching_loss(
        state,
        pred_velocity,
        target_velocity,
        mask,
        stats=None,
        use_quantiles=False,
        degree=3,
        max_control_points=13,
        max_span_count=10,
        sample_intervals=10,
        include_endpoints=True,
        width_min=1e-4,
        denominator_eps=1e-6,
        softmax_clip=30.0,
        loss_clip=None,
        span_representation="logits",
        separate_velocity_metrics=True,
    )

    np.testing.assert_allclose(np.asarray(one_jvp_loss), np.asarray(two_jvp_loss), rtol=2e-5, atol=2e-7)


def test_common_width_logit_shift_has_no_curve_velocity() -> None:
    state = _spline_state(batch_size=1)
    mask = _spline_mask(batch_size=1)
    pred_velocity = jnp.zeros_like(state).at[:, 13, :10].set(0.5)
    target_velocity = jnp.zeros_like(state)

    loss, terms = losses.compute_curve_flow_matching_loss(
        state,
        pred_velocity,
        target_velocity,
        mask,
        stats=None,
        use_quantiles=False,
        degree=3,
        max_control_points=13,
        max_span_count=10,
        sample_intervals=10,
        include_endpoints=True,
        width_min=1e-4,
        denominator_eps=1e-6,
        softmax_clip=30.0,
        loss_clip=None,
        span_representation="logits",
    )

    np.testing.assert_allclose(np.asarray(loss), np.zeros((1,), dtype=np.float32), atol=2e-10)
    np.testing.assert_allclose(np.asarray(terms["curve_fm_velocity_rms"]), np.zeros((1,), dtype=np.float32), atol=2e-5)
