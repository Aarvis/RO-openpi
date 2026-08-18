from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class _MLPProjector(nn.Module):
    input_dim: int
    hidden_dims: tuple[int, ...]
    output_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        y = x
        for index, hidden_dim in enumerate(self.hidden_dims):
            y = nn.Dense(hidden_dim, param_dtype=jnp.float32, name=f"dense_{index}")(y)
            y = nn.gelu(y)
        y = nn.Dense(self.output_dim, param_dtype=jnp.float32, name="dense_out")(y)
        return y


class OrigamiPlannerPrefixAdapter(nn.Module):
    belief_dim: int
    history_dim: int
    output_dim: int
    belief_hidden_dims: tuple[int, ...] = (256, 512)
    progress_hidden_dims: tuple[int, ...] = (128, 512)
    uncertainty_hidden_dims: tuple[int, ...] = (128, 512)
    history_hidden_dims: tuple[int, ...] = (1024,)
    use_type_embeddings: bool = True

    @nn.compact
    def __call__(
        self,
        belief: jax.Array,
        progress_transition: jax.Array,
        uncertainty: jax.Array,
        history_latent: jax.Array,
        *,
        train: bool = False,
    ) -> jax.Array:
        del train
        if belief.shape[-1] != self.belief_dim:
            raise ValueError(f"Expected belief dim {self.belief_dim}, got {belief.shape[-1]}")
        if progress_transition.shape[-1] != 2:
            raise ValueError(f"Expected progress_transition dim 2, got {progress_transition.shape[-1]}")
        if uncertainty.shape[-1] != 3:
            raise ValueError(f"Expected uncertainty dim 3, got {uncertainty.shape[-1]}")
        if history_latent.shape[-1] != self.history_dim:
            raise ValueError(f"Expected history_latent dim {self.history_dim}, got {history_latent.shape[-1]}")

        belief_token = _MLPProjector(
            input_dim=self.belief_dim,
            hidden_dims=self.belief_hidden_dims,
            output_dim=self.output_dim,
            name="belief_projector",
        )(belief)
        progress_token = _MLPProjector(
            input_dim=2,
            hidden_dims=self.progress_hidden_dims,
            output_dim=self.output_dim,
            name="progress_projector",
        )(progress_transition)
        uncertainty_token = _MLPProjector(
            input_dim=3,
            hidden_dims=self.uncertainty_hidden_dims,
            output_dim=self.output_dim,
            name="uncertainty_projector",
        )(uncertainty)
        history_token = _MLPProjector(
            input_dim=self.history_dim,
            hidden_dims=self.history_hidden_dims,
            output_dim=self.output_dim,
            name="history_projector",
        )(history_latent)

        tokens = jnp.stack([belief_token, progress_token, uncertainty_token, history_token], axis=1)
        if self.use_type_embeddings:
            type_embeddings = self.param(
                "planner_token_type_embeddings",
                nn.initializers.normal(stddev=0.02),
                (4, self.output_dim),
            )
            tokens = tokens + type_embeddings[None, :, :]
        return nn.LayerNorm(name="planner_token_norm")(tokens)
