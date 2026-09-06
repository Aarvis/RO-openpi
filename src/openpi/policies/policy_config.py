import logging
import os
import pathlib
import dataclasses
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.models.origami_tactile_adapter as _origami_tactile_adapter
import openpi.models.pi0_config as pi0_config
from openpi.policies import future_latent_runtime as _future_latent_runtime
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def _maybe_use_checkpoint_origami_stats(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    asset_id: str | None,
) -> _config.TrainConfig:
    if asset_id is None:
        return train_config
    if not isinstance(train_config.model, pi0_config.Pi0Config):
        return train_config
    if not train_config.model.origami_vla.enabled:
        return train_config

    checkpoint_asset_dir = checkpoint_dir / "assets" / asset_id
    checkpoint_norm_stats = checkpoint_asset_dir / "norm_stats.json"
    checkpoint_tactile_stats = checkpoint_asset_dir / _origami_tactile_adapter.TACTILE_NORM_STATS_FILENAME
    origami_config = train_config.model.origami_vla

    if not checkpoint_norm_stats.exists():
        logging.info(
            "Checkpoint Origami norm stats not found at %s; falling back to configured asset directory.",
            checkpoint_norm_stats,
        )
        return train_config

    if origami_config.tactile_enabled and not checkpoint_tactile_stats.exists():
        logging.info(
            "Checkpoint Origami tactile norm stats not found at %s; falling back to configured asset directory.",
            checkpoint_tactile_stats,
        )
        return train_config

    logging.info("Using Origami normalization stats from checkpoint assets: %s", checkpoint_asset_dir)
    return dataclasses.replace(
        train_config,
        model=dataclasses.replace(
            train_config.model,
            origami_vla=dataclasses.replace(
                origami_config,
                action_norm_stats_dir=str(checkpoint_asset_dir),
            ),
        ),
    )


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    ppo_actor_head_path: str | None = None,
    ppo_value_head_path: str | None = None,
    ppo_device: str | None = None,
    ppo_deterministic: bool | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    if train_config.ppo_policy is not None:
        from openpi.policies import lehome_ppo_policy

        ppo_config = train_config.ppo_policy
        runtime_config = lehome_ppo_policy.LehomePPORuntimeConfig(
            action_horizon=ppo_config.action_horizon,
            action_dim=ppo_config.action_dim,
            latent_dim=ppo_config.latent_dim,
            state_dim=ppo_config.state_dim,
            token_dim=ppo_config.token_dim,
            num_layers=ppo_config.num_layers,
            num_heads=ppo_config.num_heads,
            mlp_ratio=ppo_config.mlp_ratio,
            dropout=ppo_config.dropout,
            correction_scale=ppo_config.correction_scale,
            delta_clip=ppo_config.delta_clip,
            log_std_init=ppo_config.log_std_init,
            min_log_std=ppo_config.min_log_std,
            max_log_std=ppo_config.max_log_std,
            deterministic=ppo_config.deterministic if ppo_deterministic is None else ppo_deterministic,
            value_coef=ppo_config.value_coef,
            entropy_coef=ppo_config.entropy_coef,
            delta_coef=ppo_config.delta_coef,
        )
        base_config = dataclasses.replace(train_config, ppo_policy=None)
        base_sample_kwargs = dict(sample_kwargs or {})
        base_sample_kwargs["return_policy_latent"] = True
        base_policy = create_trained_policy(
            base_config,
            checkpoint_dir,
            repack_transforms=repack_transforms,
            sample_kwargs=base_sample_kwargs,
            default_prompt=default_prompt,
            norm_stats=norm_stats,
            pytorch_device=pytorch_device,
        )
        return lehome_ppo_policy.LehomePPOPolicy(
            base_policy,
            config=runtime_config,
            actor_head_path=ppo_actor_head_path or ppo_config.actor_head_path,
            value_head_path=ppo_value_head_path or ppo_config.value_head_path,
            device=ppo_device,
            seed=train_config.seed,
            metadata=train_config.policy_metadata,
        )

    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    train_config = _maybe_use_checkpoint_origami_stats(train_config, pathlib.Path(checkpoint_dir), data_config.asset_id)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    future_runtime = None
    if (
        not is_pytorch
        and isinstance(train_config.model, pi0_config.Pi0Config)
        and train_config.model.future_latent.enabled
        and train_config.model.future_latent.resampler_checkpoint_path
        and train_config.model.future_latent.future_predictor_checkpoint_path
    ):
        future_runtime = _future_latent_runtime.FutureLatentRuntime(
            config=train_config.model.future_latent,
            device=pytorch_device,
        )

    normalize_transform = transforms.make_normalize_transform(
        norm_stats,
        use_quantiles=data_config.use_quantile_norm,
        origami_action_mode=(
            data_config.origami_vla.action_source if data_config.origami_vla is not None else None
        ),
        origami_spline_span_representation=(
            data_config.origami_vla.spline_span_representation
            if data_config.origami_vla is not None
            else "physical_widths"
        ),
        origami_max_control_points=(
            data_config.origami_vla.max_control_points if data_config.origami_vla is not None else None
        ),
        origami_max_span_count=(
            data_config.origami_vla.max_span_count if data_config.origami_vla is not None else None
        ),
    )
    unnormalize_transform = transforms.make_unnormalize_transform(
        norm_stats,
        use_quantiles=data_config.use_quantile_norm,
        origami_action_mode=(
            data_config.origami_vla.action_source if data_config.origami_vla is not None else None
        ),
        origami_spline_span_representation=(
            data_config.origami_vla.spline_span_representation
            if data_config.origami_vla is not None
            else "physical_widths"
        ),
        origami_max_control_points=(
            data_config.origami_vla.max_control_points if data_config.origami_vla is not None else None
        ),
        origami_max_span_count=(
            data_config.origami_vla.max_span_count if data_config.origami_vla is not None else None
        ),
    )

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            normalize_transform,
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            unnormalize_transform,
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
        future_latent_runtime=future_runtime,
    )
