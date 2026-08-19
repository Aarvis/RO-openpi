from __future__ import annotations

import dataclasses
import functools
import logging
import platform

import etils.epath as epath
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.train_lib as _base_train
import openpi.training.utils as training_utils
import openpi.training.weighted_training.data_loader as _weighted_data_loader


def main(config: _config.TrainConfig):
    _base_train.init_logging()
    logging.info("Running weighted training on: %s", platform.node())

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    if config.run_val and config.resolved_val_batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Validation batch size {config.resolved_val_batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    best_info = _checkpoints.BestCheckpointInfo()
    latest_val_info = _checkpoints.BestCheckpointInfo()
    latest_val_checkpoint_manager = None
    if config.checkpoint_strategy == "best_val":
        checkpoint_manager, latest_val_checkpoint_manager, resuming, best_info, latest_val_info = (
            _checkpoints.initialize_val_checkpoint_dirs(
            config.checkpoint_dir,
            overwrite=config.overwrite,
            resume=config.resume,
            )
        )
    else:
        checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=config.keep_period,
            max_to_keep=config.max_to_keep,
            overwrite=config.overwrite,
            resume=config.resume,
        )
    _base_train.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _weighted_data_loader.create_weighted_data_loader(
        config,
        sharding=data_sharding,
    )
    val_loader = _data_loader.create_validation_data_loader(
        config,
        sharding=data_sharding,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info("Initialized weighted data loader:\n%s", training_utils.array_tree_to_info(batch))
    if val_loader is not None:
        logging.info(
            "Initialized validation data loader from repo_id=%s with batch_size=%d",
            config.val_repo_id,
            config.resolved_val_batch_size,
        )

    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = _base_train.init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info("Initialized train state:\n%s", training_utils.array_tree_to_info(train_state.params))

    if resuming:
        restore_checkpoint_manager = checkpoint_manager
        if config.checkpoint_strategy == "best_val" and latest_val_info.step is not None and latest_val_checkpoint_manager is not None:
            restore_checkpoint_manager = latest_val_checkpoint_manager
        train_state = _checkpoints.restore_state(restore_checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(_base_train.train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    pval_step = None
    if val_loader is not None:
        pval_step = jax.jit(
            functools.partial(_base_train.eval_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )

    start_step = int(train_state.step)
    best_val_loss = best_info.val_loss if best_info.val_loss is not None else float("inf")
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        completed_step = int(train_state.step)
        infos.append(info)
        if completed_step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {completed_step}: {info_str}")
            wandb.log(reduced_info, step=completed_step)
            infos = []
        batch = next(data_iter)

        if val_loader is not None and pval_step is not None and completed_step % config.val_frequency == 0:
            val_infos = []
            val_batches = 0
            val_pbar = tqdm.tqdm(
                val_loader,
                total=len(val_loader),
                desc=f"Validation @ step {completed_step}",
                dynamic_ncols=True,
                leave=False,
            )
            for val_batch in val_pbar:
                with sharding.set_mesh(mesh):
                    val_info = pval_step(train_rng, train_state, val_batch)
                val_infos.append(val_info)
                val_batches += 1

            stacked_val_infos = common_utils.stack_forest(val_infos)
            reduced_val_info = jax.device_get(jax.tree.map(jnp.mean, stacked_val_infos))
            val_info_str = ", ".join(f"val_{k}={v:.4f}" for k, v in reduced_val_info.items())
            pbar.write(f"Step {completed_step}: {val_info_str}, val_batches={val_batches}")
            wandb.log(
                {**{f"val/{k}": v for k, v in reduced_val_info.items()}, "val/num_batches": val_batches},
                step=completed_step,
            )
            val_loss = float(reduced_val_info["loss"])
            if config.checkpoint_strategy == "best_val" and latest_val_checkpoint_manager is not None:
                _checkpoints.save_latest_val_state(
                    latest_val_checkpoint_manager,
                    config.latest_val_checkpoint_dir,
                    config,
                    train_state,
                    data_loader,
                    completed_step,
                    val_loss,
                )
                pbar.write(f"Step {completed_step}: updated latest validation checkpoint with val_loss={val_loss:.4f}")
            if config.checkpoint_strategy == "best_val" and val_loss < best_val_loss:
                _checkpoints.save_best_state(
                    checkpoint_manager,
                    config.best_checkpoint_dir,
                    config,
                    train_state,
                    data_loader,
                    completed_step,
                    val_loss,
                )
                best_val_loss = val_loss
                pbar.write(f"Step {completed_step}: updated best checkpoint with val_loss={val_loss:.4f}")
                wandb.log({"val/best_loss": val_loss, "val/best_step": completed_step}, step=completed_step)

        if completed_step > start_step and _base_train.should_save_checkpoint(config, completed_step):
            _checkpoints.save_state(checkpoint_manager, config, train_state, data_loader, completed_step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if latest_val_checkpoint_manager is not None:
        latest_val_checkpoint_manager.wait_until_finished()


def cli() -> None:
    main(_config.cli())
