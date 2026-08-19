from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import json
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.models import pi0_config as _pi0_config
import openpi.models.origami_tactile_adapter as _origami_tactile_adapter
from openpi.shared import array_typing as at
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


@dataclasses.dataclass(frozen=True)
class BestCheckpointInfo:
    step: int | None = None
    val_loss: float | None = None


def _create_checkpoint_manager(
    checkpoint_dir: epath.Path, *, keep_period: int | None, max_to_keep: int | None
) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    max_to_keep: int | None,
    overwrite: bool,
    resume: bool,
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = _create_checkpoint_manager(checkpoint_dir, keep_period=keep_period, max_to_keep=max_to_keep)

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def initialize_best_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    overwrite: bool,
    resume: bool,
) -> tuple[ocp.CheckpointManager, bool, BestCheckpointInfo]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    best_checkpoint_dir = checkpoint_dir / "best"
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = _create_checkpoint_manager(best_checkpoint_dir, keep_period=None, max_to_keep=1)
    if resuming and tuple(mngr.all_steps()) == ():
        logging.info("Best-checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    best_info = load_best_checkpoint_info(best_checkpoint_dir) if resuming else BestCheckpointInfo()
    return mngr, resuming, best_info


def initialize_val_checkpoint_dirs(
    checkpoint_dir: epath.Path | str,
    *,
    overwrite: bool,
    resume: bool,
) -> tuple[ocp.CheckpointManager, ocp.CheckpointManager, bool, BestCheckpointInfo, BestCheckpointInfo]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    best_checkpoint_dir = checkpoint_dir / "best"
    latest_val_checkpoint_dir = checkpoint_dir / "latest_val"
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_val_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_manager = _create_checkpoint_manager(best_checkpoint_dir, keep_period=None, max_to_keep=1)
    latest_val_manager = _create_checkpoint_manager(latest_val_checkpoint_dir, keep_period=None, max_to_keep=1)

    if resuming and tuple(best_manager.all_steps()) == () and tuple(latest_val_manager.all_steps()) == ():
        logging.info("Validation-checkpoint directories exist, but contain no checkpoints. Aborting resume.")
        resuming = False

    best_info = load_best_checkpoint_info(best_checkpoint_dir) if resuming else BestCheckpointInfo()
    latest_val_info = load_best_checkpoint_info(latest_val_checkpoint_dir) if resuming else BestCheckpointInfo()
    return best_manager, latest_val_manager, resuming, best_info, latest_val_info


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is None or data_config.asset_id is None:
            return

        asset_dir = directory / data_config.asset_id
        _normalize.save(asset_dir, norm_stats)

        model_config = config.model
        if not isinstance(model_config, _pi0_config.Pi0Config):
            return
        if not model_config.origami_vla.enabled or not model_config.origami_vla.tactile_enabled:
            return

        stats_dir = model_config.origami_vla.action_norm_stats_dir
        if not stats_dir:
            logging.warning("Origami tactile conditioning is enabled, but action_norm_stats_dir is unset during checkpoint save.")
            return

        resolved_dir = epath.Path(_download.maybe_download(stats_dir))
        source_path = resolved_dir / _origami_tactile_adapter.TACTILE_NORM_STATS_FILENAME
        if not source_path.exists():
            logging.warning("Origami tactile normalization stats were not found at %s while saving checkpoint assets.", source_path)
            return

        target_path = asset_dir / _origami_tactile_adapter.TACTILE_NORM_STATS_FILENAME
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(source_path.read_bytes())
        logging.info("Saved Origami tactile normalization stats to %s", target_path)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def save_best_state(
    checkpoint_manager: ocp.CheckpointManager,
    best_checkpoint_dir: epath.Path | str,
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    val_loss: float,
):
    checkpoint_manager.wait_until_finished()
    save_state(checkpoint_manager, config, state, data_loader, step)
    checkpoint_manager.wait_until_finished()
    _write_best_checkpoint_info(best_checkpoint_dir, BestCheckpointInfo(step=step, val_loss=val_loss))


def save_latest_val_state(
    checkpoint_manager: ocp.CheckpointManager,
    latest_val_checkpoint_dir: epath.Path | str,
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    val_loss: float,
):
    checkpoint_manager.wait_until_finished()
    save_state(checkpoint_manager, config, state, data_loader, step)
    checkpoint_manager.wait_until_finished()
    _write_best_checkpoint_info(latest_val_checkpoint_dir, BestCheckpointInfo(step=step, val_loss=val_loss))


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


def load_best_checkpoint_info(best_checkpoint_dir: epath.Path | str) -> BestCheckpointInfo:
    checkpoint_dir = epath.Path(best_checkpoint_dir)
    metadata_path = checkpoint_dir / "checkpoint_info.json"
    if not metadata_path.exists():
        metadata_path = checkpoint_dir / "best_checkpoint.json"
    if not metadata_path.exists():
        return BestCheckpointInfo()
    payload = json.loads(metadata_path.read_text())
    return BestCheckpointInfo(step=payload.get("step"), val_loss=payload.get("val_loss"))


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _write_best_checkpoint_info(best_checkpoint_dir: epath.Path | str, info: BestCheckpointInfo) -> None:
    best_checkpoint_dir = epath.Path(best_checkpoint_dir)
    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = best_checkpoint_dir / "checkpoint_info.json"
    metadata_path.write_text(json.dumps(dataclasses.asdict(info), indent=2))


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
