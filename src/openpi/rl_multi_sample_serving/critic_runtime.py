from __future__ import annotations

from contextlib import contextmanager
from contextlib import nullcontext
import importlib.util
import logging
import os
import pathlib
from pathlib import Path
import sys
import threading
import time
from types import ModuleType

import numpy as np
import torch


logger = logging.getLogger(__name__)


def _resolve_critic_model_path(repo_root: Path | str) -> Path:
    repo_root = Path(repo_root).resolve()
    model_path = repo_root / "Datasets" / "OnlineRL" / "Critic" / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"Critic model.py not found: {model_path}")
    return model_path


def _load_critic_model_module(repo_root: Path | str) -> ModuleType:
    model_path = _resolve_critic_model_path(repo_root)
    spec = importlib.util.spec_from_file_location("lehome_online_rl_critic_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for critic model module: {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_amp_dtype(device: torch.device, amp_dtype: str) -> torch.dtype | None:
    if device.type != "cuda" or amp_dtype == "none":
        return None
    if amp_dtype == "auto":
        return torch.bfloat16
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    if amp_dtype == "float16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype}")


def _make_autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


@contextmanager
def _allow_cross_platform_path_unpickle():
    if os.name == "nt":
        yield
        return

    original_windows_path = getattr(pathlib, "WindowsPath", None)
    try:
        pathlib.WindowsPath = pathlib.PureWindowsPath
        yield
    finally:
        if original_windows_path is not None:
            pathlib.WindowsPath = original_windows_path


def _set_cuda_memory_fraction(device: torch.device, fraction: float | None) -> None:
    if fraction is None or device.type != "cuda":
        return
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"critic_gpu_fraction must be in (0, 1], got {fraction}")
    device_index = 0 if device.index is None else int(device.index)
    setter = getattr(torch.cuda, "set_per_process_memory_fraction", None)
    if setter is None:
        logger.warning("torch.cuda.set_per_process_memory_fraction is unavailable; ignoring critic_gpu_fraction")
        return
    setter(float(fraction), device_index)


class OnlineRLCriticRuntime:
    def __init__(
        self,
        *,
        critic_repo_root: Path | str,
        checkpoint_path: Path | str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        amp_dtype: str = "auto",
        gpu_memory_fraction: float | None = None,
    ) -> None:
        self._device = torch.device(device)
        _set_cuda_memory_fraction(self._device, gpu_memory_fraction)
        self._amp_dtype = _resolve_amp_dtype(self._device, amp_dtype)
        self._module = _load_critic_model_module(critic_repo_root)
        self._model = self._load_checkpoint(Path(checkpoint_path).resolve())
        self._lock = threading.Lock()

    @property
    def device(self) -> torch.device:
        return self._device

    def _load_checkpoint(self, checkpoint_path: Path):
        with _allow_cross_platform_path_unpickle():
            checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        model_config = self._module.CriticModelConfig(**checkpoint["model_config"])
        model = self._module.OnlineRLChunkCritic(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self._device)
        model.eval()
        logger.info("Loaded critic checkpoint from %s on %s", checkpoint_path, self._device)
        return model

    def score(
        self,
        *,
        policy_latent: np.ndarray,
        state: np.ndarray,
        action_chunk: np.ndarray,
    ) -> dict[str, object]:
        start_time = time.monotonic()
        policy_latent_np = np.array(policy_latent, dtype=np.float32, copy=True)
        state_np = np.array(state, dtype=np.float32, copy=True)
        action_chunk_np = np.array(action_chunk, dtype=np.float32, copy=True)
        with self._lock:
            with torch.inference_mode():
                with _make_autocast_context(self._device, self._amp_dtype):
                    prediction = self._model(
                        torch.as_tensor(policy_latent_np, dtype=torch.float32, device=self._device),
                        torch.as_tensor(state_np, dtype=torch.float32, device=self._device),
                        torch.as_tensor(action_chunk_np, dtype=torch.float32, device=self._device),
                    ).squeeze(-1)
                scores = prediction.to(torch.float32).detach().cpu().numpy()
        return {
            "scores": scores,
            "critic_timing": {
                "infer_ms": (time.monotonic() - start_time) * 1000.0,
                "num_samples": int(scores.shape[0]),
            },
        }
