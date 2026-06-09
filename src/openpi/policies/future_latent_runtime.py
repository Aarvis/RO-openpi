from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch
from torch import nn

import openpi.models.pi0_config as pi0_config
import openpi.shared.download as download
import openpi.shared.future_latent_order as _future_latent_order

from future_latent_predictor.future_predictor.model import FutureLatentPredictor
from future_latent_predictor.resampler_autoencoder.model import CAMERAS
from future_latent_predictor.resampler_autoencoder.model import ResamplerAutoencoder


logger = logging.getLogger(__name__)

_IMAGE_KEY_BY_CAMERA = {
    "top": "base_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
}


class FutureLatentRuntime:
    """Inference-time PyTorch future-latent predictor used by future-latent VLA policies."""

    def __init__(
        self,
        *,
        config: pi0_config.FutureLatentConfig,
        device: str | None = None,
    ) -> None:
        if not config.resampler_checkpoint_path:
            raise ValueError("Future latent inference requires resampler_checkpoint_path.")
        if not config.future_predictor_checkpoint_path:
            raise ValueError("Future latent inference requires future_predictor_checkpoint_path.")

        self._config = config
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._resampler = self._load_resampler(Path(download.maybe_download(config.resampler_checkpoint_path)))
        self._policy_camera_order = tuple(config.policy_camera_order)
        self._policy_reorder_indices = _future_latent_order.camera_reorder_indices(
            self.cameras,
            self._policy_camera_order,
        )
        self._future_checkpoint_path = Path(download.maybe_download(config.future_predictor_checkpoint_path))
        self._future_checkpoint = torch.load(self._future_checkpoint_path, map_location="cpu")
        self._future_predictor: FutureLatentPredictor | None = None
        self._future_predictor_state_dim: int | None = None
        logger.info(
            "Initialized future latent runtime on %s with resampler=%s future_predictor=%s "
            "runtime_camera_order=%s policy_camera_order=%s reorder_indices=%s",
            self._device,
            config.resampler_checkpoint_path,
            config.future_predictor_checkpoint_path,
            self.cameras,
            self._policy_camera_order,
            self._policy_reorder_indices,
        )

    @property
    def cameras(self) -> tuple[str, ...]:
        return self._resampler.cameras

    def _load_resampler(self, checkpoint_path: Path) -> ResamplerAutoencoder:
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        resampler_config = checkpoint.get("config", {}).get("model", {})
        resampler = ResamplerAutoencoder(
            cameras=tuple(resampler_config.get("cameras", CAMERAS)),
            input_tokens=int(resampler_config.get("input_tokens", 256)),
            input_dim=int(resampler_config.get("input_dim", 2048)),
            latent_tokens=int(resampler_config.get("latent_tokens", 24)),
            latent_dim=int(resampler_config.get("latent_dim", 512)),
            encoder_layers=int(resampler_config.get("encoder_layers", 2)),
            decoder_layers=int(resampler_config.get("decoder_layers", 2)),
            num_heads=int(resampler_config.get("num_heads", 16)),
            mlp_ratio=float(resampler_config.get("mlp_ratio", 4.0)),
            dropout=float(resampler_config.get("dropout", 0.0)),
        ).to(self._device)

        if "encoder" in checkpoint:
            missing, unexpected = resampler.load_state_dict(checkpoint["encoder"], strict=False)
            unexpected = [key for key in unexpected if not key.startswith("camera_decoders.")]
            missing = [key for key in missing if key.startswith("camera_encoders.")]
            if unexpected:
                raise RuntimeError(f"Unexpected resampler encoder keys in {checkpoint_path}: {unexpected}")
            if missing:
                raise RuntimeError(f"Missing resampler encoder keys in {checkpoint_path}: {missing[:10]}")
        elif "model" in checkpoint:
            resampler.load_state_dict(checkpoint["model"], strict=True)
        else:
            resampler.load_state_dict(checkpoint, strict=True)

        resampler.camera_decoders = nn.ModuleDict()
        resampler.eval()
        for parameter in resampler.parameters():
            parameter.requires_grad_(False)
        return resampler

    def _get_future_predictor(self, *, state_dim: int) -> FutureLatentPredictor:
        if self._future_predictor is not None:
            if self._future_predictor_state_dim != state_dim:
                raise ValueError(
                    "Future predictor was initialized for state_dim="
                    f"{self._future_predictor_state_dim}, got runtime state_dim={state_dim}."
                )
            return self._future_predictor

        checkpoint = self._future_checkpoint
        model_config: dict[str, Any] = dict(checkpoint.get("config", {}).get("model", {}))
        checkpoint_state_dim = model_config.get("state_dim")
        predictor_state_dim = state_dim if checkpoint_state_dim is None else int(checkpoint_state_dim)
        if predictor_state_dim != state_dim:
            raise ValueError(
                f"Future predictor checkpoint expects state_dim={predictor_state_dim}, "
                f"but policy runtime state has dim={state_dim}."
            )

        predictor = FutureLatentPredictor(
            state_dim=predictor_state_dim,
            num_cameras=int(model_config.get("num_cameras", len(model_config.get("cameras", self.cameras)))),
            latent_tokens=int(model_config.get("latent_tokens", self._config.latent_tokens)),
            latent_dim=int(model_config.get("latent_dim", self._config.latent_dim)),
            num_state_tokens=int(model_config.get("num_state_tokens", 4)),
            state_hidden_dim=int(model_config.get("state_hidden_dim", 1024)),
            transformer_layers=int(model_config.get("transformer_layers", 6)),
            num_heads=int(model_config.get("num_heads", 16)),
            mlp_ratio=float(model_config.get("mlp_ratio", 4.0)),
            dropout=float(model_config.get("dropout", 0.0)),
            predict_residual=bool(model_config.get("predict_residual", True)),
        ).to(self._device)
        predictor.load_state_dict(checkpoint["model"], strict=True)
        predictor.eval()
        for parameter in predictor.parameters():
            parameter.requires_grad_(False)

        self._future_predictor = predictor
        self._future_predictor_state_dim = state_dim
        return predictor

    def add_future_latents(
        self,
        inputs: dict[str, Any],
        image_embeddings: dict[str, jax.Array],
    ) -> dict[str, Any]:
        """Inject predicted compact future latents into a transformed, batched policy input dict."""
        state = np.asarray(jax.device_get(inputs["state"]), dtype=np.float32)
        if state.ndim != 2:
            raise ValueError(f"Expected batched state with shape [B,D], got {state.shape}")

        embeddings = []
        valid_masks = []
        batch_size = state.shape[0]
        for camera in self.cameras:
            image_key = _IMAGE_KEY_BY_CAMERA.get(camera)
            if image_key is None:
                raise ValueError(f"Unsupported future latent camera {camera!r}.")
            if image_key in image_embeddings:
                embedding = np.asarray(jax.device_get(image_embeddings[image_key]), dtype=np.float32)
            else:
                embedding = np.zeros((batch_size, 256, 2048), dtype=np.float32)
            embeddings.append(embedding)

            mask = inputs.get("image_mask", {}).get(image_key)
            if mask is None:
                valid = np.ones((batch_size,), dtype=bool)
            else:
                valid = np.asarray(jax.device_get(mask), dtype=bool)
            valid_masks.append(valid)

        embedding_tensor = torch.from_numpy(np.stack(embeddings, axis=1)).to(self._device)
        valid_tensor = torch.from_numpy(np.stack(valid_masks, axis=1)).to(self._device)
        state_tensor = torch.from_numpy(state).to(self._device)

        predictor = self._get_future_predictor(state_dim=state.shape[-1])
        with torch.inference_mode():
            current_latents = self._resampler.encode(embedding_tensor)
            prediction = predictor(current_latents, state_tensor)["prediction"]
            prediction = prediction * valid_tensor[:, :, None, None].to(prediction.dtype)
            reorder = list(self._policy_reorder_indices)
            prediction = prediction[:, reorder]
            valid_tensor = valid_tensor[:, reorder]

        future_latent_pred = prediction.detach().cpu().numpy().astype(np.float32)
        future_latent_valid_mask = valid_tensor.detach().cpu().numpy().astype(bool)

        result = dict(inputs)
        result["future_latent_pred"] = future_latent_pred
        # In inference, true latents are unavailable. Keep this equal to pred so the Observation schema is complete.
        result["future_latent_true"] = future_latent_pred
        result["future_latent_valid_mask"] = future_latent_valid_mask
        return result
