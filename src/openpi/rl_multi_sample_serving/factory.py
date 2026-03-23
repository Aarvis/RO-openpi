from __future__ import annotations

from pathlib import Path
from typing import Any

from openpi.policies import policy_config as _policy_config
from openpi.rl_multi_sample_serving.multi_sample_policy import MultiSamplePolicy
from openpi.training import config as _config
import openpi.transforms as transforms


def create_multi_sample_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> MultiSamplePolicy:
    base_policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        repack_transforms=repack_transforms,
        sample_kwargs=sample_kwargs,
        default_prompt=default_prompt,
        norm_stats=norm_stats,
        pytorch_device=pytorch_device,
    )
    return MultiSamplePolicy(base_policy)
