import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_lehome_example() -> dict:
    """Creates a random input example for the LeHome policy."""
    return {
        "observation/top_rgb": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/left_rgb": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/right_rgb": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.random.rand(12).astype(np.float32),
        "prompt": "fold the garment on the table",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LehomeInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        top_image = _parse_image(data["observation/top_rgb"])
        left_image = _parse_image(data["observation/left_rgb"])
        right_image = _parse_image(data["observation/right_rgb"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (top_image, left_image, right_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (top_image, left_image, right_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LehomeOutputs(transforms.DataTransformFn):
    action_dim: int = 12

    def __call__(self, data: dict) -> dict:
        result = dict(data)
        result["actions"] = np.asarray(data["actions"][:, : self.action_dim])
        return result
