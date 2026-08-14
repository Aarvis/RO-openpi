import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


@dataclasses.dataclass(frozen=True)
class LehomeRobotSplineInputs(transforms.DataTransformFn):
    """LeHome joint-state inputs for spline-conditioned pi0.5 training without image tokens."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Unsupported model type for spline-conditioned LeHome inputs: {self.model_type}")

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {},
            "image_mask": {},
        }

        if "robot_spline_coefficients" in data:
            inputs["robot_spline_coefficients"] = np.asarray(data["robot_spline_coefficients"], dtype=np.float32)
        if "robot_spline_knots" in data:
            inputs["robot_spline_knots"] = np.asarray(data["robot_spline_knots"], dtype=np.float32)

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class LehomeRobotSplineOutputs(transforms.DataTransformFn):
    action_dim: int = 12

    def __call__(self, data: dict) -> dict:
        result = dict(data)
        result["actions"] = np.asarray(data["actions"][:, : self.action_dim])
        return result
