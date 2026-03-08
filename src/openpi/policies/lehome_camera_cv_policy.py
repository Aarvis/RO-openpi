from __future__ import annotations

import dataclasses
from pathlib import Path

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.lehome_camera_cv.fk_camera_transform import get_transformer_cached


_POLICY_DATA_DIR = (Path(__file__).resolve().parent / "lehome_camera_cv").resolve()
_DEFAULT_FK_JSON_PATH = (_POLICY_DATA_DIR / "fk_from_usd_common.json").resolve()
_DEFAULT_CAMERA_CFG_JSON_PATH = (_POLICY_DATA_DIR / "top_camera_config_runtime_cv.json").resolve()


def make_lehome_camera_cv_example() -> dict:
    """Creates a random input example for the LeHome camera-CV policy."""
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
class LehomeCameraCVInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    fk_json_path: str = str(_DEFAULT_FK_JSON_PATH)
    camera_config_json_path: str = str(_DEFAULT_CAMERA_CFG_JSON_PATH)
    state_unit: str = "rad"
    dataset_joint_order_csv: str = (
        "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper"
    )
    pose_quat_order: str = "wxyz"

    def __post_init__(self) -> None:
        fk_path = Path(self.fk_json_path).resolve()
        cam_path = Path(self.camera_config_json_path).resolve()
        if not fk_path.exists():
            raise FileNotFoundError(
                "Required FK JSON not found for LehomeCameraCVInputs: "
                f"{fk_path}. Expected file in {_POLICY_DATA_DIR}."
            )
        if not cam_path.exists():
            raise FileNotFoundError(
                "Required camera config JSON not found for LehomeCameraCVInputs: "
                f"{cam_path}. Expected file in {_POLICY_DATA_DIR}."
            )

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

        raw_state = np.asarray(data["observation/state"], dtype=np.float64).reshape(-1)
        if raw_state.size != 12:
            raise ValueError(f"Expected observation/state 12D, got {raw_state.size}")

        transformer = get_transformer_cached(
            fk_json_path=str(self.fk_json_path),
            camera_config_json_path=str(self.camera_config_json_path),
            state_unit=str(self.state_unit),
            dataset_joint_order_csv=str(self.dataset_joint_order_csv),
            pose_quat_order=str(self.pose_quat_order),
        )
        state_cam_cv = transformer.state12_to_camera_pose16(raw_state).astype(np.float32)

        inputs = {
            "state": state_cam_cv,
            # Preserve original 12D joints for output-side IK post-processing.
            "state_joint": raw_state.astype(np.float32),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            # Training path: convert LeHome joint-space actions (12D per step) to
            # camera-frame EE pose actions (16D per step) using the same FK+camera
            # transform as observation/state.
            actions = np.asarray(data["actions"], dtype=np.float64)
            if actions.ndim == 1:
                if actions.shape[0] != 12:
                    raise ValueError(
                        f"Expected 1D actions with 12 values, got shape={actions.shape}"
                    )
                actions_cam_cv = transformer.state12_to_camera_pose16(actions).astype(np.float32)
            elif actions.ndim == 2:
                if actions.shape[-1] != 12:
                    raise ValueError(
                        f"Expected 2D actions with last dim 12, got shape={actions.shape}"
                    )
                actions_cam_cv = np.stack(
                    [transformer.state12_to_camera_pose16(a) for a in actions],
                    axis=0,
                ).astype(np.float32)
            else:
                raise ValueError(
                    f"Unsupported actions shape {actions.shape}. Expected (12,) or (T,12)."
                )
            inputs["actions"] = actions_cam_cv

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LehomeCameraCVOutputs(transforms.DataTransformFn):
    model_action_dim: int = 16
    output_action_dim: int = 12
    fk_json_path: str = str(_DEFAULT_FK_JSON_PATH)
    camera_config_json_path: str = str(_DEFAULT_CAMERA_CFG_JSON_PATH)
    state_unit: str = "rad"
    dataset_joint_order_csv: str = (
        "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper"
    )
    pose_quat_order: str = "wxyz"
    damping: float = 0.05
    alpha: float = 1.0
    line_search_alphas_csv: str = "1,0.5,0.25,0.05,1.5,2"
    fallback_tol_factor: float = 1.1
    pos_weight: float = 1.0
    rot_weight: float = 1.0
    max_iters: int = 80
    tol_pos_m: float = 1e-4
    tol_rot_deg: float = 0.2
    max_step_norm: float = 0.2
    enforce_limits: bool = True

    def __post_init__(self) -> None:
        fk_path = Path(self.fk_json_path).resolve()
        cam_path = Path(self.camera_config_json_path).resolve()
        if not fk_path.exists():
            raise FileNotFoundError(
                "Required FK JSON not found for LehomeCameraCVOutputs: "
                f"{fk_path}. Expected file in {_POLICY_DATA_DIR}."
            )
        if not cam_path.exists():
            raise FileNotFoundError(
                "Required camera config JSON not found for LehomeCameraCVOutputs: "
                f"{cam_path}. Expected file in {_POLICY_DATA_DIR}."
            )

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
        if actions.ndim != 2:
            raise ValueError(f"Expected actions shape (T,D) or (D,), got {actions.shape}")
        if actions.shape[-1] < self.model_action_dim:
            raise ValueError(
                f"Model output has dim={actions.shape[-1]}, expected at least {self.model_action_dim}"
            )

        state_cam_cv = np.asarray(data["state"], dtype=np.float64).reshape(-1)
        if state_cam_cv.size < 16:
            raise ValueError(f"Expected state camera pose with at least 16 dims, got {state_cam_cv.size}")

        q_src = data.get("state_joint", data.get("state"))
        q_cur = np.asarray(q_src, dtype=np.float64).reshape(-1)
        if q_cur.size < self.output_action_dim:
            raise ValueError(
                f"Expected state joint source with at least {self.output_action_dim} dims, got {q_cur.size}"
            )
        q_cur = q_cur[: self.output_action_dim].copy()

        line_search_alphas = tuple(
            float(v.strip()) for v in str(self.line_search_alphas_csv).split(",") if v.strip()
        )

        transformer = get_transformer_cached(
            fk_json_path=str(self.fk_json_path),
            camera_config_json_path=str(self.camera_config_json_path),
            state_unit=str(self.state_unit),
            dataset_joint_order_csv=str(self.dataset_joint_order_csv),
            pose_quat_order=str(self.pose_quat_order),
        )

        out_actions = []
        # Convert each predicted 16D camera-CV target into 12D joints via world-frame IK.
        for i in range(actions.shape[0]):
            target_cam16 = actions[i, : self.model_action_dim]
            target_world16 = transformer.camera_pose16_to_world_pose16(target_cam16)
            q_hat = transformer.solve_world_pose16_to_state12(
                target_pose16_world=target_world16,
                q_state12_current=q_cur,
                damping=float(self.damping),
                alpha=float(self.alpha),
                line_search_alphas=line_search_alphas,
                fallback_tol_factor=float(self.fallback_tol_factor),
                pos_weight=float(self.pos_weight),
                rot_weight=float(self.rot_weight),
                max_iters=int(self.max_iters),
                tol_pos_m=float(self.tol_pos_m),
                tol_rot_deg=float(self.tol_rot_deg),
                max_step_norm=float(self.max_step_norm),
                enforce_limits=bool(self.enforce_limits),
            )
            out_actions.append(q_hat[: self.output_action_dim])
            # Roll the IK result forward for the next horizon step.
            q_cur = q_hat.copy()

        return {"actions": np.asarray(out_actions, dtype=np.float32)}
