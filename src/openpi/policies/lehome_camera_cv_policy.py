from __future__ import annotations

import dataclasses
from pathlib import Path

import einops
import numpy as np
from PIL import Image

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.lehome_camera_cv.fk_camera_transform import get_transformer_cached


_POLICY_DATA_DIR = (Path(__file__).resolve().parent / "lehome_camera_cv").resolve()
_DEFAULT_FK_JSON_PATH = (_POLICY_DATA_DIR / "fk_from_usd_common.json").resolve()
_DEFAULT_CAMERA_CFG_JSON_PATH = (_POLICY_DATA_DIR / "top_camera_config_runtime_cv.json").resolve()
_DEFAULT_IMAGE_MASK_CSV = "base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb"
_DEFAULT_FAST_IMAGE_MASK_CSV = "base_0_rgb,base_1_rgb,wrist_0_rgb"
_PIL_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def make_lehome_camera_cv_example() -> dict:
    """Creates a random input example for the LeHome camera-CV policy."""
    return {
        "observation/top_rgb": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/left_rgb": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/right_rgb": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.random.rand(12).astype(np.float32),
        "prompt": "fold the garment on the table",
    }


def _resize_image_exact(image: np.ndarray, *, target_height: int, target_width: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image with 3 channels, got shape={image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    resized = Image.fromarray(image).resize((target_width, target_height), resample=_PIL_BILINEAR)
    return np.asarray(resized)


def _parse_image(
    image,
    *,
    rotate_180: bool = False,
    target_height: int | None = None,
    target_width: int | None = None,
) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if rotate_180:
        image = np.rot90(image, 2, axes=(0, 1))
    if target_height is not None or target_width is not None:
        if target_height is None or target_width is None:
            raise ValueError("Both target_height and target_width must be provided together.")
        if image.shape[:2] != (target_height, target_width):
            image = _resize_image_exact(image, target_height=target_height, target_width=target_width)
    return image


def _parse_optional_image(
    image,
    *,
    fallback: np.ndarray,
    target_height: int | None = None,
    target_width: int | None = None,
) -> tuple[np.ndarray, np.bool_]:
    if image is None:
        return np.zeros_like(fallback), np.False_
    return _parse_image(image, target_height=target_height, target_width=target_width), np.True_


def _parse_mask_indices(indices_csv: str, *, vector_length: int) -> np.ndarray:
    mask = np.ones(vector_length, dtype=bool)
    for token in str(indices_csv).split(","):
        token = token.strip()
        if not token:
            continue
        index = int(token)
        if index < 0 or index >= vector_length:
            raise ValueError(f"Mask index {index} out of range for vector length {vector_length}")
        mask[index] = False
    return mask


def _parse_valid_image_names(names: tuple[str, ...], valid_names_csv: str) -> tuple[np.bool_, ...]:
    tokens = {token.strip() for token in str(valid_names_csv).split(",") if token.strip()}
    if not tokens:
        return tuple(np.False_ for _ in names)
    unknown = sorted(tokens.difference(names))
    if unknown:
        raise ValueError(f"Unknown image mask names {unknown}; expected one of {names}")
    return tuple(np.bool_(name in tokens) for name in names)


def _valid_image_names_csv_for_model(valid_image_names_csv: str, model_type: _model.ModelType) -> str:
    if model_type == _model.ModelType.PI0_FAST and valid_image_names_csv == _DEFAULT_IMAGE_MASK_CSV:
        return _DEFAULT_FAST_IMAGE_MASK_CSV
    return valid_image_names_csv


def _canonicalize_quaternion(
    quat: np.ndarray,
    *,
    quat_order: str,
    ref_quat: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4).copy()
    if ref_quat is not None:
        ref = np.asarray(ref_quat, dtype=np.float64).reshape(4)
        if float(np.linalg.norm(q)) > eps and float(np.linalg.norm(ref)) > eps and float(np.dot(q, ref)) < 0.0:
            q = -q
        return q

    scalar_idx = 0 if quat_order == "wxyz" else 3
    if q[scalar_idx] < -eps:
        return -q
    if abs(float(q[scalar_idx])) <= eps:
        for value in q:
            if abs(float(value)) <= eps:
                continue
            if float(value) < 0.0:
                return -q
            break
    return q


def _canonicalize_pose8_quaternion(
    pose8: np.ndarray,
    *,
    quat_order: str,
    ref_pose8: np.ndarray | None = None,
) -> np.ndarray:
    pose = np.asarray(pose8, dtype=np.float64).reshape(8).copy()
    ref_quat = None if ref_pose8 is None else np.asarray(ref_pose8, dtype=np.float64).reshape(8)[3:7]
    pose[3:7] = _canonicalize_quaternion(pose[3:7], quat_order=quat_order, ref_quat=ref_quat)
    return pose


def _canonicalize_pose16_quaternions(
    pose16: np.ndarray,
    *,
    quat_order: str,
    ref_pose16: np.ndarray | None = None,
) -> np.ndarray:
    pose = np.asarray(pose16, dtype=np.float64).reshape(16).copy()
    ref = None if ref_pose16 is None else np.asarray(ref_pose16, dtype=np.float64).reshape(16)
    pose[:8] = _canonicalize_pose8_quaternion(
        pose[:8],
        quat_order=quat_order,
        ref_pose8=None if ref is None else ref[:8],
    )
    pose[8:16] = _canonicalize_pose8_quaternion(
        pose[8:16],
        quat_order=quat_order,
        ref_pose8=None if ref is None else ref[8:16],
    )
    return pose


def _canonicalize_pose16_sequence(
    pose16_seq: np.ndarray,
    *,
    quat_order: str,
    initial_ref_pose16: np.ndarray | None = None,
) -> np.ndarray:
    seq = np.asarray(pose16_seq, dtype=np.float64)
    if seq.ndim != 2 or seq.shape[1] != 16:
        raise ValueError(f"Expected pose16 sequence with shape (T,16), got {seq.shape}")
    out = seq.copy()
    prev = None if initial_ref_pose16 is None else _canonicalize_pose16_quaternions(
        initial_ref_pose16,
        quat_order=quat_order,
    )
    for i in range(out.shape[0]):
        out[i] = _canonicalize_pose16_quaternions(out[i], quat_order=quat_order, ref_pose16=prev)
        prev = out[i]
    return out


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
    valid_image_names_csv: str = _DEFAULT_IMAGE_MASK_CSV
    masked_state_indices_csv: str = ""
    masked_action_indices_csv: str = ""
    fill_value: float = 0.0
    target_image_height: int | None = None
    target_image_width: int | None = None
    include_images: bool = True
    inference_mode: str = "real"

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
        if self.inference_mode not in {"real", "sim"}:
            raise ValueError(f"Unsupported inference_mode={self.inference_mode!r}. Expected 'real' or 'sim'.")

    def __call__(self, data: dict) -> dict:
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
        state_cam_cv = _canonicalize_pose16_quaternions(
            transformer.state12_to_camera_pose16(raw_state),
            quat_order=str(self.pose_quat_order),
        ).astype(np.float32)
        state_mask = _parse_mask_indices(self.masked_state_indices_csv, vector_length=16)
        state_cam_cv = state_cam_cv.copy()
        state_cam_cv[~state_mask] = float(self.fill_value)

        inputs = {
            "state": state_cam_cv,
            "state_mask": state_mask.astype(bool),
            # Preserve original 12D joints for output-side IK post-processing.
            "state_joint": raw_state.astype(np.float32),
        }
        if self.include_images:
            top_image = _parse_image(
                data["observation/top_rgb"],
                rotate_180=self.inference_mode == "sim",
                target_height=self.target_image_height,
                target_width=self.target_image_width,
            )
            left_image = _parse_image(
                data["observation/left_rgb"],
                target_height=self.target_image_height,
                target_width=self.target_image_width,
            )
            right_image = _parse_image(
                data["observation/right_rgb"],
                target_height=self.target_image_height,
                target_width=self.target_image_width,
            )

            match self.model_type:
                case _model.ModelType.PI0 | _model.ModelType.PI05:
                    names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                    images = (top_image, left_image, right_image)
                    image_masks = _parse_valid_image_names(
                        names,
                        _valid_image_names_csv_for_model(self.valid_image_names_csv, self.model_type),
                    )
                case _model.ModelType.PI0_FAST:
                    names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                    images = (top_image, left_image, right_image)
                    image_masks = _parse_valid_image_names(
                        names,
                        _valid_image_names_csv_for_model(self.valid_image_names_csv, self.model_type),
                    )
                case _:
                    raise ValueError(f"Unsupported model type: {self.model_type}")

            inputs["image"] = dict(zip(names, images, strict=True))
            inputs["image_mask"] = dict(zip(names, image_masks, strict=True))

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
                actions_cam_cv = _canonicalize_pose16_sequence(
                    transformer.state12_to_camera_pose16(actions)[np.newaxis, :],
                    quat_order=str(self.pose_quat_order),
                    initial_ref_pose16=state_cam_cv,
                )[0].astype(np.float32)
            elif actions.ndim == 2:
                if actions.shape[-1] != 12:
                    raise ValueError(
                        f"Expected 2D actions with last dim 12, got shape={actions.shape}"
                    )
                actions_cam_cv = _canonicalize_pose16_sequence(
                    np.stack(
                        [transformer.state12_to_camera_pose16(a) for a in actions],
                        axis=0,
                    ),
                    quat_order=str(self.pose_quat_order),
                    initial_ref_pose16=state_cam_cv,
                ).astype(np.float32)
            else:
                raise ValueError(
                    f"Unsupported actions shape {actions.shape}. Expected (12,) or (T,12)."
                )
            action_mask_1d = _parse_mask_indices(self.masked_action_indices_csv, vector_length=16)
            actions_cam_cv = actions_cam_cv.copy()
            actions_cam_cv[..., ~action_mask_1d] = float(self.fill_value)
            inputs["actions"] = actions_cam_cv
            inputs["action_mask"] = np.broadcast_to(action_mask_1d, actions_cam_cv.shape).copy()

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        for key in ("future_latent_pred", "future_latent_true", "future_latent_valid_mask"):
            if key in data:
                inputs[key] = data[key]

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
        state_cam_cv = _canonicalize_pose16_quaternions(state_cam_cv[:16], quat_order=str(self.pose_quat_order))

        actions = _canonicalize_pose16_sequence(
            actions[:, : self.model_action_dim],
            quat_order=str(self.pose_quat_order),
            initial_ref_pose16=state_cam_cv,
        )

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
            target_cam16 = actions[i]
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

        result = dict(data)
        result["actions"] = np.asarray(out_actions, dtype=np.float32)
        return result


@dataclasses.dataclass(frozen=True)
class LehomePrecomputed16DInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    state_dim: int = 16
    gripper_dim_indices_csv: str = "7,15"
    valid_image_names_csv: str = _DEFAULT_IMAGE_MASK_CSV
    masked_state_indices_csv: str = ""
    masked_action_indices_csv: str = ""
    fill_value: float = 0.0
    target_image_height: int | None = None
    target_image_width: int | None = None
    include_images: bool = True

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32).reshape(-1)
        if state.size != self.state_dim:
            raise ValueError(f"Expected observation/state {self.state_dim}D, got {state.size}")

        invalid_indices = ",".join(
            token
            for token in (str(self.gripper_dim_indices_csv) + "," + str(self.masked_state_indices_csv)).split(",")
            if token.strip()
        )
        valid_dim_mask = _parse_mask_indices(invalid_indices, vector_length=self.state_dim)
        masked_state = state.copy()
        masked_state[~valid_dim_mask] = self.fill_value

        inputs = {
            "state": masked_state,
            "state_mask": valid_dim_mask.astype(bool),
        }
        if self.include_images:
            top_image = _parse_image(
                data["observation/top_rgb"],
                target_height=self.target_image_height,
                target_width=self.target_image_width,
            )
            left_image, left_valid = _parse_optional_image(
                data.get("observation/left_rgb"),
                fallback=top_image,
                target_height=self.target_image_height,
                target_width=self.target_image_width,
            )
            right_image, right_valid = _parse_optional_image(
                data.get("observation/right_rgb"),
                fallback=top_image,
                target_height=self.target_image_height,
                target_width=self.target_image_width,
            )

            match self.model_type:
                case _model.ModelType.PI0 | _model.ModelType.PI05:
                    names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                    images = (top_image, left_image, right_image)
                    configured_masks = _parse_valid_image_names(
                        names,
                        _valid_image_names_csv_for_model(self.valid_image_names_csv, self.model_type),
                    )
                    image_masks = (
                        configured_masks[0],
                        np.bool_(configured_masks[1] and left_valid),
                        np.bool_(configured_masks[2] and right_valid),
                    )
                case _model.ModelType.PI0_FAST:
                    names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                    images = (top_image, np.zeros_like(top_image), right_image)
                    configured_masks = _parse_valid_image_names(
                        names,
                        _valid_image_names_csv_for_model(self.valid_image_names_csv, self.model_type),
                    )
                    image_masks = (
                        configured_masks[0],
                        configured_masks[1],
                        np.bool_(configured_masks[2] and right_valid),
                    )
                case _:
                    raise ValueError(f"Unsupported model type: {self.model_type}")

            inputs["image"] = dict(zip(names, images, strict=True))
            inputs["image_mask"] = dict(zip(names, image_masks, strict=True))

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim == 1:
                if actions.shape[0] != self.state_dim:
                    raise ValueError(
                        f"Expected 1D actions with {self.state_dim} values, got shape={actions.shape}"
                    )
            elif actions.ndim == 2:
                if actions.shape[-1] != self.state_dim:
                    raise ValueError(
                        f"Expected 2D actions with last dim {self.state_dim}, got shape={actions.shape}"
                    )
            else:
                raise ValueError(
                    f"Unsupported actions shape {actions.shape}. Expected ({self.state_dim},) or (T,{self.state_dim})."
                )

            action_invalid_indices = ",".join(
                token
                for token in (str(self.gripper_dim_indices_csv) + "," + str(self.masked_action_indices_csv)).split(",")
                if token.strip()
            )
            valid_action_mask = _parse_mask_indices(action_invalid_indices, vector_length=self.state_dim)
            masked_actions = actions.copy()
            masked_actions[..., ~valid_action_mask] = self.fill_value
            inputs["actions"] = masked_actions
            inputs["action_mask"] = np.broadcast_to(valid_action_mask, masked_actions.shape).copy()

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        for key in ("future_latent_pred", "future_latent_true", "future_latent_valid_mask"):
            if key in data:
                inputs[key] = data[key]

        return inputs
