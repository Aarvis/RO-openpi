from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object JSON at {path}, got {type(obj)}")
    return obj


def _mat4_from_any(x: Any, label: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{label} must be 4x4, got shape={arr.shape}")
    return arr


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def _is_row_vector_style(T: np.ndarray) -> bool:
    return (
        abs(float(T[3, 3]) - 1.0) < 1e-9
        and float(np.linalg.norm(T[3, :3])) > 1e-9
        and float(np.linalg.norm(T[:3, 3])) < 1e-9
    )


def _load_T_world_cam_cv(camera_cfg_path: Path) -> np.ndarray:
    cfg = _load_json(camera_cfg_path)
    T_raw = None
    for k in ("T_world_cam_cv", "T_world_camera_cv", "T_world_camera", "T_world_cam"):
        if k in cfg:
            T_raw = cfg[k]
            break
    if T_raw is None and isinstance(cfg.get("world_to_camera_source"), dict):
        src = cfg["world_to_camera_source"]
        for k in ("T_world_cam_cv", "T_world_camera_cv", "T_world_camera", "T_world_cam"):
            if k in src:
                T_raw = src[k]
                break
    if T_raw is None:
        raise KeyError(f"No T_world_cam_cv-like key found in: {camera_cfg_path}")

    T = _mat4_from_any(T_raw, "T_world_cam_cv")
    if _is_row_vector_style(T):
        T = T.T
    return T


def _load_camera_frame_z_flip_180(camera_cfg_path: Path) -> tuple[bool, np.ndarray]:
    cfg = _load_json(camera_cfg_path)
    flip_cfg = cfg.get("camera_frame_z_flip_180", {})
    if flip_cfg is None:
        return False, np.diag([-1.0, -1.0, 1.0]).astype(np.float64)
    if not isinstance(flip_cfg, dict):
        raise ValueError(f"camera_frame_z_flip_180 must be an object in: {camera_cfg_path}")

    enabled = bool(flip_cfg.get("enabled", False))
    R_flip = np.asarray(
        flip_cfg.get(
            "R_flip",
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ),
        dtype=np.float64,
    )
    if R_flip.shape != (3, 3):
        raise ValueError(f"camera_frame_z_flip_180.R_flip must be 3x3, got shape={R_flip.shape}")
    return enabled, R_flip


def _is_actuated_joint(joint_rec: dict[str, Any]) -> bool:
    return str(joint_rec.get("type", "revolute")).lower() in ("revolute", "prismatic")


def _axis_angle_rot(axis: np.ndarray, q_rad: float) -> np.ndarray:
    u = _normalize(axis.astype(np.float64))
    ux, uy, uz = u
    c = math.cos(q_rad)
    s = math.sin(q_rad)
    v = 1.0 - c
    return np.array(
        [
            [ux * ux * v + c, ux * uy * v - uz * s, ux * uz * v + uy * s],
            [uy * ux * v + uz * s, uy * uy * v + c, uy * uz * v - ux * s],
            [uz * ux * v - uy * s, uz * uy * v + ux * s, uz * uz * v + c],
        ],
        dtype=np.float64,
    )


def _motion_matrix(joint_type: str, axis_joint: list[float], q_rad: float) -> np.ndarray:
    jt = (joint_type or "").lower()
    axis = np.asarray(axis_joint if axis_joint else [0.0, 0.0, 1.0], dtype=np.float64)
    if np.linalg.norm(axis) < 1e-12:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    if jt == "revolute":
        T[:3, :3] = _axis_angle_rot(axis, q_rad)
    elif jt == "prismatic":
        T[:3, 3] = _normalize(axis) * q_rad
    return T


def _quat_wxyz_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError(f"Expected 4D quaternion, got {q}")
    w, x, y, z = q.tolist()
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    n = float(np.linalg.norm(quat))
    if n < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / n


def _quat_xyzw_to_order(quat_xyzw: np.ndarray, quat_order: str) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64).reshape(4).tolist()
    if quat_order == "xyzw":
        return np.asarray([x, y, z, w], dtype=np.float64)
    if quat_order == "wxyz":
        return np.asarray([w, x, y, z], dtype=np.float64)
    raise ValueError(f"Unsupported quat order: {quat_order}")


def _so3_log_rotvec_world(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Left-invariant SO(3) log error in world coordinates."""
    R_err = R_target @ R_current.T
    tr = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(tr)
    if theta < 1e-9:
        return np.zeros(3, dtype=np.float64)

    if abs(math.pi - theta) < 1e-6:
        axis = np.array(
            [
                math.sqrt(max((R_err[0, 0] + 1.0) * 0.5, 0.0)),
                math.sqrt(max((R_err[1, 1] + 1.0) * 0.5, 0.0)),
                math.sqrt(max((R_err[2, 2] + 1.0) * 0.5, 0.0)),
            ],
            dtype=np.float64,
        )
        k = int(np.argmax(axis))
        if axis[k] > 1e-12:
            if k == 0:
                axis[1] = (R_err[0, 1] + R_err[1, 0]) / (4.0 * axis[0])
                axis[2] = (R_err[0, 2] + R_err[2, 0]) / (4.0 * axis[0])
            elif k == 1:
                axis[0] = (R_err[0, 1] + R_err[1, 0]) / (4.0 * axis[1])
                axis[2] = (R_err[1, 2] + R_err[2, 1]) / (4.0 * axis[1])
            else:
                axis[0] = (R_err[0, 2] + R_err[2, 0]) / (4.0 * axis[2])
                axis[1] = (R_err[1, 2] + R_err[2, 1]) / (4.0 * axis[2])
        axis_n = float(np.linalg.norm(axis))
        if axis_n < 1e-12:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = axis / axis_n
    else:
        denom = max(2.0 * math.sin(theta), 1e-12)
        w_hat = (R_err - R_err.T) / denom
        axis = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=np.float64)
        axis_n = float(np.linalg.norm(axis))
        if axis_n < 1e-12:
            return np.zeros(3, dtype=np.float64)
        axis = axis / axis_n
    return axis * theta


def _weighted_error_cost(
    pos_norm: float,
    rot_norm: float,
    pos_weight: float,
    rot_weight: float,
) -> float:
    return float((pos_weight * pos_norm) ** 2 + (rot_weight * rot_norm) ** 2)


def _compute_ee_transform(
    T_world_base: np.ndarray,
    chain_joint_names: list[str],
    joint_transforms: dict[str, dict[str, Any]],
    joint_name_to_q: dict[str, float],
) -> np.ndarray:
    T = T_world_base.copy()
    for jn in chain_joint_names:
        rec = joint_transforms[jn]
        T_parent_joint = _mat4_from_any(rec["T_parent_joint_static"], f"{jn}.T_parent_joint_static")
        T_joint_child = _mat4_from_any(rec["T_joint_child_static"], f"{jn}.T_joint_child_static")
        axis_joint = rec.get("axis_joint_frame", [0.0, 0.0, 1.0])
        q = float(joint_name_to_q.get(jn, 0.0))
        T_motion = _motion_matrix(str(rec.get("type", "revolute")), axis_joint, q)
        T = T @ T_parent_joint @ T_motion @ T_joint_child
    return T


def _compute_fk_and_jacobian(
    T_world_base: np.ndarray,
    chain_joint_names: list[str],
    joint_transforms: dict[str, dict[str, Any]],
    joint_name_to_q: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      T_world_ee: (4,4)
      J_spatial_6xn: top=linear, bottom=angular
    """
    T = T_world_base.copy()

    joint_types: list[str] = []
    joint_axis_world: list[np.ndarray] = []
    joint_origin_world: list[np.ndarray] = []

    for jn in chain_joint_names:
        rec = joint_transforms[jn]
        T_parent_joint = _mat4_from_any(rec["T_parent_joint_static"], f"{jn}.T_parent_joint_static")
        T_joint_child = _mat4_from_any(rec["T_joint_child_static"], f"{jn}.T_joint_child_static")
        axis_joint = np.asarray(rec.get("axis_joint_frame", [0.0, 0.0, 1.0]), dtype=np.float64)
        if np.linalg.norm(axis_joint) < 1e-12:
            axis_joint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis_joint = _normalize(axis_joint)

        T = T @ T_parent_joint
        R_world_joint = T[:3, :3]
        o_world_joint = T[:3, 3].copy()
        z_world = R_world_joint @ axis_joint

        joint_type = str(rec.get("type", "revolute")).lower()
        if joint_type in ("revolute", "prismatic"):
            joint_types.append(joint_type)
            joint_axis_world.append(z_world)
            joint_origin_world.append(o_world_joint)

        q = float(joint_name_to_q.get(jn, 0.0))
        T_motion = _motion_matrix(str(rec.get("type", "revolute")), axis_joint.tolist(), q)
        T = T @ T_motion @ T_joint_child

    T_world_ee = T
    p_ee = T_world_ee[:3, 3].copy()

    n = len(joint_types)
    J = np.zeros((6, n), dtype=np.float64)
    for i in range(n):
        jt = joint_types[i]
        z = joint_axis_world[i]
        o = joint_origin_world[i]
        if jt == "revolute":
            J[:3, i] = np.cross(z, p_ee - o)
            J[3:, i] = z
        elif jt == "prismatic":
            J[:3, i] = z
            J[3:, i] = 0.0
    return T_world_ee, J


def _read_joint_limits(
    chain_joint_names: list[str],
    joint_transforms: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(chain_joint_names), -np.inf, dtype=np.float64)
    upper = np.full(len(chain_joint_names), np.inf, dtype=np.float64)
    for i, jn in enumerate(chain_joint_names):
        rec = joint_transforms[jn]
        lim = rec.get("limit_as_rad", {})
        lo = lim.get("lower")
        hi = lim.get("upper")
        if lo is not None:
            lower[i] = float(lo)
        if hi is not None:
            upper[i] = float(hi)
    return lower, upper


class LehomeFKCameraCVTransformer:
    def __init__(
        self,
        fk_json_path: str | Path,
        camera_config_json_path: str | Path,
        *,
        state_unit: str = "rad",
        dataset_joint_order: tuple[str, ...] = DEFAULT_DATASET_JOINT_ORDER,
        pose_quat_order: str = "wxyz",
    ) -> None:
        self._fk_path = Path(fk_json_path).resolve()
        self._camera_cfg_path = Path(camera_config_json_path).resolve()
        self._state_unit = state_unit
        self._dataset_joint_order = tuple(dataset_joint_order)
        self._pose_quat_order = pose_quat_order
        if self._state_unit not in ("rad", "deg"):
            raise ValueError(f"Unsupported state_unit={self._state_unit}, expected 'rad' or 'deg'.")

        self._fk = _load_json(self._fk_path)
        self._joint_transforms = dict(self._fk["joint_transforms"])
        self._chain_joint_names = list(self._fk["ordered_chains"]["ordered_joint_names_to_ee"])
        self._actuated_chain_joint_names = [
            jn for jn in self._chain_joint_names if _is_actuated_joint(self._joint_transforms[jn])
        ]
        self._lower_lim_rad, self._upper_lim_rad = _read_joint_limits(
            self._actuated_chain_joint_names, self._joint_transforms
        )
        self._name_to_idx = {name: i for i, name in enumerate(self._dataset_joint_order)}
        missing = [n for n in self._actuated_chain_joint_names + ["gripper"] if n not in self._name_to_idx]
        if missing:
            raise ValueError(f"dataset_joint_order missing required names: {missing}")

        self._T_world_base_left = _mat4_from_any(
            self._fk["base_setup"]["left_arm_base_world"]["T_world_base"],
            "left base T_world_base",
        )
        self._T_world_base_right = _mat4_from_any(
            self._fk["base_setup"]["right_arm_base_world"]["T_world_base"],
            "right base T_world_base",
        )

        self._T_world_cam_cv = _load_T_world_cam_cv(self._camera_cfg_path)
        self._T_cam_cv_world = np.linalg.inv(self._T_world_cam_cv)
        self._camera_frame_z_flip_enabled, R_flip = _load_camera_frame_z_flip_180(self._camera_cfg_path)
        self._T_camera_frame_z_flip = np.eye(4, dtype=np.float64)
        self._T_camera_frame_z_flip[:3, :3] = R_flip
        self._T_camera_frame_z_unflip = np.linalg.inv(self._T_camera_frame_z_flip)

    def state12_to_camera_pose16(self, state12: np.ndarray) -> np.ndarray:
        state = np.asarray(state12, dtype=np.float64).reshape(-1)
        if state.size != 12:
            raise ValueError(f"Expected 12D observation.state, got {state.size}")
        left_pose8, right_pose8 = self._state12_to_world_pose8_pair(state)
        left_cam = self._world_pose8_to_cam_pose8(left_pose8)
        right_cam = self._world_pose8_to_cam_pose8(right_pose8)
        pose16 = np.concatenate([left_cam, right_cam], axis=0)
        if self._camera_frame_z_flip_enabled:
            pose16 = self._apply_camera_frame_z_flip_pose16(pose16)
        return pose16.astype(np.float32)

    def camera_pose16_to_world_pose16(self, pose16_camera_cv: np.ndarray) -> np.ndarray:
        pose16 = np.asarray(pose16_camera_cv, dtype=np.float64).reshape(-1)
        if pose16.size < 16:
            raise ValueError(f"Expected at least 16D camera action pose, got {pose16.size}")
        if self._camera_frame_z_flip_enabled:
            pose16 = self._apply_camera_frame_z_flip_pose16(pose16[:16], inverse=True)
        left_world = self._cam_pose8_to_world_pose8(pose16[:8])
        right_world = self._cam_pose8_to_world_pose8(pose16[8:16])
        return np.concatenate([left_world, right_world], axis=0).astype(np.float64)

    def solve_world_pose16_to_state12(
        self,
        target_pose16_world: np.ndarray,
        q_state12_current: np.ndarray,
        *,
        damping: float = 0.05,
        alpha: float = 1.0,
        line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.05, 1.5, 2.0),
        fallback_tol_factor: float = 1.1,
        pos_weight: float = 1.0,
        rot_weight: float = 1.0,
        max_iters: int = 80,
        tol_pos_m: float = 1e-4,
        tol_rot_deg: float = 0.2,
        max_step_norm: float = 0.2,
        enforce_limits: bool = True,
    ) -> np.ndarray:
        target_pose = np.asarray(target_pose16_world, dtype=np.float64).reshape(-1)
        if target_pose.size < 16:
            raise ValueError(f"Expected at least 16D world target pose, got {target_pose.size}")
        q_cur = np.asarray(q_state12_current, dtype=np.float64).reshape(-1)
        if q_cur.size < 12:
            raise ValueError(f"Expected 12D current joint state, got {q_cur.size}")

        q_cur_local = q_cur[:12].copy()
        if self._state_unit == "deg":
            q_cur_local = np.deg2rad(q_cur_local)

        left_out = self._solve_arm_ik_to_pose8(
            arm="left",
            q_arm_state=q_cur_local[:6],
            target_pose8_world=target_pose[:8],
            damping=damping,
            alpha=alpha,
            line_search_alphas=line_search_alphas,
            fallback_tol_factor=fallback_tol_factor,
            pos_weight=pos_weight,
            rot_weight=rot_weight,
            max_iters=max_iters,
            tol_pos_m=tol_pos_m,
            tol_rot_deg=tol_rot_deg,
            max_step_norm=max_step_norm,
            enforce_limits=enforce_limits,
        )
        right_out = self._solve_arm_ik_to_pose8(
            arm="right",
            q_arm_state=q_cur_local[6:12],
            target_pose8_world=target_pose[8:16],
            damping=damping,
            alpha=alpha,
            line_search_alphas=line_search_alphas,
            fallback_tol_factor=fallback_tol_factor,
            pos_weight=pos_weight,
            rot_weight=rot_weight,
            max_iters=max_iters,
            tol_pos_m=tol_pos_m,
            tol_rot_deg=tol_rot_deg,
            max_step_norm=max_step_norm,
            enforce_limits=enforce_limits,
        )
        q_out = np.concatenate([left_out, right_out], axis=0)
        if self._state_unit == "deg":
            q_out = np.rad2deg(q_out)
        return q_out.astype(np.float64)

    def _state12_to_world_pose8_pair(self, state12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left_state = np.asarray(state12[:6], dtype=np.float64).copy()
        right_state = np.asarray(state12[6:12], dtype=np.float64).copy()
        if self._state_unit == "deg":
            left_state = np.deg2rad(left_state)
            right_state = np.deg2rad(right_state)

        left_map = {n: float(left_state[self._name_to_idx[n]]) for n in self._actuated_chain_joint_names}
        right_map = {n: float(right_state[self._name_to_idx[n]]) for n in self._actuated_chain_joint_names}
        left_gripper = float(left_state[self._name_to_idx["gripper"]])
        right_gripper = float(right_state[self._name_to_idx["gripper"]])

        T_left = _compute_ee_transform(
            self._T_world_base_left, self._chain_joint_names, self._joint_transforms, left_map
        )
        T_right = _compute_ee_transform(
            self._T_world_base_right, self._chain_joint_names, self._joint_transforms, right_map
        )
        return (
            self._transform_to_pose8(T_left, left_gripper),
            self._transform_to_pose8(T_right, right_gripper),
        )

    def _transform_to_pose8(self, T_world_ee: np.ndarray, gripper: float) -> np.ndarray:
        pos = np.asarray(T_world_ee[:3, 3], dtype=np.float64)
        quat_xyzw = _rot_to_quat_xyzw(T_world_ee[:3, :3])
        quat = _quat_xyzw_to_order(quat_xyzw, self._pose_quat_order)
        return np.concatenate([pos, quat, np.asarray([gripper], dtype=np.float64)], axis=0)

    def _world_pose8_to_cam_pose8(self, pose8_world: np.ndarray) -> np.ndarray:
        T_world_ee = self._pose8_to_transform(pose8_world)
        T_cam_ee = self._T_cam_cv_world @ T_world_ee
        return self._transform_to_pose8(T_cam_ee, float(pose8_world[7]))

    def _cam_pose8_to_world_pose8(self, pose8_cam: np.ndarray) -> np.ndarray:
        T_cam_ee = self._pose8_to_transform(pose8_cam)
        T_world_ee = self._T_world_cam_cv @ T_cam_ee
        return self._transform_to_pose8(T_world_ee, float(pose8_cam[7]))

    def _apply_camera_frame_z_flip_pose8(self, pose8_cam: np.ndarray, *, inverse: bool = False) -> np.ndarray:
        pose = np.asarray(pose8_cam, dtype=np.float64).reshape(8)
        T_cam_ee = self._pose8_to_transform(pose)
        T_flip = self._T_camera_frame_z_unflip if inverse else self._T_camera_frame_z_flip
        return self._transform_to_pose8(T_flip @ T_cam_ee, float(pose[7]))

    def _apply_camera_frame_z_flip_pose16(self, pose16_cam: np.ndarray, *, inverse: bool = False) -> np.ndarray:
        pose = np.asarray(pose16_cam, dtype=np.float64).reshape(-1)
        if pose.size < 16:
            raise ValueError(f"Expected at least 16D camera pose, got {pose.size}")
        left = self._apply_camera_frame_z_flip_pose8(pose[:8], inverse=inverse)
        right = self._apply_camera_frame_z_flip_pose8(pose[8:16], inverse=inverse)
        if pose.size == 16:
            return np.concatenate([left, right], axis=0)
        return np.concatenate([left, right, pose[16:]], axis=0)

    def _pose8_to_transform(self, pose8: np.ndarray) -> np.ndarray:
        v = np.asarray(pose8, dtype=np.float64).reshape(-1)
        if v.size != 8:
            raise ValueError(f"Expected 8D pose8, got {v.size}")
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = v[:3]
        quat = v[3:7]
        if self._pose_quat_order == "wxyz":
            T[:3, :3] = _quat_wxyz_to_rot(quat)
        elif self._pose_quat_order == "xyzw":
            x, y, z, w = quat.tolist()
            T[:3, :3] = _quat_wxyz_to_rot(np.asarray([w, x, y, z], dtype=np.float64))
        else:
            raise ValueError(f"Unsupported pose_quat_order: {self._pose_quat_order}")
        return T

    def _solve_arm_ik_to_pose8(
        self,
        *,
        arm: str,
        q_arm_state: np.ndarray,
        target_pose8_world: np.ndarray,
        damping: float,
        alpha: float,
        line_search_alphas: tuple[float, ...],
        fallback_tol_factor: float,
        pos_weight: float,
        rot_weight: float,
        max_iters: int,
        tol_pos_m: float,
        tol_rot_deg: float,
        max_step_norm: float,
        enforce_limits: bool,
    ) -> np.ndarray:
        q_arm = np.asarray(q_arm_state, dtype=np.float64).reshape(-1)
        if q_arm.size != 6:
            raise ValueError(f"Expected 6D arm state for {arm}, got {q_arm.size}")

        name_to_idx = self._name_to_idx
        q_chain = np.asarray([q_arm[name_to_idx[jn]] for jn in self._actuated_chain_joint_names], dtype=np.float64)
        T_target = self._pose8_to_transform(target_pose8_world)
        q_chain_out = self._solve_dls_ik_chain(
            arm=arm,
            q_init_chain=q_chain,
            T_target_world_ee=T_target,
            damping=damping,
            alpha=alpha,
            line_search_alphas=line_search_alphas,
            fallback_tol_factor=fallback_tol_factor,
            pos_weight=pos_weight,
            rot_weight=rot_weight,
            max_iters=max_iters,
            tol_pos_m=tol_pos_m,
            tol_rot_deg=tol_rot_deg,
            max_step_norm=max_step_norm,
            enforce_limits=enforce_limits,
        )

        q_out = q_arm.copy()
        for j, jn in enumerate(self._actuated_chain_joint_names):
            q_out[name_to_idx[jn]] = q_chain_out[j]

        if ("gripper" in name_to_idx) and ("gripper" not in self._actuated_chain_joint_names):
            q_out[name_to_idx["gripper"]] = float(np.asarray(target_pose8_world, dtype=np.float64)[7])
        return q_out

    def _solve_dls_ik_chain(
        self,
        *,
        arm: str,
        q_init_chain: np.ndarray,
        T_target_world_ee: np.ndarray,
        damping: float,
        alpha: float,
        line_search_alphas: tuple[float, ...],
        fallback_tol_factor: float,
        pos_weight: float,
        rot_weight: float,
        max_iters: int,
        tol_pos_m: float,
        tol_rot_deg: float,
        max_step_norm: float,
        enforce_limits: bool,
    ) -> np.ndarray:
        if arm == "left":
            T_world_base = self._T_world_base_left
        elif arm == "right":
            T_world_base = self._T_world_base_right
        else:
            raise ValueError(f"Unsupported arm: {arm}")

        q = np.asarray(q_init_chain, dtype=np.float64).copy()
        if q.size != len(self._actuated_chain_joint_names):
            raise ValueError(
                f"q_init_chain size mismatch: got {q.size}, expected {len(self._actuated_chain_joint_names)}"
            )

        ls_factors = [float(v) for v in line_search_alphas if float(v) > 0.0]
        if not ls_factors:
            ls_factors = [1.0]

        tol_rot_rad = math.radians(float(tol_rot_deg))
        fallback_tol_factor = float(max(fallback_tol_factor, 1.0))
        for _ in range(int(max_iters)):
            q_map = {jn: float(q[i]) for i, jn in enumerate(self._actuated_chain_joint_names)}
            T_cur, J = _compute_fk_and_jacobian(
                T_world_base=T_world_base,
                chain_joint_names=self._chain_joint_names,
                joint_transforms=self._joint_transforms,
                joint_name_to_q=q_map,
            )
            p_cur = T_cur[:3, 3]
            R_cur = T_cur[:3, :3]
            p_tgt = T_target_world_ee[:3, 3]
            R_tgt = T_target_world_ee[:3, :3]
            e_pos = p_tgt - p_cur
            e_rot = _so3_log_rotvec_world(R_tgt, R_cur)
            pos_norm = float(np.linalg.norm(e_pos))
            rot_norm = float(np.linalg.norm(e_rot))
            if pos_norm <= tol_pos_m and rot_norm <= tol_rot_rad:
                break

            dx = np.concatenate([pos_weight * e_pos, rot_weight * e_rot], axis=0)
            W = np.diag([pos_weight, pos_weight, pos_weight, rot_weight, rot_weight, rot_weight])
            Jw = W @ J
            A = Jw @ Jw.T + (damping * damping) * np.eye(6, dtype=np.float64)
            dq = Jw.T @ np.linalg.solve(A, dx)

            step_norm = float(np.linalg.norm(dq))
            if max_step_norm > 0.0 and step_norm > max_step_norm:
                dq = dq * (max_step_norm / max(step_norm, 1e-12))

            cur_cost = _weighted_error_cost(pos_norm, rot_norm, pos_weight, rot_weight)
            best_q = None
            best_cost = float("inf")
            for ls in ls_factors:
                q_try = q + float(alpha * ls) * dq
                if enforce_limits:
                    q_try = np.minimum(np.maximum(q_try, self._lower_lim_rad), self._upper_lim_rad)

                q_try_map = {jn: float(q_try[i]) for i, jn in enumerate(self._actuated_chain_joint_names)}
                T_try, _ = _compute_fk_and_jacobian(
                    T_world_base=T_world_base,
                    chain_joint_names=self._chain_joint_names,
                    joint_transforms=self._joint_transforms,
                    joint_name_to_q=q_try_map,
                )
                e_pos_try = p_tgt - T_try[:3, 3]
                e_rot_try = _so3_log_rotvec_world(R_tgt, T_try[:3, :3])
                cost_try = _weighted_error_cost(
                    float(np.linalg.norm(e_pos_try)),
                    float(np.linalg.norm(e_rot_try)),
                    pos_weight,
                    rot_weight,
                )
                if cost_try < best_cost:
                    best_cost = cost_try
                    best_q = q_try

            if best_q is None:
                break
            if best_cost < cur_cost:
                q = best_q
            else:
                # Fallback accept if error is already close enough.
                if pos_norm <= fallback_tol_factor * tol_pos_m and rot_norm <= fallback_tol_factor * tol_rot_rad:
                    break
                # Last-resort take best candidate to keep motion progressing.
                q = best_q

        if enforce_limits:
            q = np.minimum(np.maximum(q, self._lower_lim_rad), self._upper_lim_rad)
        return q


@functools.lru_cache(maxsize=8)
def get_transformer_cached(
    fk_json_path: str,
    camera_config_json_path: str,
    state_unit: str,
    dataset_joint_order_csv: str,
    pose_quat_order: str,
) -> LehomeFKCameraCVTransformer:
    dataset_joint_order = tuple(s.strip() for s in dataset_joint_order_csv.split(",") if s.strip())
    if len(dataset_joint_order) != 6:
        raise ValueError(f"dataset_joint_order must have 6 names, got {len(dataset_joint_order)}")
    return LehomeFKCameraCVTransformer(
        fk_json_path=fk_json_path,
        camera_config_json_path=camera_config_json_path,
        state_unit=state_unit,
        dataset_joint_order=dataset_joint_order,
        pose_quat_order=pose_quat_order,
    )
