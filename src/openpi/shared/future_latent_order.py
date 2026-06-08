from __future__ import annotations

SIDECAR_CAMERA_COLUMNS = ("top", "right_wrist", "left_wrist")
POLICY_FUTURE_LATENT_CAMERA_ORDER = ("top", "left_wrist", "right_wrist")


def camera_reorder_indices(source_order: tuple[str, ...], target_order: tuple[str, ...]) -> tuple[int, ...]:
    if len(set(source_order)) != len(source_order):
        raise ValueError(f"Future latent source camera order contains duplicates: {source_order}")
    if len(set(target_order)) != len(target_order):
        raise ValueError(f"Future latent target camera order contains duplicates: {target_order}")

    missing = [camera for camera in target_order if camera not in source_order]
    if missing:
        raise ValueError(
            "Future latent source camera order is missing cameras required by target order: "
            f"source={source_order}, target={target_order}, missing={tuple(missing)}"
        )
    return tuple(source_order.index(camera) for camera in target_order)
