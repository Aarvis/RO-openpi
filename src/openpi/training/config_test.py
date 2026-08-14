from openpi.models import pi0_config
from openpi.training import config as training_config
from openpi import transforms as _transforms


def test_model_transform_factory_skips_resize_images_for_robot_spline_without_image_prefix():
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        action_dim=12,
        discrete_state_input=True,
        robot_spline=pi0_config.RobotSplineConfig(
            enabled=True,
            use_image_prefix=False,
        ),
    )

    transforms = training_config.ModelTransformFactory()(model_config)

    assert not any(isinstance(transform, _transforms.ResizeImages) for transform in transforms.inputs)


def test_model_transform_factory_keeps_resize_images_for_standard_pi05():
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        action_dim=12,
        discrete_state_input=True,
    )

    transforms = training_config.ModelTransformFactory()(model_config)

    assert any(isinstance(transform, _transforms.ResizeImages) for transform in transforms.inputs)


def test_robot_spline_joint_delta_finetune_uses_base_pi05_action_dim():
    cfg = training_config.get_config("pi05_lehome_robot_spline_joint_delta_finetune")

    assert isinstance(cfg.model, pi0_config.Pi0Config)
    assert cfg.model.action_dim == 32
    assert cfg.data.action_dim == 12
