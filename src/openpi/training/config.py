"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.lehome_camera_cv_policy as lehome_camera_cv_policy
import openpi.policies.lehome_policy as lehome_policy
import openpi.policies.lehome_robot_spline_policy as lehome_robot_spline_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.origami_vla_dataset as _origami_vla_dataset
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.shared.nnx_utils as nnx_utils
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False
    # If true, the dataset already stores full action chunks per sample and the loader must not
    # reconstruct them from future timesteps using delta_timestamps.
    prechunked_actions: bool = False
    # Optional multi-dataset specs for custom co-training loaders.
    multi_dataset_specs: Sequence[Any] = ()
    multi_state_unit: str = "rad"
    multi_pose_quat_order: str = "wxyz"
    multi_action_dim: int = 16
    multi_use_sample_weights: bool = False
    robot_spline_sidecar_root: str | None = None
    robot_spline_expand_pairings: bool = False
    robot_spline_sidecar_required: bool = True
    future_latent_sidecar_root: str | None = None
    future_latent_filter_included_datasets: bool = False
    future_latent_sidecar_required: bool = True
    origami_vla: _origami_vla_dataset.OrigamiVlaSettings | None = None
    dataset_split: Literal["train", "val", "all"] = "train"

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class NoOpTransformFactory(GroupFactory):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        del model_config
        return _transforms.Group()


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    @staticmethod
    def _uses_image_prefix(model_config: _model.BaseModelConfig) -> bool:
        if isinstance(model_config, pi0_config.Pi0Config):
            return not (model_config.robot_spline.enabled and not model_config.robot_spline.use_image_prefix)
        return True

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        resize_transforms = (
            [_transforms.ResizeImages(224, 224)] if self._uses_image_prefix(model_config) else []
        )
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        *resize_transforms,
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        *resize_transforms,
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                            discrete_tactile_input=model_config.origami_vla.tactile_prompt_input,
                            clip_discrete_inputs=model_config.origami_vla.prompt_discrete_clip,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        *resize_transforms,
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class LehomeCameraCVDatasetSpec:
    repo_id: str
    sample_weight: float = 1.0
    include_in_future_latent_dataset: bool = True
    apply_camera_cv_transform: bool = True
    fk_json_path: str = str(lehome_camera_cv_policy._DEFAULT_FK_JSON_PATH)
    camera_config_json_path: str = str(lehome_camera_cv_policy._DEFAULT_CAMERA_CFG_JSON_PATH)
    dataset_joint_order_csv: str = "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper"
    valid_image_names_csv: str = "base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb"
    masked_state_indices_csv: str = ""
    masked_action_indices_csv: str = ""
    target_image_height: int | None = None
    target_image_width: int | None = None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=NoOpTransformFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class OrigamiVlaDataConfig(DataConfigFactory):
    dataset_root: str = "D:/Sampled_Reprocessed_Dataset"
    manifest_root: str = "D:/Sampled_Reprocessed_Dataset/metadata/openpi_origami_vla/no_hmm_v1"
    prompt: str = "fold paper into airplane"
    local_target_npz_name: str = "local_delta_reached_state_targets_K15_include_current_restrict_true_state.npz"
    local_target_index_name: str = "local_delta_reached_state_targets_K15_include_current_restrict_true_state_index.parquet"
    action_source: Literal["spline", "action_chunk", "bspline_points"] = "spline"
    action_filename: str = "action_65d.npy"
    action_chunk_stride: int = 1
    drop_horizon_clipped: bool = False
    planner_arrays_filename: str = "planner_vla_rollout_features.npz"
    planner_index_filename: str = "planner_vla_rollout_index.parquet"
    planner_branch: str = "alias"
    planner_value_variant: Literal["final", "raw"] = "final"
    tactile_filename: str = "tactile_60d.npy"
    dataset_backend: Literal["video", "shard"] = "video"
    shard_root: str | None = None
    shard_manifest_name: str = "shard_manifest.json"
    shard_rows_name: str = "rows.parquet"
    shard_complete_marker_name: str = "complete.marker"
    shard_require_complete: bool = True
    shard_max_cached_shards: int = 2
    shard_use_stored_row_order: bool = True
    sample_weight_column: str = "sample_weight"
    require_sample_weight: bool = False
    image_source_type: Literal["video", "frame_cache"] = "video"
    frame_cache_root_relpath: str = "arrays/vla_frame_cache_224_uint8"
    fail_on_missing_modalities: bool = True
    max_rows: int | None = None
    image_modalities: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "ooi_rgb": "videos/ooi.mp4",
            "base_0_rgb": "videos/head_left.mp4",
            "left_wrist_0_rgb": "videos/wrist_left.mp4",
            "right_wrist_0_rgb": "videos/wrist_right.mp4",
        }
    )
    frame_cache_modalities: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "ooi_rgb": "ooi_rgb_224x224_uint8.npy",
            "base_0_rgb": "base_0_rgb_224x224_uint8.npy",
            "left_wrist_0_rgb": "left_wrist_0_rgb_224x224_uint8.npy",
            "right_wrist_0_rgb": "right_wrist_0_rgb_224x224_uint8.npy",
        }
    )
    load_tactile_images: bool = False
    tactile_deform_video: str = "videos/tactile_deform.mp4"
    tactile_raw_video: str = "videos/tactile_raw.mp4"
    tactile_require_raw_video: bool = False
    tactile_image_size: int = 224
    tactile_raw_input_dropout_prob: float = 0.0
    tactile_raw_dropout_seed: int = 1234
    tactile_image_input_dropout_prob: float = 0.0
    tactile_image_dropout_seed: int = 1234
    tactile_raw_grid: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "rows": 2,
            "cols": 5,
            "expected_width": 1600,
            "expected_height": 480,
            "resize_mode": "resize",
        }
    )
    tactile_deform_grid: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "rows": 2,
            "cols": 5,
            "expected_width": 1200,
            "expected_height": 480,
            "resize_mode": "resize",
        }
    )
    include_planner_features: bool = True
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=NoOpTransformFactory)
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)
    limit_loader_caches: bool = False
    max_cached_episodes: int = 8
    max_cached_videos: int = 32

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not isinstance(model_config, pi0_config.Pi0Config):
            raise TypeError("Origami VLA config currently expects a Pi0Config model.")
        if not model_config.origami_vla.enabled and self.action_source != "action_chunk":
            raise ValueError("Spline Origami VLA data config requires model.origami_vla.enabled=True.")

        settings = _origami_vla_dataset.OrigamiVlaSettings(
            dataset_root=self.dataset_root,
            manifest_root=self.manifest_root,
            local_target_npz_name=self.local_target_npz_name,
            local_target_index_name=self.local_target_index_name,
            action_source=self.action_source,
            action_filename=self.action_filename,
            action_chunk_stride=self.action_chunk_stride,
            drop_horizon_clipped=self.drop_horizon_clipped,
            planner_arrays_filename=self.planner_arrays_filename,
            planner_index_filename=self.planner_index_filename,
            planner_branch=self.planner_branch,
            planner_value_variant=self.planner_value_variant,
            planner_belief_dim=model_config.origami_vla.belief_dim,
            planner_history_dim=model_config.origami_vla.history_dim,
            tactile_filename=self.tactile_filename,
            dataset_backend=self.dataset_backend,
            shard_root=self.shard_root,
            shard_manifest_name=self.shard_manifest_name,
            shard_rows_name=self.shard_rows_name,
            shard_complete_marker_name=self.shard_complete_marker_name,
            shard_require_complete=self.shard_require_complete,
            shard_max_cached_shards=self.shard_max_cached_shards,
            shard_use_stored_row_order=self.shard_use_stored_row_order,
            max_control_points=model_config.origami_vla.max_control_points,
            action_horizon=model_config.action_horizon,
            max_span_count=model_config.origami_vla.max_span_count,
            degree=model_config.origami_vla.degree,
            spline_span_representation=model_config.origami_vla.spline_span_representation,
            state_dim=int(model_config.state_dim or model_config.action_dim),
            action_dim=model_config.action_dim,
            tactile_dim=model_config.origami_vla.tactile_dim,
            prompt=self.prompt,
            sample_weight_column=self.sample_weight_column,
            require_sample_weight=self.require_sample_weight,
            image_source_type=self.image_source_type,
            image_modalities=dict(self.image_modalities),
            frame_cache_root_relpath=self.frame_cache_root_relpath,
            frame_cache_modalities=dict(self.frame_cache_modalities),
            load_tactile_images=self.load_tactile_images or model_config.origami_vla.ftp_tactile_enabled,
            tactile_deform_video=self.tactile_deform_video,
            tactile_raw_video=self.tactile_raw_video,
            tactile_require_raw_video=self.tactile_require_raw_video,
            tactile_image_size=self.tactile_image_size,
            tactile_raw_input_dropout_prob=self.tactile_raw_input_dropout_prob,
            tactile_raw_dropout_seed=self.tactile_raw_dropout_seed,
            tactile_image_input_dropout_prob=self.tactile_image_input_dropout_prob,
            tactile_image_dropout_seed=self.tactile_image_dropout_seed,
            tactile_raw_grid=dict(self.tactile_raw_grid),
            tactile_deform_grid=dict(self.tactile_deform_grid),
            include_planner_features=self.include_planner_features,
            fail_on_missing_modalities=self.fail_on_missing_modalities,
            limit_loader_caches=self.limit_loader_caches,
            max_cached_episodes=self.max_cached_episodes,
            max_cached_videos=self.max_cached_videos,
            max_rows=self.max_rows,
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            origami_vla=settings,
            dataset_split="train",
        )


@dataclasses.dataclass(frozen=True)
class OrigamiCompActionChunkManifestBuildConfig:
    checkpoint_planner_manifest_root: str | None = None
    ignore_checkpoint_planner_split: bool = True
    num_val_episodes: int = 0
    val_seed: int = 1234
    val_episode_uids: tuple[str, ...] = ()
    frame_stride: int = 1
    keep_horizon_clipped: bool = False
    planner_export_root: str | None = None
    planner_assignment_mode: Literal["expand_view_modes", "episode_sampled"] = "episode_sampled"
    train_planner_view_modes: tuple[str, ...] = ("frame_stride_10", "fixed_7", "fixed_15", "random_mix")
    val_planner_view_modes: tuple[str, ...] = ("frame_stride_10", "random_mix")
    planner_value_variant: Literal["final", "raw"] = "final"
    planner_assignment_seed: int = 1234
    planner_dropout_episode_prob: float = 0.16
    # Phase-two robustness settings. They are opt-in: a zero perturbation
    # probability preserves the pre-existing manifest behavior exactly.
    planner_speed_stratified_assignment: bool = False
    planner_speed_stratification_bins: int = 10
    # Conditional probability among rows from planner-enabled episodes.
    planner_perturb_present_row_prob: float = 0.0
    planner_perturb_offset_probs: dict[int, float] = dataclasses.field(
        default_factory=lambda: {-2: 0.10, -1: 0.40, 1: 0.40, 2: 0.10}
    )
    # Donors always come from a non-transition progress segment. Negative
    # offsets use the tail of the preceding segment; positive offsets use the
    # head of the following segment.
    planner_perturb_previous_tail_fraction: float = 0.30
    planner_perturb_next_head_fraction: float = 0.30
    planner_perturb_previous_two_tail_fraction: float = 0.10
    planner_perturb_next_two_head_fraction: float = 0.10
    planner_perturb_seed: int = 5678
    planner_view_mode_probs: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "frame_stride_10": 0.5,
            "random_mix": 0.25,
            "fixed_7": 0.25,
            "fixed_15": 0.0,
        }
    )
    planner_branch_probs: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "posterior": 0.5,
            "prior": 0.5,
        }
    )
    planner_index_name: str = "planner_vla_rollout_index.parquet"
    planner_arrays_name: str = "planner_vla_rollout_features.npz"
    planner_complete_marker_name: str = "export_complete.marker"
    require_planner_complete_marker: bool = True
    allow_missing_planner_rows: bool = False
    speed_weighting: bool = True
    speed_label_relpaths: tuple[str, ...] = ("labels/checkpoints.json", "labels/transfer_checkpoints.json")
    speed_stats_split: Literal["train", "val", "all"] = "train"
    speed_semantic_group_size: int = 2
    speed_final_unpaired_policy: Literal["keep", "drop", "error"] = "keep"
    speed_done_policy: Literal["neutral", "weighted"] = "neutral"
    speed_alpha: float = 1.5
    speed_min_weight: float = 0.5
    speed_max_weight: float = 2.0
    speed_epsilon_frames: float = 1.0e-6
    speed_weight_val: bool = False
    train_index_name: str = "train_index.parquet"
    val_index_name: str = "val_index.parquet"


@dataclasses.dataclass(frozen=True)
class OrigamiCompActionChunkShardBuildConfig:
    shard_root: str | None = None
    split: Literal["train", "val", "all"] = "train"
    target_shard_bytes: str = "128GiB"
    target_num_shards: int | None = None
    max_episodes_per_shard: int | None = None
    num_workers: int = 8
    seed: int = 1234
    season_column: str = "source_season"
    row_order: Literal["episode_sequential", "shuffled_index"] = "shuffled_index"
    image_size: int = 224
    overwrite: bool = False
    skip_existing: bool = True
    require_manifest_verified: bool = False
    shard_manifest_name: str = "shard_manifest.json"
    shard_plan_name: str = "shard_plan.parquet"
    rows_name: str = "rows.parquet"
    metadata_name: str = "metadata.json"
    complete_marker_name: str = "complete.marker"
    max_shards_per_run: int | None = None
    progress_update_frames: int = 256
    progress_max_active_bars: int = 8
    progress_poll_seconds: float = 0.25
    progress_leave_active_bars: bool = False


@dataclasses.dataclass(frozen=True)
class OrigamiCompActionChunkDataConfig(OrigamiVlaDataConfig):
    action_source: Literal["spline", "action_chunk", "bspline_points"] = "action_chunk"
    local_target_npz_name: str = ""
    action_filename: str = "action_65d.npy"
    action_chunk_stride: int = 1
    drop_horizon_clipped: bool = True
    include_planner_features: bool = False
    dataset_backend: Literal["video", "shard"] = "video"
    shard_root: str | None = None
    shard_manifest_name: str = "shard_manifest.json"
    shard_rows_name: str = "rows.parquet"
    shard_complete_marker_name: str = "complete.marker"
    shard_require_complete: bool = True
    shard_max_cached_shards: int = 2
    shard_use_stored_row_order: bool = True
    manifest_build: OrigamiCompActionChunkManifestBuildConfig = dataclasses.field(
        default_factory=OrigamiCompActionChunkManifestBuildConfig
    )
    shard_build: OrigamiCompActionChunkShardBuildConfig = dataclasses.field(
        default_factory=OrigamiCompActionChunkShardBuildConfig
    )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLehomeDataConfig(DataConfigFactory):
    """
    Data config for LeHome datasets stored in LeRobot format.

    Expected dataset feature keys:
      - observation.images.top_rgb
      - observation.images.left_rgb
      - observation.images.right_rgb
      - observation.state
      - actions
      - task (used as prompt if prompt_from_task=True)
    """

    use_delta_joint_actions: bool = False
    action_dim: int = 12

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/top_rgb": "observation.images.top_rgb",
                        "observation/left_rgb": "observation.images.left_rgb",
                        "observation/right_rgb": "observation.images.right_rgb",
                        "observation/state": "observation.state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[lehome_policy.LehomeInputs(model_type=model_config.model_type)],
            outputs=[lehome_policy.LehomeOutputs(action_dim=self.action_dim)],
        )

        if self.use_delta_joint_actions:
            # For dual-arm SO101: (5 arm joints + 1 gripper) x 2
            # Keep grippers absolute and convert arm joints to delta.
            delta_action_mask = _transforms.make_bool_mask(5, -1, 5, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLehomeRobotSplineDataConfig(DataConfigFactory):
    """LeHome joint-state config conditioned on predicted robot spline sidecars and no image tokens."""

    use_delta_joint_actions: bool = True
    action_dim: int = 12
    forced_prompt: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/state": "observation.state",
                        "actions": "actions",
                        "prompt": "prompt",
                        "robot_spline_coefficients": "robot_spline/coefficients",
                        "robot_spline_knots": "robot_spline/knots",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[lehome_robot_spline_policy.LehomeRobotSplineInputs(model_type=model_config.model_type)],
            outputs=[lehome_robot_spline_policy.LehomeRobotSplineOutputs(action_dim=self.action_dim)],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(5, -1, 5, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)
        if self.forced_prompt is not None:
            model_transforms = _transforms.Group(
                inputs=[_transforms.SetPrompt(self.forced_prompt), *model_transforms.inputs],
                outputs=model_transforms.outputs,
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLehomeCameraCVDataConfig(DataConfigFactory):
    """
    LeHome data config variant that transforms observation.state (12D joints) into
    camera-frame EE pose features (16D) using:
      - fk_from_usd_common.json
      - top_camera_config_runtime_cv.json
    """

    use_delta_joint_actions: bool = False
    # Number of model action dims consumed from model output (camera-CV EE pose).
    action_dim: int = 16
    # Final action dims returned by policy after IK conversion.
    output_action_dim: int = 12
    state_unit: str = "rad"
    pose_quat_order: str = "wxyz"
    fk_json_path: str = str(lehome_camera_cv_policy._DEFAULT_FK_JSON_PATH)
    camera_config_json_path: str = str(lehome_camera_cv_policy._DEFAULT_CAMERA_CFG_JSON_PATH)
    dataset_joint_order_csv: str = (
        "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper"
    )
    forced_prompt: str | None = None
    # damping: float = 0.05
    damping: float = 0.05
    alpha: float = 1.0
    # line_search_alphas_csv: str = "1,0.5,0.25,0.05,1.5,2"
    line_search_alphas_csv: str = "1"
    fallback_tol_factor: float = 1.1
    pos_weight: float = 1.0
    rot_weight: float = 1.0
    # max_iters: int = 80
    max_iters: int = 20
    tol_pos_m: float = 1e-4
    tol_rot_deg: float = 0.2
    max_step_norm: float = 0.2
    enforce_limits: bool = True #True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        future_latent_repack = (
            {
                "future_latent_pred": "future_latent/pred",
                "future_latent_true": "future_latent/true",
                "future_latent_valid_mask": "future_latent/valid_mask",
            }
            if self.base_config is not None
            and self.base_config.future_latent_sidecar_root is not None
            else {}
        )
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/top_rgb": "observation.images.top_rgb",
                        "observation/left_rgb": "observation.images.left_rgb",
                        "observation/right_rgb": "observation.images.right_rgb",
                        "observation/state": "observation.state",
                        "actions": "actions",
                        "prompt": "prompt",
                        **future_latent_repack,
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                lehome_camera_cv_policy.LehomeCameraCVInputs(
                    model_type=model_config.model_type,
                    fk_json_path=self.fk_json_path,
                    camera_config_json_path=self.camera_config_json_path,
                    state_unit=self.state_unit,
                    pose_quat_order=self.pose_quat_order,
                    dataset_joint_order_csv=self.dataset_joint_order_csv,
                )
            ],
            outputs=[
                lehome_camera_cv_policy.LehomeCameraCVOutputs(
                    model_action_dim=self.action_dim,
                    output_action_dim=self.output_action_dim,
                    fk_json_path=self.fk_json_path,
                    camera_config_json_path=self.camera_config_json_path,
                    state_unit=self.state_unit,
                    pose_quat_order=self.pose_quat_order,
                    dataset_joint_order_csv=self.dataset_joint_order_csv,
                    damping=self.damping,
                    alpha=self.alpha,
                    line_search_alphas_csv=self.line_search_alphas_csv,
                    fallback_tol_factor=self.fallback_tol_factor,
                    pos_weight=self.pos_weight,
                    rot_weight=self.rot_weight,
                    max_iters=self.max_iters,
                    tol_pos_m=self.tol_pos_m,
                    tol_rot_deg=self.tol_rot_deg,
                    max_step_norm=self.max_step_norm,
                    enforce_limits=self.enforce_limits,
                )
            ],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(5, -1, 5, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)
        if self.forced_prompt is not None:
            model_transforms = _transforms.Group(
                inputs=[_transforms.SetPrompt(self.forced_prompt), *model_transforms.inputs],
                outputs=model_transforms.outputs,
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLehomeCameraCVActionChunkedDataConfig(DataConfigFactory):
    """
    LeHome camera-CV data config for LeRobot datasets that already store full action chunks
    per sample under `actions` with shape (T, 12).
    """

    use_delta_joint_actions: bool = False
    action_dim: int = 16
    output_action_dim: int = 12
    state_unit: str = "rad"
    pose_quat_order: str = "wxyz"
    fk_json_path: str = str(lehome_camera_cv_policy._DEFAULT_FK_JSON_PATH)
    camera_config_json_path: str = str(lehome_camera_cv_policy._DEFAULT_CAMERA_CFG_JSON_PATH)
    dataset_joint_order_csv: str = (
        "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper"
    )
    damping: float = 0.05
    alpha: float = 1.0
    line_search_alphas_csv: str = "1,0.5"
    fallback_tol_factor: float = 1.1
    pos_weight: float = 1.0
    rot_weight: float = 1.0
    max_iters: int = 20
    tol_pos_m: float = 1e-4
    tol_rot_deg: float = 0.2
    max_step_norm: float = 0.2
    enforce_limits: bool = True
    future_latent_sidecar_root: str | None = None
    future_latent_filter_included_datasets: bool = False
    future_latent_sidecar_required: bool = True
    use_sample_weights: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/top_rgb": "observation.images.top_rgb",
                        "observation/left_rgb": "observation.images.left_rgb",
                        "observation/right_rgb": "observation.images.right_rgb",
                        "observation/state": "observation.state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                lehome_camera_cv_policy.LehomeCameraCVInputs(
                    model_type=model_config.model_type,
                    fk_json_path=self.fk_json_path,
                    camera_config_json_path=self.camera_config_json_path,
                    state_unit=self.state_unit,
                    pose_quat_order=self.pose_quat_order,
                    dataset_joint_order_csv=self.dataset_joint_order_csv,
                )
            ],
            outputs=[
                lehome_camera_cv_policy.LehomeCameraCVOutputs(
                    model_action_dim=self.action_dim,
                    output_action_dim=self.output_action_dim,
                    fk_json_path=self.fk_json_path,
                    camera_config_json_path=self.camera_config_json_path,
                    state_unit=self.state_unit,
                    pose_quat_order=self.pose_quat_order,
                    dataset_joint_order_csv=self.dataset_joint_order_csv,
                    damping=self.damping,
                    alpha=self.alpha,
                    line_search_alphas_csv=self.line_search_alphas_csv,
                    fallback_tol_factor=self.fallback_tol_factor,
                    pos_weight=self.pos_weight,
                    rot_weight=self.rot_weight,
                    max_iters=self.max_iters,
                    tol_pos_m=self.tol_pos_m,
                    tol_rot_deg=self.tol_rot_deg,
                    max_step_norm=self.max_step_norm,
                    enforce_limits=self.enforce_limits,
                )
            ],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(5, -1, 5, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            prechunked_actions=True,
            action_sequence_keys=(),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLehomeCameraCVMultiCoTrainDataConfig(DataConfigFactory):
    repo_id: str = "local/lehome_camera_cv_multi_cotrain"
    dataset_specs: tuple[LehomeCameraCVDatasetSpec, ...] = (
        LehomeCameraCVDatasetSpec(repo_id="local/lehome_cotrain_dataset_a", sample_weight=1.0),
        LehomeCameraCVDatasetSpec(repo_id="local/lehome_cotrain_dataset_b", sample_weight=1.0),
        LehomeCameraCVDatasetSpec(repo_id="local/lehome_cotrain_dataset_c", sample_weight=1.0),
    )
    action_dim: int = 16
    output_action_dim: int = 12
    state_unit: str = "rad"
    pose_quat_order: str = "wxyz"
    inference_fk_json_path: str = str(lehome_camera_cv_policy._DEFAULT_FK_JSON_PATH)
    inference_camera_config_json_path: str = str(lehome_camera_cv_policy._DEFAULT_CAMERA_CFG_JSON_PATH)
    inference_dataset_joint_order_csv: str = (
        "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper"
    )
    inference_mode: Literal["real", "sim"] = "real"
    forced_prompt: str | None = None
    inference_target_image_height: int | None = None
    inference_target_image_width: int | None = None
    damping: float = 0.05
    alpha: float = 1.0
    line_search_alphas_csv: str = "1"
    fallback_tol_factor: float = 1.1
    pos_weight: float = 1.0
    rot_weight: float = 1.0
    max_iters: int = 20
    tol_pos_m: float = 1e-4
    tol_rot_deg: float = 0.2
    max_step_norm: float = 2.0
    enforce_limits: bool = True
    future_latent_sidecar_root: str | None = None
    future_latent_filter_included_datasets: bool = False
    future_latent_sidecar_required: bool = True
    use_sample_weights: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.dataset_specs:
            raise ValueError("LeRobotLehomeCameraCVMultiCoTrainDataConfig requires at least one dataset spec.")

        data_transforms = _transforms.Group(
            inputs=[
                lehome_camera_cv_policy.LehomeCameraCVInputs(
                    model_type=model_config.model_type,
                    fk_json_path=self.inference_fk_json_path,
                    camera_config_json_path=self.inference_camera_config_json_path,
                    state_unit=self.state_unit,
                    pose_quat_order=self.pose_quat_order,
                    dataset_joint_order_csv=self.inference_dataset_joint_order_csv,
                    inference_mode=self.inference_mode,
                    target_image_height=self.inference_target_image_height,
                    target_image_width=self.inference_target_image_width,
                )
            ],
            outputs=[
                lehome_camera_cv_policy.LehomeCameraCVOutputs(
                    model_action_dim=self.action_dim,
                    output_action_dim=self.output_action_dim,
                    fk_json_path=self.inference_fk_json_path,
                    camera_config_json_path=self.inference_camera_config_json_path,
                    state_unit=self.state_unit,
                    pose_quat_order=self.pose_quat_order,
                    dataset_joint_order_csv=self.inference_dataset_joint_order_csv,
                    damping=self.damping,
                    alpha=self.alpha,
                    line_search_alphas_csv=self.line_search_alphas_csv,
                    fallback_tol_factor=self.fallback_tol_factor,
                    pos_weight=self.pos_weight,
                    rot_weight=self.rot_weight,
                    max_iters=self.max_iters,
                    tol_pos_m=self.tol_pos_m,
                    tol_rot_deg=self.tol_rot_deg,
                    max_step_norm=self.max_step_norm,
                    enforce_limits=self.enforce_limits,
                )
            ],
        )
        base_config = self.create_base_config(assets_dirs, model_config)
        model_transforms = ModelTransformFactory()(model_config)
        if self.forced_prompt is not None:
            model_transforms = _transforms.Group(
                inputs=[_transforms.SetPrompt(self.forced_prompt), *model_transforms.inputs],
                outputs=model_transforms.outputs,
            )
        return dataclasses.replace(
            base_config,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
            multi_dataset_specs=self.dataset_specs,
            multi_state_unit=self.state_unit,
            multi_pose_quat_order=self.pose_quat_order,
            multi_action_dim=self.action_dim,
            multi_use_sample_weights=self.use_sample_weights,
            future_latent_sidecar_root=self.future_latent_sidecar_root,
            future_latent_filter_included_datasets=self.future_latent_filter_included_datasets,
            future_latent_sidecar_required=self.future_latent_sidecar_required,
            use_quantile_norm=True,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLehomePrecomputed16DDataConfig(DataConfigFactory):
    """
    LeHome pretraining config for datasets that already store 16D camera-CV state/action values
    and only provide the top camera on disk.
    """

    action_dim: int = 16
    gripper_dim_indices_csv: str = "7,15"
    fill_value: float = 0.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/top_rgb": "observation.images.top_rgb",
                        "observation/state": "observation.state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                lehome_camera_cv_policy.LehomePrecomputed16DInputs(
                    model_type=model_config.model_type,
                    state_dim=self.action_dim,
                    gripper_dim_indices_csv=self.gripper_dim_indices_csv,
                    fill_value=self.fill_value,
                )
            ]
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LehomePPOPolicyConfig:
    """Runtime config for frozen-VLA PPO correction heads.

    This is inference/rollout metadata only. The base VLA checkpoint is still
    loaded through the normal policy checkpoint path, while actor/value heads can
    be supplied separately by the policy server.
    """

    action_horizon: int = 10
    action_dim: int = 12
    latent_dim: int = 1024
    state_dim: int = 12
    token_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    correction_scale: float = 0.03
    delta_clip: float = 2.0
    log_std_init: float = -0.5
    min_log_std: float = -5.0
    max_log_std: float = 1.0
    deterministic: bool = False
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    delta_coef: float = 0.001
    base_checkpoint_path: str | None = None
    actor_head_path: str | None = None
    value_head_path: str | None = None


@dataclasses.dataclass(frozen=True)
class ParamLrMultiplier:
    regex: str
    multiplier: float

    def __post_init__(self) -> None:
        if not self.regex:
            raise ValueError("ParamLrMultiplier.regex must be non-empty.")
        if self.multiplier <= 0.0:
            raise ValueError(f"ParamLrMultiplier.multiplier must be positive, got {self.multiplier}.")


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99
    # Optional update scaling for future-latent adapter fine-tuning. When set, the configured
    # lr_schedule is treated as the adapter LR and all non-adapter trainable updates are multiplied
    # by this value. For example, adapter LR 5e-5 and VLA LR 5e-6 => 0.1.
    non_adapter_lr_multiplier: float | None = None
    adapter_param_regex: str = ".*future_latent_adapter.*"
    # Optional base-LR anchored update scaling. The configured lr_schedule remains
    # the base model LR, and matching parameter paths receive base_lr * multiplier.
    param_lr_multipliers: tuple[ParamLrMultiplier, ...] = ()

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # If true, run validation on a separate dataset during training.
    run_val: bool = False
    # Validation dataset repo id. Uses the same transforms and normalization assets as training.
    val_repo_id: str | None = None
    # Optional validation predicted-robot-spline sidecar root. If unset, the training sidecar root is reused.
    val_robot_spline_sidecar_root: str | None = None
    # How often (in steps) to run validation.
    val_frequency: int = 1000
    # Global validation batch size. If not provided, defaults to the training batch size.
    val_batch_size: int | None = None
    # Strategy used for saving checkpoints. "manual" preserves the current step-based checkpointing,
    # while "best_val" rewrites a single `best` checkpoint whenever validation loss improves.
    checkpoint_strategy: Literal["manual", "best_val"] = "manual"
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # Explicit completed training steps to save checkpoints at. When non-empty, this overrides save_interval.
    save_steps: tuple[int, ...] = ()
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000
    # Number of recent checkpoints to keep in addition to any keep_period-preserved checkpoints.
    max_to_keep: int | None = 1

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True
    # If true, logs a few first-batch camera images to wandb. Keep disabled for large/future-latent runs.
    log_first_batch_images: bool = False

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If set, serve this config as a frozen base policy with PPO correction/value
    # heads for rollout collection. The standard training path ignores this
    # field; training for these heads should use a separate PPO script.
    ppo_policy: LehomePPOPolicyConfig | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def best_checkpoint_dir(self) -> pathlib.Path:
        """Get the directory for the single best-validation checkpoint."""
        return (self.checkpoint_dir / "best").resolve()

    @property
    def latest_val_checkpoint_dir(self) -> pathlib.Path:
        """Get the directory for the most recent validation checkpoint."""
        return (self.checkpoint_dir / "latest_val").resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    @property
    def resolved_val_batch_size(self) -> int:
        return self.val_batch_size or self.batch_size

    def __post_init__(self) -> None:
        if (
            isinstance(self.model, pi0_config.Pi0Config)
            and self.model.origami_vla.enabled
            and self.model.origami_vla.action_norm_stats_dir is None
            and isinstance(self.data, OrigamiVlaDataConfig)
        ):
            asset_id = self.data.assets.asset_id or self.data.repo_id
            norm_stats_dir = (pathlib.Path(self.assets_base_dir) / self.name / str(asset_id)).resolve()
            object.__setattr__(
                self,
                "model",
                dataclasses.replace(
                    self.model,
                    origami_vla=dataclasses.replace(
                        self.model.origami_vla,
                        action_norm_stats_dir=str(norm_stats_dir),
                    ),
                ),
            )
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if self.run_val and not self.val_repo_id:
            raise ValueError("--val_repo_id must be set when --run_val is true.")
        if self.non_adapter_lr_multiplier is not None and self.param_lr_multipliers:
            raise ValueError("Use either non_adapter_lr_multiplier or param_lr_multipliers, not both.")
        if self.checkpoint_strategy == "best_val" and not self.run_val:
            raise ValueError("--run_val must be true when --checkpoint_strategy is best_val.")
        if self.val_frequency <= 0:
            raise ValueError("--val_frequency must be greater than 0.")
        if self.val_batch_size is not None and self.val_batch_size <= 0:
            raise ValueError("--val_batch_size must be greater than 0 when set.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning LeHome configs.
    #
    TrainConfig(
        # Example config for fine-tuning on a LeHome LeRobot dataset converted from episode JSON.
        name="pi05_lehome_robot_finetune",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        num_workers=32,
        run_val=True,
        val_repo_id="local/lehome_all_episodes",
        val_frequency=1000,
        val_batch_size=64,
        data=LeRobotLehomeDataConfig(
            # Replace with your local/HF LeRobot repo id.
            # repo_id="huggingaccounttest/lehome-openpi-episode",
            repo_id="local/lehome_all_episodes",
            base_config=DataConfig(prompt_from_task=True),
            # LeHome actions are typically joint-space absolute targets.
            use_delta_joint_actions=False,
            action_dim=12,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=64,
        log_interval = 50,
        save_interval = 20_000,
    ),
    TrainConfig(
        # LeHome variant: convert observation.state -> FK(world EE) -> camera_cv EE pose at policy input.
        name="pi05_lehome_camera_cv_robot_finetune",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        num_workers=32,
        run_val=False,
        checkpoint_strategy="manual",
        val_repo_id="local/lehome_val_episodes",
        val_frequency=1000,
        val_batch_size=400,
        data=LeRobotLehomeCameraCVDataConfig(
            # repo_id="local/lehome_all_top_garment",
            repo_id="local/lehome_all_top_garment",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            action_dim=16,
            output_action_dim=12,
            state_unit="rad",
            pose_quat_order="wxyz",
            fk_json_path=str(lehome_camera_cv_policy._DEFAULT_FK_JSON_PATH),
            camera_config_json_path=str(lehome_camera_cv_policy._DEFAULT_CAMERA_CFG_JSON_PATH),
            dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=220,
            peak_lr=1e-4,
            decay_steps=1800,
            decay_lr=5e-6,
        ),
        #/workspace/lehome-openpi/checkpoint_pretrain/params
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=1800,
        batch_size=400,
        log_interval=50,
        save_steps=(900,),
        keep_period=None,
        max_to_keep=4,
        # num_train_steps=1000,
        # batch_size=64,
        # log_interval=50,
        # save_interval=670,
    ),
    TrainConfig(
        name="pi05_lehome_camera_cv_multi_cotrain_robot_finetune",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=5, discrete_state_input=True),
        num_workers=32,
        run_val=False,
        checkpoint_strategy="manual",
        data=LeRobotLehomeCameraCVMultiCoTrainDataConfig(
            repo_id="local/lehome_camera_cv_multi_cotrain",
            base_config=DataConfig(prompt_from_task=False),
            dataset_specs=(
                LehomeCameraCVDatasetSpec(
                    repo_id="local/lehome_pretrain_all_garment_round2_data", #human_pretrain
                    sample_weight=1.6,
                    apply_camera_cv_transform=False,
                    fk_json_path=str(lehome_camera_cv_policy._DEFAULT_FK_JSON_PATH),
                    camera_config_json_path=str(lehome_camera_cv_policy._DEFAULT_CAMERA_CFG_JSON_PATH),
                    dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
                    valid_image_names_csv="base_0_rgb",
                    masked_state_indices_csv="7,15",
                    masked_action_indices_csv="2,3,4,5,6,7,10,11,12,13,14,15",
                    target_image_height=480,
                    target_image_width=640,
                ),
                LehomeCameraCVDatasetSpec(
                    repo_id="local/lehome_robot_sim_all_garment_round2_data", #robot_sim_dataset
                    sample_weight=4.0,
                    apply_camera_cv_transform=True,
                    fk_json_path=str(lehome_camera_cv_policy._POLICY_DATA_DIR / "sim_so101_fk_from_usd_common.json"),
                    camera_config_json_path=str(lehome_camera_cv_policy._POLICY_DATA_DIR / "sim_top_camera_config_runtime_cv.json"),
                    dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
                    valid_image_names_csv="base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb",
                    masked_state_indices_csv="",
                    masked_action_indices_csv="",
                    target_image_height=480,
                    target_image_width=640,
                ),
                LehomeCameraCVDatasetSpec(
                    repo_id="local/lehome_robot_real_all_garment_round2_data", #robot_real_dataset
                    sample_weight=8.0,
                    apply_camera_cv_transform=True,
                    fk_json_path=str(lehome_camera_cv_policy._POLICY_DATA_DIR / "real_so101_fk_from_usd_common.json"),
                    camera_config_json_path=str(lehome_camera_cv_policy._POLICY_DATA_DIR / "real_top_camera_config_runtime_cv.json"),
                    dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
                    valid_image_names_csv="base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb",
                    masked_state_indices_csv="",
                    masked_action_indices_csv="",
                    target_image_height=480,
                    target_image_width=640,
                ),
            ),
            inference_fk_json_path=str(
                pathlib.Path(__file__).resolve().parents[1]
                / "policies"
                / "lehome_camera_cv"
                / "real_so101_fk_from_usd_common.json"
            ),
            inference_camera_config_json_path=str(
                pathlib.Path(__file__).resolve().parents[1]
                / "policies"
                / "lehome_camera_cv"
                / "real_top_camera_config_runtime_cv.json"
            ),
            inference_dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
            inference_mode="real",
            forced_prompt="fold the garment on the table",
            inference_target_image_height=480,
            inference_target_image_width=640,
            action_dim=16,
            output_action_dim=12,
            state_unit="rad",
            pose_quat_order="wxyz",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=220,
            peak_lr=1e-4,
            decay_steps=1800,
            decay_lr=5e-6,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=1800,
        batch_size=400,
        log_interval=50,
        save_steps=(900,),
        keep_period=None,
        max_to_keep=4,
    ),
    TrainConfig(
        # Single-dataset LeHome robot fine-tune with future-latent conditioning enabled.
        name="pi05_lehome_camera_cv_robot_finetune_future_latent",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=True,
            future_latent=pi0_config.FutureLatentConfig(
                enabled=True,
                num_cameras=3,
                latent_tokens=24,
                latent_dim=512,
                adapter_hidden_dim=1024,
                output_dim=2048,
                predicted_latent_prob=0.50,
                true_latent_prob=0.30,
                dropped_latent_prob=0.20,
                sidecar_root="/ephemeral/sim_only_trained_future_latents_sim_round_config/vla_future_latent_sidecar_sim_only_v1",
                resampler_checkpoint_path="/scratch/future_latent_runs/resampler_autoencoder_v1/resampler_encoder_latest.pt",
                future_predictor_checkpoint_path="/scratch/future_latent_runs/future_predictor_v1/future_predictor_latest.pt",
                policy_camera_order=("top", "left_wrist", "right_wrist"),
                freeze_image_encoder=True,
                freeze_resampler=True,
                freeze_future_predictor=True,
            ),
        ),
        num_workers=32,
        run_val=False,
        checkpoint_strategy="manual",
        val_repo_id="local/lehome_val_episodes",
        val_frequency=1000,
        val_batch_size=400,
        data=LeRobotLehomeCameraCVDataConfig(
            repo_id="local/lehome_all_top_garment",
            base_config=DataConfig(
                prompt_from_task=True,
                future_latent_sidecar_root="/scratch/vla_future_latent_sidecar/output",
                future_latent_sidecar_required=True,
            ),
            use_delta_joint_actions=False,
            action_dim=16,
            output_action_dim=12,
            state_unit="rad",
            pose_quat_order="wxyz",
            fk_json_path=str(lehome_camera_cv_policy._POLICY_DATA_DIR / "sim_so101_fk_from_usd_common.json"),
            camera_config_json_path=str(
                lehome_camera_cv_policy._POLICY_DATA_DIR / "sim_top_camera_config_runtime_cv_no_flip.json"
            ),
            dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
            forced_prompt="fold the garment on the table",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=220,
            peak_lr=5e-5,
            decay_steps=2400,
            decay_lr=5e-6,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*(lora|future_latent_adapter).*",
        ),
        freeze_filter=nnx_utils.PathRegex("PaliGemma/img/.*"),
        non_adapter_lr_multiplier=0.1,
        adapter_param_regex=".*future_latent_adapter.*",
        num_train_steps=2400,
        batch_size=192,
        log_interval=50,
        save_steps=(900,),
        keep_period=None,
        max_to_keep=4,
    ),
    TrainConfig(
        name="pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=5,
            discrete_state_input=True,
            future_latent=pi0_config.FutureLatentConfig(
                enabled=True,
                num_cameras=3,
                latent_tokens=24,
                latent_dim=512,
                adapter_hidden_dim=1024,
                output_dim=2048,
                predicted_latent_prob=0.50,
                true_latent_prob=0.30,
                dropped_latent_prob=0.20,
                sidecar_root="/scratch/vla_future_latent_sidecar/output",
                resampler_checkpoint_path="/scratch/future_latent_runs/resampler_autoencoder_v1/resampler_encoder_latest.pt",
                future_predictor_checkpoint_path="/scratch/future_latent_runs/future_predictor_v1/future_predictor_latest.pt",
                policy_camera_order=("top", "left_wrist", "right_wrist"),
                freeze_image_encoder=True,
                freeze_resampler=True,
                freeze_future_predictor=True,
            ),
        ),
        num_workers=32,
        run_val=False,
        checkpoint_strategy="manual",
        data=LeRobotLehomeCameraCVMultiCoTrainDataConfig(
            repo_id="local/lehome_camera_cv_multi_cotrain",
            base_config=DataConfig(prompt_from_task=False),
            dataset_specs=(
                LehomeCameraCVDatasetSpec(
                    repo_id="local/lehome_pretrain_all_garment_round2_data", #human_pretrain
                    sample_weight=1.6,
                    include_in_future_latent_dataset=True,
                    apply_camera_cv_transform=False,
                    fk_json_path=str(lehome_camera_cv_policy._DEFAULT_FK_JSON_PATH),
                    camera_config_json_path=str(lehome_camera_cv_policy._DEFAULT_CAMERA_CFG_JSON_PATH),
                    dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
                    valid_image_names_csv="base_0_rgb",
                    masked_state_indices_csv="7,15",
                    masked_action_indices_csv="2,3,4,5,6,7,10,11,12,13,14,15",
                    target_image_height=480,
                    target_image_width=640,
                ),
                LehomeCameraCVDatasetSpec(
                    repo_id="local/lehome_robot_sim_all_garment_round2_data", #robot_sim_dataset
                    sample_weight=4.0,
                    include_in_future_latent_dataset=True,
                    apply_camera_cv_transform=True,
                    fk_json_path=str(lehome_camera_cv_policy._POLICY_DATA_DIR / "sim_so101_fk_from_usd_common.json"),
                    camera_config_json_path=str(lehome_camera_cv_policy._POLICY_DATA_DIR / "sim_top_camera_config_runtime_cv.json"),
                    dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
                    valid_image_names_csv="base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb",
                    masked_state_indices_csv="",
                    masked_action_indices_csv="",
                    target_image_height=480,
                    target_image_width=640,
                ),
                LehomeCameraCVDatasetSpec(
                    repo_id="local/lehome_robot_real_all_garment_round2_data", #robot_real_dataset
                    sample_weight=8.0,
                    include_in_future_latent_dataset=True,
                    apply_camera_cv_transform=True,
                    fk_json_path=str(lehome_camera_cv_policy._POLICY_DATA_DIR / "real_so101_fk_from_usd_common.json"),
                    camera_config_json_path=str(lehome_camera_cv_policy._POLICY_DATA_DIR / "real_top_camera_config_runtime_cv.json"),
                    dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
                    valid_image_names_csv="base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb",
                    masked_state_indices_csv="",
                    masked_action_indices_csv="",
                    target_image_height=480,
                    target_image_width=640,
                ),
            ),
            inference_fk_json_path=str(
                pathlib.Path(__file__).resolve().parents[1]
                / "policies"
                / "lehome_camera_cv"
                / "real_so101_fk_from_usd_common.json"
            ),
            inference_camera_config_json_path=str(
                pathlib.Path(__file__).resolve().parents[1]
                / "policies"
                / "lehome_camera_cv"
                / "real_top_camera_config_runtime_cv.json"
            ),
            inference_dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
            inference_mode="real",
            forced_prompt="fold the garment on the table",
            inference_target_image_height=480,
            inference_target_image_width=640,
            action_dim=16,
            output_action_dim=12,
            state_unit="rad",
            pose_quat_order="wxyz",
            future_latent_sidecar_root="/scratch/vla_future_latent_sidecar/output",
            future_latent_filter_included_datasets=True,
            future_latent_sidecar_required=True,
            use_sample_weights=True,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=220,
            peak_lr=5e-5,
            decay_steps=2400,
            decay_lr=5e-6,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/scratch/pretrain_multidata_cotrain_base_with_state_hum_sim_rob_3_epoch/params",
            missing_regex=".*(lora|future_latent_adapter).*",
        ),
        freeze_filter=nnx_utils.PathRegex("PaliGemma/img/.*"),
        non_adapter_lr_multiplier=0.1,
        adapter_param_regex=".*future_latent_adapter.*",
        num_train_steps=2400,
        batch_size=192,
        log_interval=50,
        save_steps=(900,),
        keep_period=None,
        max_to_keep=4,
    ),
    TrainConfig(
        # Spline-conditioned pi0.5 fine-tune: current joint state + predicted robot spline, no image prefix.
        name="pi05_lehome_robot_spline_joint_delta_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            # Keep the base pi0.5 action width so checkpoint restore stays shape-compatible.
            # LeHome's real 12D actions are handled by the data/output transforms and padded
            # to the model width via PadStatesAndActions.
            action_dim=32,
            discrete_state_input=True,
            robot_spline=pi0_config.RobotSplineConfig(
                enabled=True,
                use_image_prefix=False,
                control_count=13,
                degree=3,
                control_point_dim=2048,
                model_dim=512,
                num_layers=2,
                num_heads=8,
                ffn_dim=2048,
                width_fourier_bands=8,
                width_hidden_dim=512,
                rope_base=10000.0,
            ),
        ),
        num_workers=16,
        run_val=False,
        checkpoint_strategy="manual",
        data=LeRobotLehomeRobotSplineDataConfig(
            repo_id="E:/Lehome-Dataset/lehome_round_2_dataset/sim_dataset/robot_sim_ft_lehome_all_garment_data_z180",
            assets=AssetsConfig(asset_id="pi05_lehome_robot_spline_joint_delta_finetune"),
            base_config=DataConfig(
                prompt_from_task=True,
                robot_spline_sidecar_root=(
                    "E:/Lehome-Dataset/lehome_round_2_dataset/sim_dataset/"
                    "robot_sim_ft_lehome_all_garment_data_z180/embeddings/"
                    "robot_sim_multiview_vae_joint_full_visual_epoch/"
                    "predicted_robot_local_splines_default_run_n010"
                ),
                robot_spline_expand_pairings=True,
                robot_spline_sidecar_required=True,
            ),
            use_delta_joint_actions=True,
            action_dim=12,
            forced_prompt="fold the garment on the table",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2000,
            peak_lr=2e-4,
            decay_steps=50000,
            decay_lr=2e-6,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*(lora|robot_spline_adapter).*",
        ),
        freeze_filter=nnx_utils.PathRegex("PaliGemma/img/.*"),
        non_adapter_lr_multiplier=0.5,
        adapter_param_regex=".*robot_spline_adapter.*",
        num_train_steps=50000,
        batch_size=128,
        log_interval=100,
        save_interval=5000,
    ),
    TrainConfig(
        # Inference-only rollout config: load a trained pi0.5 LeHome camera-CV
        # checkpoint as a frozen base policy and add PPO correction/value heads.
        name="pi05_lehome_trained_vla_with_ppo_heads",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        num_workers=32,
        run_val=False,
        checkpoint_strategy="manual",
        data=LeRobotLehomeCameraCVDataConfig(
            repo_id="local/lehome_all_top_garment",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            action_dim=16,
            output_action_dim=12,
            state_unit="rad",
            pose_quat_order="wxyz",
            fk_json_path=str(lehome_camera_cv_policy._DEFAULT_FK_JSON_PATH),
            camera_config_json_path=str(lehome_camera_cv_policy._DEFAULT_CAMERA_CFG_JSON_PATH),
            dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        policy_metadata={
            "policy_family": "pi05_lehome_frozen_vla_ppo_heads",
            "base_config": "pi05_lehome_camera_cv_robot_finetune",
        },
        ppo_policy=LehomePPOPolicyConfig(
            action_horizon=10,
            action_dim=12,
            latent_dim=1024,
            state_dim=12,
            token_dim=256,
            num_layers=4,
            num_heads=8,
            correction_scale=0.03,
            delta_clip=2.0,
            log_std_init=-0.5,
            value_coef=0.5,
            entropy_coef=0.01,
            delta_coef=0.001,
            # Fill these in when serving without CLI path overrides.
            base_checkpoint_path=None,
            actor_head_path=None,
            value_head_path=None,
        ),
        num_train_steps=1,
        batch_size=1,
        log_interval=1,
        keep_period=None,
        max_to_keep=1,
    ),
    TrainConfig(
        # Inference-only rollout config: load the multi-cotrain future-latent VLA
        # checkpoint as a frozen base policy and add PPO correction/value heads.
        #
        # This is intentionally not a training config for the VLA. The only trainable
        # modules in the rollout stack are the optional PyTorch PPO heads, which are
        # loaded separately by the policy server.
        name="pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent_with_ppo_heads",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=True,
            future_latent=pi0_config.FutureLatentConfig(
                enabled=True,
                num_cameras=3,
                latent_tokens=24,
                latent_dim=512,
                adapter_hidden_dim=1024,
                output_dim=2048,
                predicted_latent_prob=0.60,
                true_latent_prob=0.30,
                dropped_latent_prob=0.10,
                sidecar_root="/ephemeral2/robot_future_latents_depedency_cotrain_final/vla_future_latent_sidecar_offset10_v2",
                resampler_checkpoint_path="/ephemeral2/robot_future_latents_depedency_cotrain_final/resampler_autoencoder_sim_real_run_v1/resampler_encoder_latest.pt",
                future_predictor_checkpoint_path="/ephemeral2/robot_future_latents_depedency_cotrain_final/robot_future_predictor_v2/future_predictor_latest.pt",
                policy_camera_order=("top", "left_wrist", "right_wrist"),
                freeze_image_encoder=True,
                freeze_resampler=True,
                freeze_future_predictor=True,
            ),
        ),
        num_workers=32,
        run_val=False,
        checkpoint_strategy="manual",
        data=LeRobotLehomeCameraCVMultiCoTrainDataConfig(
            repo_id="local/lehome_camera_cv_multi_cotrain",
            base_config=DataConfig(prompt_from_task=False),
            dataset_specs=(
                LehomeCameraCVDatasetSpec(
                    repo_id="local/lehome_pretrain_all_garment_round2_data",
                    sample_weight=1.6,
                    include_in_future_latent_dataset=False,
                    apply_camera_cv_transform=False,
                    fk_json_path=str(lehome_camera_cv_policy._DEFAULT_FK_JSON_PATH),
                    camera_config_json_path=str(lehome_camera_cv_policy._DEFAULT_CAMERA_CFG_JSON_PATH),
                    dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
                    valid_image_names_csv="base_0_rgb",
                    masked_state_indices_csv="7,15",
                    masked_action_indices_csv="2,3,4,5,6,7,10,11,12,13,14,15",
                    target_image_height=480,
                    target_image_width=640,
                ),
                LehomeCameraCVDatasetSpec(
                    repo_id="local/lehome_robot_sim_all_garment_round2_data",
                    sample_weight=0.15,
                    include_in_future_latent_dataset=True,
                    apply_camera_cv_transform=True,
                    fk_json_path=str(lehome_camera_cv_policy._POLICY_DATA_DIR / "sim_so101_fk_from_usd_common.json"),
                    camera_config_json_path=str(lehome_camera_cv_policy._POLICY_DATA_DIR / "sim_top_camera_config_runtime_cv.json"),
                    dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
                    valid_image_names_csv="base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb",
                    masked_state_indices_csv="",
                    masked_action_indices_csv="",
                    target_image_height=480,
                    target_image_width=640,
                ),
                LehomeCameraCVDatasetSpec(
                    repo_id="local/lehome_robot_real_all_garment_round2_data",
                    sample_weight=1.0,
                    include_in_future_latent_dataset=True,
                    apply_camera_cv_transform=True,
                    fk_json_path=str(lehome_camera_cv_policy._POLICY_DATA_DIR / "real_so101_fk_from_usd_common.json"),
                    camera_config_json_path=str(lehome_camera_cv_policy._POLICY_DATA_DIR / "real_top_camera_config_runtime_cv.json"),
                    dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
                    valid_image_names_csv="base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb",
                    masked_state_indices_csv="",
                    masked_action_indices_csv="",
                    target_image_height=480,
                    target_image_width=640,
                ),
            ),
            inference_fk_json_path=str(
                pathlib.Path(__file__).resolve().parents[1]
                / "policies"
                / "lehome_camera_cv"
                / "real_so101_fk_from_usd_common.json"
            ),
            inference_camera_config_json_path=str(
                pathlib.Path(__file__).resolve().parents[1]
                / "policies"
                / "lehome_camera_cv"
                / "real_top_camera_config_runtime_cv.json"
            ),
            inference_dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
            inference_mode="real",
            forced_prompt="fold the garment on the table",
            inference_target_image_height=480,
            inference_target_image_width=640,
            action_dim=16,
            output_action_dim=12,
            state_unit="rad",
            pose_quat_order="wxyz",
            future_latent_sidecar_root="/ephemeral2/robot_future_latents_depedency_cotrain_final/vla_future_latent_sidecar_offset10_v2",
            future_latent_filter_included_datasets=True,
            future_latent_sidecar_required=False,
            use_sample_weights=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/ephemeral2/pretrain_multidata_cotrain_base_with_state_hum_sim_rob_3_epoch/params",
            missing_regex=".*(lora|future_latent_adapter).*",
        ),
        freeze_filter=nnx_utils.PathRegex("PaliGemma/img/.*"),
        non_adapter_lr_multiplier=0.1,
        adapter_param_regex=".*future_latent_adapter.*",
        policy_metadata={
            "policy_family": "pi05_lehome_multi_cotrain_future_latent_frozen_vla_ppo_heads",
            "base_config": "pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent",
            "inference_only": True,
        },
        ppo_policy=LehomePPOPolicyConfig(
            action_horizon=10,
            action_dim=12,
            latent_dim=1024,
            state_dim=12,
            token_dim=256,
            num_layers=4,
            num_heads=8,
            correction_scale=0.02,
            delta_clip=2.0,
            log_std_init=-0.5,
            value_coef=1.0,
            entropy_coef=0.002,
            delta_coef=0.05,
            # Supply --policy.dir or OPENPI_CHECKPOINT_DIR when serving.
            base_checkpoint_path=None,
            # Leave these unset for rollout one to collect with freshly initialized
            # PPO heads. Set them for later rollouts when continuing from trained heads.
            actor_head_path=None,
            value_head_path=None,
        ),
        num_train_steps=1,
        batch_size=1,
        log_interval=1,
        keep_period=None,
        max_to_keep=1,
    ),
    TrainConfig(
        # LeHome pretraining config for datasets that already store 16D camera-CV state/action values
        # and only provide top-camera images. Wrist views are masked absent and gripper dims are
        # excluded from stats and action loss. Gripper state dims are tokenized as explicit missing
        # slots when discrete_state_input is enabled.
        name="pi05_lehome_precomputed_16d_pretrain",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=True),
        num_workers=32,
        run_val=False,
        data=LeRobotLehomePrecomputed16DDataConfig(
            repo_id="local/lehome_precomputed_16d_pretrain",
            base_config=DataConfig(prompt_from_task=True),
            action_dim=16,
            gripper_dim_indices_csv="7,15",
            fill_value=0.0,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2000,
            peak_lr=1e-4,
            decay_steps=50000,
            decay_lr=5e-6,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50000,
        batch_size=256,
        log_interval=100,
        save_interval=10000,
    ),
    TrainConfig(
        # LeHome camera-CV variant for datasets that already store full action chunks per datapoint.
        name="pi05_lehome_camera_cv_action_chunked_robot_finetune",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        num_workers=32,
        run_val=True,
        val_repo_id="local/lehome_action_chunked_sample",
        val_frequency=1000,
        val_batch_size=64,
        data=LeRobotLehomeCameraCVActionChunkedDataConfig(
            repo_id="local/lehome_action_chunked_sample",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            action_dim=16,
            output_action_dim=12,
            state_unit="rad",
            pose_quat_order="wxyz",
            dataset_joint_order_csv="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=4200,
            peak_lr=1e-4,
            decay_steps=63000,
            decay_lr=5e-6,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=63000,
        batch_size=64,
        log_interval=200,
        save_steps=(8400, 20000, 40000),
        keep_period=None,
        max_to_keep=4,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_origami_checkpoint_spline_vla",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=65,
            state_dim=65,
            action_horizon=19,
            max_token_len=448,
            discrete_state_input=True,
            image_keys=("ooi_rgb", "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
            origami_vla=pi0_config.OrigamiVlaConfig(
                enabled=True,
                episode_execution_speed_preference=True,
                belief_dim=39,
                history_dim=512,
                planner_belief_hidden_dims=(256, 512),
                planner_progress_hidden_dims=(128, 512),
                planner_uncertainty_hidden_dims=(128, 512),
                planner_history_hidden_dims=(1024,),
                planner_use_type_embeddings=True,
                degree=3,
                max_control_points=18,
                max_span_count=15,
                curve_sample_count=120,
                smooth_l1_beta=0.05,
                curve_loss_weight=0.5,
                start_loss_weight=0.1,
                end_loss_weight=0.25,
                width_loss_weight=0.05,
                width_min=1e-4,
                use_quantile_norm=True,
                tactile_enabled=True,
                tactile_dim=60,
                tactile_finger_count=10,
                tactile_channels_per_finger=6,
                tactile_token_dim=256,
                tactile_finger_hidden_dims=(64, 128),
                tactile_transformer_layers=2,
                tactile_attention_heads=4,
                tactile_ffn_dim=512,
                tactile_use_type_embeddings=True,
                tactile_quantile_low=0.005,
                tactile_quantile_high=0.995,
                tactile_min_scale=1.0e-6,
                tactile_soft_clip_scale=5.0,
                use_speed_efficiency_weight=True,
                normalize_speed_efficiency_weighted_loss=True,
                speed_efficiency_weight_eps=1.0e-6,
            ),
        ),
        data=OrigamiVlaDataConfig(
            repo_id="local/origami_sampled_vla",
            assets=AssetsConfig(asset_id="sampled_reprocessed_dataset_origami_vla"),
            dataset_root="D:/Sampled_Reprocessed_Dataset",
            manifest_root="D:/Sampled_Reprocessed_Dataset/metadata/openpi_origami_vla/no_hmm_v1",
            prompt="fold paper into airplane",
            tactile_filename="tactile_60d.npy",
            sample_weight_column="sample_weight",
            require_sample_weight=True,
            image_source_type="frame_cache",
            frame_cache_root_relpath="arrays/vla_frame_cache_224_uint8",
            frame_cache_modalities={
                "ooi_rgb": "ooi_rgb_224x224_uint8.npy",
                "base_0_rgb": "base_0_rgb_224x224_uint8.npy",
                "left_wrist_0_rgb": "left_wrist_0_rgb_224x224_uint8.npy",
                "right_wrist_0_rgb": "right_wrist_0_rgb_224x224_uint8.npy",
            },
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*(lora|origami_planner_adapter|origami_tactile_adapter|action_in_proj|action_out_proj).*",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=2e-4,
            decay_steps=60_000,
            decay_lr=2e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        non_adapter_lr_multiplier=0.5,
        adapter_param_regex=".*(origami_planner_adapter|origami_tactile_adapter|action_in_proj|action_out_proj).*",
        batch_size=32,
        num_workers=8,
        num_train_steps=60_000,
        log_interval=100,
        run_val=True,
        val_repo_id="local/origami_sampled_vla",
        val_frequency=2_000,
        val_batch_size=32,
        checkpoint_strategy="manual",
        save_interval=5_000,
        keep_period=10_000,
        max_to_keep=3,
    ),
    TrainConfig(
        name="pi05_origami_comp_action_chunk",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=65,
            state_dim=65,
            action_horizon=25,
            # Re-measure with scripts/scan_origami_comp_prompt_token_lengths.py after norm stats are computed.
            max_token_len=768,
            discrete_state_input=True,
            image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
            origami_vla=pi0_config.OrigamiVlaConfig(
                enabled=True,
                action_mode="action_chunk",
                disable_auxiliary_losses=True,
                episode_execution_speed_preference=True,
                use_speed_efficiency_weight=True,
                normalize_speed_efficiency_weighted_loss=True,
                speed_efficiency_weight_eps=1.0e-6,
                belief_dim=29,
                history_dim=512,
                tactile_dim=60,
                tactile_prompt_input=True,
                prompt_discrete_clip=True,
                tactile_enabled=False,
                ftp_tactile_enabled=True,
                ftp_tactile_branch="auto",
                ftp_tactile_image_size=224,
                ftp_tactile_patch_size=16,
                ftp_tactile_width=768,
                ftp_tactile_depth=3,
                ftp_tactile_heads=12,
                ftp_tactile_mlp_ratio=4,
                ftp_tactile_backbone_micro_batch=64,
                ftp_tactile_freeze_backbone=True,
                ftp_tactile_include_cls_token=True,
                ftp_tactile_normalize_images=True,
                ftp_tactile_adapter_dim=512,
                ftp_tactile_prefix_dim=2048,
                ftp_tactile_tokens_per_finger=4,
                ftp_tactile_hands=2,
                ftp_tactile_fingers_per_hand=5,
                ftp_tactile_resampler_layers=2,
                ftp_tactile_resampler_heads=8,
                ftp_tactile_resampler_ffn_dim=2048,
                ftp_tactile_cross_finger_layers=4,
                ftp_tactile_cross_finger_heads=8,
                ftp_tactile_cross_finger_ffn_dim=2048,
                ftp_tactile_dropout=0.1,
                ftp_tactile_rope_enabled=True,
            ),
        ),
        data=OrigamiCompActionChunkDataConfig(
            repo_id="local/origami_comp_action_chunk",
            assets=AssetsConfig(asset_id="competition_paper_reprocessed_origami_comp_action_chunk"),
            dataset_root="E:/Robot-Origami-Challenge/Competition_Paper_Reprocessed_Dataset",
            manifest_root=(
                "E:/Robot-Origami-Challenge/Competition_Paper_Reprocessed_Dataset/"
                "metadata/openpi_origami_comp_action_chunk/no_hmm_224_headleft_tactile_prompt_planner"
            ),
            prompt="fold paper into airplane",
            planner_branch="posterior",
            planner_value_variant="final",
            tactile_filename="tactile_60d.npy",
            dataset_backend="video",
            shard_root=(
                "E:/Robot-Origami-Challenge/Competition_Paper_Reprocessed_Dataset/"
                "metadata/openpi_origami_comp_action_chunk_shards/no_hmm_224_headleft_tactile_prompt_planner"
            ),
            shard_manifest_name="shard_manifest.json",
            shard_rows_name="rows.parquet",
            shard_complete_marker_name="complete.marker",
            shard_require_complete=True,
            shard_max_cached_shards=2,
            shard_use_stored_row_order=True,
            load_tactile_images=True,
            tactile_deform_video="videos/tactile_deform.mp4",
            tactile_raw_video="videos/tactile_raw.mp4",
            tactile_require_raw_video=False,
            tactile_image_size=224,
            tactile_raw_input_dropout_prob=0.5,
            tactile_raw_dropout_seed=1234,
            sample_weight_column="sample_weight",
            require_sample_weight=True,
            image_source_type="video",
            image_modalities={
                "base_0_rgb": "videos/head_left.mp4",
                "left_wrist_0_rgb": "videos/wrist_left.mp4",
                "right_wrist_0_rgb": "videos/wrist_right.mp4",
            },
            frame_cache_modalities={
                "base_0_rgb": "base_0_rgb_224x224_uint8.npy",
                "left_wrist_0_rgb": "left_wrist_0_rgb_224x224_uint8.npy",
                "right_wrist_0_rgb": "right_wrist_0_rgb_224x224_uint8.npy",
            },
            include_planner_features=True,
            limit_loader_caches=True,
            max_cached_episodes=4,
            max_cached_videos=24,
            manifest_build=OrigamiCompActionChunkManifestBuildConfig(
                checkpoint_planner_manifest_root=None,
                ignore_checkpoint_planner_split=True,
                num_val_episodes=0,
                val_seed=1234,
                val_episode_uids=(),
                frame_stride=1,
                keep_horizon_clipped=False,
                planner_export_root=(
                    "E:/Robot-Origami-Challenge/Competition_Paper_Reprocessed_Dataset/"
                    "metadata/checkpoint_planner_vla_rollout_exports/"
                    "no_hmm_224_headleft_tactile_distill_prior__gamma10_future15_thr065_recomputed"
                ),
                planner_assignment_mode="episode_sampled",
                train_planner_view_modes=("frame_stride_10", "fixed_7", "fixed_15", "random_mix"),
                val_planner_view_modes=("frame_stride_10", "random_mix"),
                planner_value_variant="final",
                planner_assignment_seed=1234,
                planner_dropout_episode_prob=1.00,
                planner_view_mode_probs={
                    "frame_stride_10": 0.5,
                    "random_mix": 0.25,
                    "fixed_7": 0.25,
                    "fixed_15": 0.0,
                },
                planner_branch_probs={
                    "posterior": 0.5,
                    "prior": 0.5,
                },
                planner_index_name="planner_vla_rollout_index.parquet",
                planner_arrays_name="planner_vla_rollout_features.npz",
                planner_complete_marker_name="export_complete.marker",
                require_planner_complete_marker=True,
                allow_missing_planner_rows=False,
                speed_weighting=True,
                speed_label_relpaths=("labels/checkpoints.json", "labels/transfer_checkpoints.json"),
                speed_stats_split="train",
                speed_semantic_group_size=2,
                speed_final_unpaired_policy="keep",
                speed_done_policy="neutral",
                speed_alpha=2.2,
                speed_min_weight=0.7,
                speed_max_weight=1.3,
                speed_epsilon_frames=1.0e-6,
                speed_weight_val=False,
                train_index_name="train_index.parquet",
                val_index_name="val_index.parquet",
            ),
            shard_build=OrigamiCompActionChunkShardBuildConfig(
                shard_root=(
                    "E:/Robot-Origami-Challenge/Competition_Paper_Reprocessed_Dataset/"
                    "metadata/openpi_origami_comp_action_chunk_shards/no_hmm_224_headleft_tactile_prompt_planner"
                ),
                split="train",
                target_shard_bytes="128GiB",
                target_num_shards=None,
                max_episodes_per_shard=None,
                num_workers=8,
                seed=1234,
                season_column="source_season",
                row_order="shuffled_index",
                image_size=224,
                overwrite=False,
                skip_existing=True,
                require_manifest_verified=False,
                shard_manifest_name="shard_manifest.json",
                shard_plan_name="shard_plan.parquet",
                rows_name="rows.parquet",
                metadata_name="metadata.json",
                complete_marker_name="complete.marker",
                max_shards_per_run=None,
                progress_update_frames=256,
                progress_max_active_bars=8,
                progress_poll_seconds=0.25,
                progress_leave_active_bars=False,
            ),
            data_transforms=lambda model: _transforms.Group(
                inputs=[_transforms.DeltaActions(_transforms.make_bool_mask(65))],
                outputs=[_transforms.AbsoluteActions(_transforms.make_bool_mask(65))],
            ),
        ),
        weight_loader=weight_loaders.CompositeWeightLoader(
            loaders=(
                weight_loaders.CheckpointWeightLoader(
                    "gs://openpi-assets/checkpoints/pi05_base/params",
                    missing_regex=(
                        ".*(lora|origami_planner_adapter|origami_ftp_tactile_prefix_encoder|"
                        "action_in_proj|action_out_proj).*"
                    ),
                ),
                weight_loaders.NpzSubsetWeightLoader(
                    (
                        "E:/Robot-Origami-Challenge/Competition_Paper_Reprocessed_Dataset/"
                        "metadata/openpi_adapter_pretraining/no_hmm_224_headleft_tactile_distill/"
                        "openpi_origami_planner_adapter_prefixed_params.npz"
                    ),
                    strict=True,
                ),
                weight_loaders.OrbaxSubsetWeightLoader(
                    (
                        "E:/Robot-Origami-Challenge/Competition_Paper_Reprocessed_Dataset/"
                        "metadata/ftp_tactile_prefix_encoder_runs/sharpawave_40x2048_jax/params"
                    ),
                    key_prefix="origami_ftp_tactile_prefix_encoder",
                    strict=True,
                ),
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=700,
            peak_lr=8e-5,
            decay_steps=87_500,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=nnx_utils.PathRegex(".*origami_ftp_tactile_prefix_encoder/backbone/.*"),
        param_lr_multipliers=(
            ParamLrMultiplier(regex=".*origami_planner_adapter.*", multiplier=3.0),
            ParamLrMultiplier(regex=".*origami_ftp_tactile_prefix_encoder.*", multiplier=1.5),
            ParamLrMultiplier(regex=".*origami_ftp_tactile_prefix_encoder/prefix_projection.*", multiplier=3.0),
        ),
        batch_size=32,
        num_workers=16,
        num_train_steps=87_500, #140_000,
        log_interval=50,
        run_val=False,
        val_repo_id=None,
        val_frequency=2_000,
        val_batch_size=32,
        checkpoint_strategy="manual",
        save_steps=(20_000, 40_000, 60_000),
        # save_interval=5_000,
        # keep_period=10_000,
        max_to_keep=20,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

_ORIGAMI_COMP_ACTION_CHUNK_CONFIG = next(
    config for config in _CONFIGS if config.name == "pi05_origami_comp_action_chunk"
)

_ORIGAMI_COMP_ACTION_CHUNK_PHASE2_MANIFEST_ROOT = (
    "E:/Robot-Origami-Challenge/Competition_Paper_Reprocessed_Dataset/"
    "metadata/openpi_origami_comp_action_chunk/no_hmm_224_headleft_tactile_prompt_planner_phase2_f25_f30_raw"
)
_ORIGAMI_COMP_ACTION_CHUNK_PHASE2_SHARD_ROOT = (
    "E:/Robot-Origami-Challenge/Competition_Paper_Reprocessed_Dataset/"
    "metadata/openpi_origami_comp_action_chunk_shards/no_hmm_224_headleft_tactile_prompt_planner_phase2_f25_f30_raw"
)
_ORIGAMI_COMP_ACTION_CHUNK_PHASE2_EXPORT_ROOT = (
    "E:/Robot-Origami-Challenge/Competition_Paper_Reprocessed_Dataset/"
    "metadata/checkpoint_planner_vla_rollout_exports/"
    "no_hmm_224_headleft_tactile_distill_unseen_F25_F30_F50"
)

_CONFIGS.append(
    dataclasses.replace(
        _ORIGAMI_COMP_ACTION_CHUNK_CONFIG,
        name="pi05_origami_comp_action_chunk_phase2",
        # Phase 2 owns these action-target settings.  They are deliberately
        # independent from Phase 1: edit the two literals here, then rebuild
        # the Phase 2 manifest and shards.
        model=dataclasses.replace(
            _ORIGAMI_COMP_ACTION_CHUNK_CONFIG.model,
            action_horizon=25,
        ),
        data=dataclasses.replace(
            _ORIGAMI_COMP_ACTION_CHUNK_CONFIG.data,
            manifest_root=_ORIGAMI_COMP_ACTION_CHUNK_PHASE2_MANIFEST_ROOT,
            # Phase 2 consumes rebuilt immutable shards, not the Phase 1 rows.
            dataset_backend="shard",
            shard_root=_ORIGAMI_COMP_ACTION_CHUNK_PHASE2_SHARD_ROOT,
            planner_branch="posterior",
            planner_value_variant="raw",
            action_chunk_stride=1,
            manifest_build=dataclasses.replace(
                _ORIGAMI_COMP_ACTION_CHUNK_CONFIG.data.manifest_build,
                planner_export_root=_ORIGAMI_COMP_ACTION_CHUNK_PHASE2_EXPORT_ROOT,
                train_planner_view_modes=("frame_stride_25", "frame_stride_30", "random_mix"),
                val_planner_view_modes=("frame_stride_25", "frame_stride_30", "random_mix"),
                planner_value_variant="raw",
                planner_dropout_episode_prob=0.45,
                planner_speed_stratified_assignment=True,
                planner_speed_stratification_bins=10,
                planner_perturb_present_row_prob=0.27,
                planner_perturb_offset_probs={-2: 0.05, -1: 0.30, 1: 0.50, 2: 0.15},
                planner_perturb_previous_tail_fraction=0.30,
                planner_perturb_next_head_fraction=0.30,
                planner_perturb_previous_two_tail_fraction=0.10,
                planner_perturb_next_two_head_fraction=0.10,
                planner_perturb_seed=5678,
                planner_view_mode_probs={
                    "frame_stride_25": 1.0 / 3.0,
                    "frame_stride_30": 1.0 / 3.0,
                    "random_mix": 1.0 / 3.0,
                },
                planner_branch_probs={"posterior": 0.5, "prior": 0.5},
            ),
            shard_build=dataclasses.replace(
                _ORIGAMI_COMP_ACTION_CHUNK_CONFIG.data.shard_build,
                shard_root=_ORIGAMI_COMP_ACTION_CHUNK_PHASE2_SHARD_ROOT,
                overwrite=False,
                skip_existing=True,
            ),
        ),
        # Phase 2 must restore the full Phase 1 model, including the action,
        # planner, and tactile-adapter parameters.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            # Replace the experiment and step components with the completed
            # Phase 1 run. This must be the checkpoint's `params` directory.
            "E:/Robot-Origami-Challenge/openpi/checkpoints/pi05_origami_comp_action_chunk/"
            "REPLACE_WITH_PHASE1_EXPERIMENT/REPLACE_WITH_PHASE1_STEP/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=1e-5,
            decay_steps=60_000,
            decay_lr=1e-6,
        ),
        # Intentionally explicit Phase 2 controls. Adjust these here without
        # changing the original Phase 1 recipe.
        batch_size=32,
        num_workers=8,
        num_train_steps=60_000,
        checkpoint_strategy="manual",
        save_steps=(1_000, 2_000, 3_000, 5_000, 10_000, 20_000, 40_000, 60_000),
        max_to_keep=10,
    )
)

_ORIGAMI_COMP_ACTION_SPLINE_MANIFEST_ROOT = (
    "E:/Robot-Origami-Challenge/Competition_Paper_Reprocessed_Dataset/"
    "metadata/openpi_origami_comp_action_spline/no_hmm_224_headleft_tactile_prompt_planner"
)
_ORIGAMI_COMP_ACTION_SPLINE_SHARD_ROOT = (
    "E:/Robot-Origami-Challenge/Competition_Paper_Reprocessed_Dataset/"
    "metadata/openpi_origami_comp_action_spline_shards/no_hmm_224_headleft_tactile_prompt_planner"
)
_CONFIGS.append(
    dataclasses.replace(
        _ORIGAMI_COMP_ACTION_CHUNK_CONFIG,
        name="pi05_origami_comp_action_spline",
        model=dataclasses.replace(
            _ORIGAMI_COMP_ACTION_CHUNK_CONFIG.model,
            action_horizon=14,
            origami_vla=dataclasses.replace(
                _ORIGAMI_COMP_ACTION_CHUNK_CONFIG.model.origami_vla,
                action_mode="spline",
                max_control_points=13,
                max_span_count=10,
                degree=3,
                spline_span_representation="logits",
                compute_auxiliary_metrics=True,
                backprop_auxiliary_losses=True,
                curve_loss_weight=0.05,
                start_loss_weight=0.01,
                end_loss_weight=0.01,
                width_loss_weight=0.01,
                curve_fm_enabled=True,
                curve_fm_backprop=False,
                curve_fm_loss_weight=0.0,
                curve_fm_sample_intervals=10,
                curve_fm_include_endpoints=True,
                curve_fm_width_min=1e-4,
                curve_fm_denominator_eps=1e-6,
                curve_fm_softmax_clip=30.0,
                curve_fm_loss_clip=None,
                curve_fm_separate_velocity_metrics=False,
                mask_action_noise=True,
                action_norm_stats_dir=None,
                disable_auxiliary_losses=False,
            ),
        ),
        data=dataclasses.replace(
            _ORIGAMI_COMP_ACTION_CHUNK_CONFIG.data,
            repo_id="local/origami_comp_action_spline",
            assets=AssetsConfig(asset_id="competition_paper_reprocessed_origami_comp_action_spline"),
            manifest_root=_ORIGAMI_COMP_ACTION_SPLINE_MANIFEST_ROOT,
            dataset_backend="shard",
            shard_root=_ORIGAMI_COMP_ACTION_SPLINE_SHARD_ROOT,
            action_source="spline",
            local_target_npz_name="local_delta_action_cubic_knotspans10.npz",
            local_target_index_name="local_delta_action_cubic_knotspans10_index.parquet",
            action_filename="",
            action_chunk_stride=1,
            drop_horizon_clipped=True,
            tactile_image_input_dropout_prob=0.0,
            tactile_image_dropout_seed=4321,
            shard_build=dataclasses.replace(
                _ORIGAMI_COMP_ACTION_CHUNK_CONFIG.data.shard_build,
                shard_root=_ORIGAMI_COMP_ACTION_SPLINE_SHARD_ROOT,
            ),
            data_transforms=NoOpTransformFactory(),
        ),
    )
)

_CONFIGS.append(
    dataclasses.replace(
        _ORIGAMI_COMP_ACTION_CHUNK_CONFIG,
        name="pi05_origami_comp_action_bspline_points",
        model=dataclasses.replace(
            _ORIGAMI_COMP_ACTION_CHUNK_CONFIG.model,
            action_horizon=18,
            origami_vla=dataclasses.replace(
                _ORIGAMI_COMP_ACTION_CHUNK_CONFIG.model.origami_vla,
                action_mode="bspline_points",
                max_control_points=17,
                max_span_count=10,
                degree=3,
                bspline_control_point_count=13,
                bspline_point_count=17,
                bspline_width_logit_count=10,
                bspline_curve_sample_intervals=120,
                bspline_softmax_clip=30.0,
                bspline_denominator_eps=1.0e-6,
                bspline_aux_metrics_enabled=True,
                compute_auxiliary_metrics=True,
                backprop_auxiliary_losses=False,
                curve_loss_weight=1.0,
                start_loss_weight=1.0,
                end_loss_weight=1.0,
                width_loss_weight=1.0,
                curve_fm_enabled=False,
                curve_fm_backprop=False,
                curve_fm_loss_weight=0.0,
                mask_action_noise=True,
                action_norm_stats_dir=None,
                disable_auxiliary_losses=False,
            ),
        ),
        data=dataclasses.replace(
            _ORIGAMI_COMP_ACTION_CHUNK_CONFIG.data,
            repo_id="local/origami_comp_action_bspline_points",
            assets=AssetsConfig(asset_id="competition_paper_reprocessed_origami_comp_action_bspline_points"),
            manifest_root=_ORIGAMI_COMP_ACTION_SPLINE_MANIFEST_ROOT,
            dataset_backend="video",
            shard_root=None,
            action_source="bspline_points",
            local_target_npz_name="local_delta_action_bspline_points_knotspans10.npz",
            local_target_index_name="local_delta_action_bspline_points_knotspans10_index.parquet",
            action_filename="",
            action_chunk_stride=1,
            drop_horizon_clipped=True,
            data_transforms=NoOpTransformFactory(),
        ),
    )
)

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
