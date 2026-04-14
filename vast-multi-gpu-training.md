wand token: 86db3e27e1bed5224ebac6150d2b59c668eb8f76

git clone --recurse-submodules https://github.com/Aarvis/lehome-openpi.git

git submodule update --init --recursive

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .


Upload dataset to Hf
hf upload huggingaccounttest/lehome_train_episodes \
  "/datadrive/hf_cache/lerobot/local/lehome_train_episodes" \
  . \
  --repo-type dataset && \
hf repos settings huggingaccounttest/lehome_train_episodes \
  --repo-type dataset \
  --gated manual




hf upload huggingaccounttest/lehome_val_episodes \
  "/datadrive/hf_cache/lerobot/local/lehome_val_episodes" \
  . \
  --repo-type dataset && \
hf repos settings huggingaccounttest/lehome_val_episodes \
  --repo-type dataset \
  --gated manual

hf upload huggingaccounttest/robot_only_finetune_with_state_1_epoch \
  "/workspace/lehome-openpi/checkpoints/pi05_lehome_camera_cv_robot_finetune/lehome_train_with_state_injected/800" \
  . \
  --repo-type model



```bash
mkdir -p /workspace/cache/openpi
export OPENPI_DATA_HOME=/workspace/cache/openpi
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export HF_LEROBOT_HOME=/workspace/.hf_home/lerobot
unset TRANSFORMERS_CACHE   # removes the deprecation warning path usage

mkdir -p /workspace/tmp
export HF_DATASETS_CACHE=/workspace/.hf_home/datasets
export TMPDIR=/workspace/tmp

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Download Dataset from HF
mkdir -p "${HF_LEROBOT_HOME}/huggingaccounttest/lehome_all_top_garment" && \
hf download huggingaccounttest/lehome_all_top_garment \
  --repo-type dataset \
  --local-dir "${HF_LEROBOT_HOME}/local/lehome_all_top_garment"


mkdir -p "${HF_LEROBOT_HOME}/huggingaccounttest/lehome_pretrain_human_all_tops_1" && \
hf download huggingaccounttest/lehome_pretrain_human_all_tops_1 \
  --repo-type dataset \
  --local-dir "${HF_LEROBOT_HOME}/local/lehome_train_episodes"






hf download huggingaccounttest/lehome_hum_pretrain16D_all_top_2_epoch \
  --repo-type model \
  --local-dir "/workspace/lehome-openpi/checkpoint_pretrain"

hf download huggingaccounttest/full-30epoch-OF-run \
  --repo-type model \
  --local-dir "D:\LeHome-Challenge\submission\openpi\submission\checkpoint"

Update Training Config for Policy

compute norm_stats for training data

uv run scripts/compute_norm_stats.py --config-name pi05_lehome_camera_cv_robot_finetune


XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 uv run scripts/train.py \
  pi05_lehome_camera_cv_robot_finetune\
  --exp-name lehome_robot_ft_state_injected\
  --overwrite \
  --fsdp-devices 1