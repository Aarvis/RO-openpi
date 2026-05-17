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

hf upload huggingaccounttest/lehome_all_garment_data \
  "/workspace/.hf_home/lerobot/local/lehome_all_garment_data" \
  . \
  --repo-type dataset


hf upload huggingaccounttest/lehome_pretrain_all_garment_data \
  "/workspace/.hf_home/lerobot/local/lehome_all_garment_data_16d_pretrain" \
  . \
  --repo-type dataset





hf upload huggingaccounttest/lehome_val_episodes \
  "/datadrive/hf_cache/lerobot/local/lehome_val_episodes" \
  . \
  --repo-type dataset && \
hf repos settings huggingaccounttest/lehome_val_episodes \
  --repo-type dataset \
  --gated manual

hf upload huggingaccounttest/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_20_epoch\
  "/workspace/LEHOME/lehome-openpi/checkpoints/pi05_lehome_camera_cv_robot_finetune/lehome_pretrain_base_robot_ft_both_with_state_all_garment/21100" \
  . \
  --repo-type model


hf upload huggingaccounttest/robot_ft_only_with_state_all_garment_2_epoch \
  "/workspace/LEHOME/lehome-openpi/checkpoints/pi05_lehome_camera_cv_robot_finetune/robot_only_ft_with_state_all_garments/1350" \
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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```


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
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


mkdir -p /scratch/cache/openpi
export OPENPI_DATA_HOME=/scratch/cache/openpi
export HF_HOME=/scratch/.hf_home
export HUGGINGFACE_HUB_CACHE=/scratch/.hf_home/hub
export HF_LEROBOT_HOME=/scratch/.hf_home/lerobot
unset TRANSFORMERS_CACHE   # removes the deprecation warning path usage

mkdir -p /scratch/tmp
export HF_DATASETS_CACHE=/scratch/.hf_home/datasets
export TMPDIR=/scratch/tmp

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1



mkdir -p /mnt/persist/cache/openpi
export OPENPI_DATA_HOME=/mnt/persist/cache/openpi
export HF_HOME=/mnt/persist/.hf_home
export HUGGINGFACE_HUB_CACHE=/mnt/persist/.hf_home/hub
export HF_LEROBOT_HOME=/mnt/persist/.hf_home/lerobot
unset TRANSFORMERS_CACHE   # removes the deprecation warning path usage

mkdir -p /mnt/persist/tmp
export HF_DATASETS_CACHE=/mnt/persist/.hf_home/datasets
export TMPDIR=/mnt/persist/tmp

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1



Download Dataset from HF
mkdir -p "${HF_LEROBOT_HOME}/local/lehome_all_top_garment" && \
hf download huggingaccounttest/lehome_all_top_garment \
  --repo-type dataset \
  --local-dir "${HF_LEROBOT_HOME}/local/lehome_all_top_garment"


mkdir -p "${HF_LEROBOT_HOME}/local/lehome_all_garment_data" && \
hf download huggingaccounttest/lehome_all_garment_data \
  --repo-type dataset \
  --local-dir "${HF_LEROBOT_HOME}/local/lehome_all_garment_data"


mkdir -p "${HF_LEROBOT_HOME}/local/lehome_all_garment_data_16d_pretrain" && \
hf download huggingaccounttest/lehome_all_garment_data_16d_pretrain \
  --repo-type dataset \
  --local-dir "${HF_LEROBOT_HOME}/local/lehome_all_garment_data_16d_pretrain"

mkdir -p "${HF_LEROBOT_HOME}/local/lehome_val_episodes" && \
hf download huggingaccounttest/lehome_val_episodes \
  --repo-type dataset \
  --local-dir "${HF_LEROBOT_HOME}/local/lehome_val_episodes"


huggingaccounttest/lehome_train_episodes





hf download huggingaccounttest/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_20_epoch \
  --repo-type model \
  --local-dir "/mnt/persist/LEHOME/lehome-openpi/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_20_epoch"

hf download huggingaccounttest/pretrain_base_all_garment_4_epoch \
  --repo-type model \
  --local-dir "/home/ubuntu/LEHOME/lehome-openpi/pretrain_base_all_garment_4_epoch"


hf download huggingaccounttest/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_10_epoch \
  --repo-type model \
  --local-dir "/workspace/lehome-openpi/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_10_epoch" 


hf download huggingaccounttest/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_10_epoch \
  --repo-type model \
  --local-dir "/home/lehome-submission/submission/checkpoint/b_4_ft_10"  


hf download huggingaccounttest/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_4_epoch \
  --repo-type model \
  --local-dir "/workspace/lehome-openpi/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_4_epoch"  


/workspace/LEHOME/lehome-openpi/pretrain_base_all_garment_4_epoch
/params

hf download huggingaccounttest/full-30epoch-OF-run \
  --repo-type model \
  --local-dir "D:\LeHome-Challenge\submission\openpi\submission\checkpoint"


hf download huggingaccounttest/robot_ft_only_with_state_all_garment_10_epoch \
  --repo-type model \
  --local-dir "/home/lehome-submission/submission/checkpoint/ft_with_state_all_garment"

Update Training Config for Policy

compute norm_stats for training data

uv run scripts/compute_norm_stats.py --config-name pi05_lehome_camera_cv_robot_finetune

uv run scripts/compute_norm_stats.py --config-name pi05_lehome_precomputed_16d_pretrain

uv run scripts/compute_norm_stats.py --config-name pi05_lehome_camera_cv_robot_finetune


XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 uv run scripts/train.py \
  pi05_lehome_precomputed_16d_pretrain\
  --exp-name lehome_pretrain_with_state_all_garments\
  --overwrite \
  --fsdp-devices 1

XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 uv run scripts/train.py \
  pi05_lehome_camera_cv_robot_finetune\
  --exp-name lehome_pretrain_base_robot_ft_both_with_state_all_garment_ah5\
  --overwrite \
  --fsdp-devices 1


XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 uv run scripts/train.py \
  pi05_lehome_camera_cv_robot_finetune\
  --exp-name robot_only_ft_with_state_all_garments\
  --overwrite \
  --fsdp-devices 1





uv run scripts/multi_serve_policy.py \
  --num-servers 5 \
  --start-port 8000 \
  --gpu-id 0 \
  --total-gpu-fraction 0.96 \
  --xla-preallocate \
  --stagger-seconds 2.0 \
  --log-dir logs/multi_serve_policy \
  -- \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_robot_finetune \
  --policy.dir /workspace/lehome-openpi/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_10_epoch



uv run scripts/multi_serve_policy.py \
  --num-servers 5 \
  --start-port 8000 \
  --gpu-id 0 \
  --total-gpu-fraction 0.96 \
  --xla-preallocate \
  --stagger-seconds 2.0 \
  --log-dir logs/multi_serve_policy \
  -- \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_robot_finetune \
  --policy.dir /home/ubuntu/LEHOME/lehome-openpi/checkpoints/pi05_lehome_camera_cv_robot_finetune/lehome_pretrain_base_robot_ft_both_with_state_all_garment_ah5/27000




  uv run scripts/multi_serve_policy.py \
  --num-servers 5 \
  --start-port 8000 \
  --gpu-id 0 \
  --total-gpu-fraction 0.96 \
  --xla-preallocate \
  --stagger-seconds 2.0 \
  --log-dir logs/multi_serve_policy \
  -- \
  --send-policy-latent \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_robot_finetune \
  --policy.dir /mnt/persist/LEHOME/lehome-openpi/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_2_epoch


  uv run scripts/multi_serve_policy.py \
  --num-servers 4 \
  --start-port 8000 \
  --gpu-id 0 \
  --total-gpu-fraction 0.96 \
  --xla-preallocate \
  --stagger-seconds 2.0 \
  --log-dir logs/multi_serve_policy \
  -- \
  --send-policy-latent \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_robot_finetune \
  --policy.dir /datadrive/LEHOME/lehome-openpi/checkpoints/pi05_lehome_camera_cv_robot_finetune/lehome_cv_weighted_run1/63000



  
  curl.exe -i `
  -H "Connection: Upgrade" `
  -H "Upgrade: websocket" `
  -H "Sec-WebSocket-Version: 13" `
  -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" `
  "http://156.19.254.6:8000/"


uv run scripts/multi_best_sample_serve_policy.py \
  --num-servers 5 \
  --start-port 8000 \
  --gpu-id 0 \
  --total-gpu-fraction 0.96 \
  --critic-gpu-fraction 0.10 \
  --critic-port 7998 \
  --num-samples 50 \
  --noise-scale 1.0 \
  --xla-preallocate \
  --stagger-seconds 2.0 \
  --log-dir logs/multi_best_sample_serve_policy \
  --critic-checkpoint "/mnt/persist/LEHOME/lehome-challenge/Datasets/OnlineRL/Critic/checkpoints/default_run/best.pt" \
  --critic-repo-root "/mnt/persist/LEHOME/lehome-challenge" \
  --critic-device cuda \
  --critic-amp-dtype auto \
  -- \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_robot_finetune \
  --policy.dir "/mnt/persist/LEHOME/lehome-openpi/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_10_epoch"


uv run scripts/multi_best_sample_serve_policy.py \
  --num-servers 2 \
  --start-port 8000 \
  --gpu-id 0 \
  --total-gpu-fraction 0.96 \
  --critic-gpu-fraction 0.05 \
  --critic-port 7998 \
  --num-samples 50 \
  --noise-scale 0.5 \
  --xla-preallocate \
  --stagger-seconds 2.0 \
  --log-dir logs/multi_best_sample_serve_policy \
  --critic-checkpoint "/workspace/lehome-challenge/Datasets/OnlineRL/Critic/checkpoint_seen_only_best/best.pt" \
  --critic-repo-root "/workspace/lehome-challenge/" \
  --critic-device cuda \
  --critic-amp-dtype auto \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_robot_finetune \
  --policy.dir "/workspace/lehome-openpi/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_10_epoch"


uv run scripts/multi_best_sample_serve_policy.py \
  --num-servers 4 \
  --start-port 8000 \
  --gpu-id 0 \
  --total-gpu-fraction 0.96 \
  --critic-gpu-fraction 0.05 \
  --critic-port 7998 \
  --num-samples 50 \
  --noise-scale 0.5 \
  --xla-preallocate \
  --stagger-seconds 2.0 \
  --log-dir logs/multi_best_sample_serve_policy \
  --critic-checkpoint "/workspace/lehome-challenge/Datasets/OnlineRL/Critic/checkpoint_best/best.pt" \
  --critic-repo-root "/workspace/lehome-challenge/" \
  --critic-device cuda \
  --critic-amp-dtype auto \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_robot_finetune \
  --policy.dir "/workspace/lehome-openpi/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_4_epoch"