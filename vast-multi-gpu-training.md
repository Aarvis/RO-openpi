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
  --repo-type dataset 
  --gated manual

hf upload huggingaccounttest/lehome_all_garment_data \
  "/workspace/.hf_home/lerobot/local/lehome_all_garment_data" \
  . \
  --repo-type dataset


huggingaccounttest/cotrain_base_future_latent_ah10_robot_only_polish8

 



hf upload huggingaccounttest/sim_future_latent_ft_from_base_v1_epoch11_new \
  "/ephemeral2/checkpoints/pi05_lehome_camera_cv_robot_finetune_future_latent/resume_from_epoch10/15000" \
  . \
  --repo-type model


hf upload huggingaccounttest/lehome_pretrain_all_garment_data \
  "/workspace/.hf_home/lerobot/local/lehome_all_garment_data_16d_pretrain" \
  . \
  --repo-type dataset


hf upload huggingaccounttest/lehome_robot_real_all_garment_round2_data `
  "C:\Work\robot_real_ft_lehome_all_garment_data_episode_parquets_with_images" `
  . `
  --repo-type dataset

hf upload-large-folder huggingaccounttest/lehome_pretrain_all_garment_round2_data `
  "D:\Lehome-Dataset\lehome_round_2_dataset\pretrain_dataset\pretrain_lehome_all_garment_data_z180" `
  --repo-type dataset `
  --num-workers 16

hf upload huggingaccounttest/lehome_robot_real_all_garment_round2_data `
  "C:\Work\robot_real_ft_lehome_all_garment_data_episode_parquets_with_images" `
  . `
  --repo-type dataset `
  --delete "*"





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


hf upload huggingaccounttest/cotrain_base_ah10_robot_only_polish3\
  "~/lehome-openpi/checkpoints/pi05_lehome_camera_cv_multi_cotrain_robot_finetune/cotrain_base_ah10_robot_only_polish6/3500" \
  . \
  --repo-type model

hf download huggingaccounttest/cotrain_base_ah10_robot_only_polish3\
  --local-dir ./cotrain_base_ah10_robot_only_polish3 \
  --repo-type model


hf download huggingaccounttest/lehome_robot_real_all_garment_round2_data --local-dir D:\Lehome-Dataset\real-robot-hf --repo-type dataset




hf upload huggingaccounttest/docker-image-sim-lehome-policy-r55-cotrainbase-polish1-3-5090\
  "/home/ubuntu/sim-lehome-policy-r55-cotrainbase-polish1-3-5090" \
  . \
  --repo-type model


hf upload huggingaccounttest/multidata_cotrain_base_human_sim_robot_polish1_3_epoch\
  "/home/ubuntu/LEHOME/lehome-openpi/checkpoints/pi05_lehome_camera_cv_multi_cotrain_robot_finetune/multicotrain_pretrain_base_real_sim_human_polish_1/4450" \
  . \
  --repo-type model


hf upload huggingaccounttest/robot_future_latents_dependency_multi_cotrain_base3\
  "/scratch2/sim_future_latent_dependency_weights_data" \
  . \
  --repo-type model


robot_future_latents_dependency_multi_cotrain_base3



hf upload huggingaccounttest/sim_future_latent_ft_from_base_v1_epoch11\
  "/ephemeral/checkpoints/pi05_lehome_camera_cv_robot_finetune_future_latent/resume_from_epoch10/15000" \
  . \
  --repo-type model



  hf upload huggingaccounttest/\
  "uv pip install -e /workspace/lehome-openpi/packages/openpi-client" \
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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,
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

mkdir -p ~/LEHOME/cache/openpi
export OPENPI_DATA_HOME=~/LEHOME/cache/openpi
export HF_HOME=~/LEHOME/.hf_home
export HUGGINGFACE_HUB_CACHE=~/LEHOME/.hf_home/hub
export HF_LEROBOT_HOME=~/LEHOME/.hf_home/lerobot
unset TRANSFORMERS_CACHE   # removes the deprecation warning path usage

mkdir -p ~/LEHOME/tmp
export HF_DATASETS_CACHE=~/LEHOME/.hf_home/datasets
export TMPDIR=~/LEHOME/tmp

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


mkdir -p /ephemeral2/cache/openpi
export OPENPI_DATA_HOME=/ephemeral2/cache/openpi
export HF_HOME=/ephemeral2/.hf_home
export HUGGINGFACE_HUB_CACHE=/ephemeral2/.hf_home/hub
export HF_LEROBOT_HOME=/ephemeral2/.hf_home/lerobot
unset TRANSFORMERS_CACHE   # removes the deprecation warning path usage

mkdir -p /ephemeral2/tmp
export HF_DATASETS_CACHE=/ephemeral2/.hf_home/datasets
export TMPDIR=/ephemeral2/tmp

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p /ephemeral2/cache/openpi
export OPENPI_DATA_HOME=/ephemeral2/cache/openpi
export HF_HOME=/ephemeral2/.hf_home
export HUGGINGFACE_HUB_CACHE=/ephemeral2/.hf_home/hub
export HF_LEROBOT_HOME=/ephemeral2/.hf_home/lerobot
unset TRANSFORMERS_CACHE   # removes the deprecation warning path usage

mkdir -p /ephemeral2/tmp
export HF_DATASETS_CACHE=/ephemeral2/.hf_home/datasets
export TMPDIR=/ephemeral2/tmp

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1




uv run scripts/multi_serve_policy.py \
  --num-servers 4 \
  --start-port 8000 \
  --gpu-id 0 \
  --total-gpu-fraction 0.95 \
  --xla-preallocate \
  --log-dir logs/multi_serve_policy_future_latent \
  -- \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent \
  --policy.dir /workspace/LEHOME/lehome-openpi/cotrain_base_future_latent_sim_only_trial_1500


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
export CUDA_VISIBLE_DEVICES=0,1,2,3
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
export CUDA_VISIBLE_DEVICES=0,1
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


mkdir -p "${HF_LEROBOT_HOME}/local/lehome_pretrain_all_garment_round2_data" && \
hf download huggingaccounttest/lehome_pretrain_all_garment_round2_data \
  --repo-type dataset \
  --local-dir "${HF_LEROBOT_HOME}/local/lehome_pretrain_all_garment_round2_data"

hf download huggingaccounttest/lehome_robot_real_all_garment_round2_data \
  --repo-type dataset \
  --local-dir "${HF_LEROBOT_HOME}/local/lehome_robot_real_all_garment_round2_data"

hf download huggingaccounttest/lehome_robot_sim_all_garment_round2_data \
  --repo-type dataset \
  --local-dir "${HF_LEROBOT_HOME}/local/lehome_robot_sim_all_garment_round2_data"


mkdir -p "${HF_LEROBOT_HOME}/local/lehome_val_episodes" && \
hf download huggingaccounttest/lehome_val_episodes \
  --repo-type dataset \
  --local-dir "${HF_LEROBOT_HOME}/local/lehome_val_episodes"


hf download lehome/dataset_challenge_real --repo-type dataset --local-dir "D:\Lehome-Dataset\robot_real_vanilla"




huggingaccounttest/lehome_train_episodes


hf download huggingaccounttest/cotrain_base_future_latent_sim_only_trial_1500 \
  --repo-type model \
  --local-dir "/workspace/LEHOME/lehome-openpi/cotrain_base_future_latent_sim_only_trial_1500"


hf download huggingaccounttest/sim_future_latent_finetune_furthur_from_13epoch \
  --repo-type model \
  --local-dir "/workspace/lehome-openpi/sim_future_latent_finetune_furthur_from_13epoch"

hf download huggingaccounttest/sim_future_latent_finetune_furthur_from_13epoch \
  --repo-type model \
  --local-dir "/ephemeral2/sim_future_latents_dependency_multi_cotrain_base3"


hf download huggingaccounttest/sim_future_latent_ft_from_base_v1_epoch10 \
  --repo-type model \
  --local-dir "/workspace/lehome-openpi/sim_future_latent_ft_from_base_v1_epoch10"


uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_robot_finetune_future_latent \
  --policy.dir /workspace/lehome-openpi/sim_future_latent_ft_from_base_v1_epoch10

cd ..
cd /workspace/lehome-openpi

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

uv run --no-sync scripts/multi_serve_policy.py \
  --num-servers 1 \
  --start-port 8003 \
  --gpu-id 3 \
  --total-gpu-fraction 0.90 \
  --xla-preallocate \
  --log-dir logs/multi_serve_policy_future_latent_gpu0 \
  -- \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent \
  --policy.dir /workspace/lehome-openpi/cotrain_base_future_latent_ah10_robot_only_polish8

uv run --no-sync scripts/multi_serve_policy.py \
  --num-servers 1 \
  --start-port 8003 \
  --gpu-id 3 \
  --total-gpu-fraction 0.90 \
  --xla-preallocate \
  --log-dir logs/multi_serve_policy_future_latent_gpu0 \
  -- \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent_with_ppo_heads \
  --policy.dir /workspace/lehome-openpi/cotrain_base_future_latent_ah10_robot_only_polish8

uv run --no-sync scripts/multi_serve_policy.py \
  --num-servers 1 \
  --start-port 8000 \
  --gpu-id 0 \
  --total-gpu-fraction 0.90 \
  --xla-preallocate \
  --log-dir logs/multi_serve_policy_gpu0 \
  -- \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent \
  --policy.dir /dev/shm/cotrain_base_future_latent_sim_only_trial_1500

nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | awk -F',' '$2 ~ /python/ {print $1}' | xargs -r kill -9

uv run --no-sync scripts/multi_serve_policy.py \
  --num-servers 1 \
  --start-port 8000 \
  --gpu-id 0 \
  --total-gpu-fraction 0.90 \
  --xla-preallocate \
  --log-dir logs/multi_serve_policy_future_latent_gpu0 \
  -- \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_robot_finetune \
  --policy.dir /dev/shm/robot_ft_only_with_state_all_garment_4_epoch

source /venv/main/bin/activate


hf download huggingaccounttest/robot_ft_only_with_state_all_garment_4_epoch \
  --repo-type model \
  --local-dir "/dev/shm/robot_ft_only_with_state_all_garment_4_epoch" 

hf download huggingaccounttest/robot_future_latents_depedency_cotrain_final\
  --repo-type model \
  --local-dir "/workspace/lehome-openpi/robot_future_latents_depedency_cotrain_final" \
  --include "resampler_autoencoder_sim_real_run_v1/**"

  --include "robot_future_predictor_v2/**"




hf download huggingaccounttest/sim_future_latent_ft_from_base_v1_epoch15_new\
  --repo-type model \
  --local-dir "/workspace/lehome-openpi/sim_future_latent_ft_from_base_v1_epoch15_new"


  




hf download huggingaccounttest/multidata_cotrain_base_human_sim_robot_polish1_3_epoch \
  --repo-type model \
  --local-dir "/home/ubuntu/LEHOME/multidata_cotrain_base_human_sim_robot_polish1_3_epoch"

hf download huggingaccounttest/pretrain_base_all_garment_4_epoch \
  --repo-type model \
  --local-dir "/home/ubuntu/LEHOME/lehome-openpi/pretrain_base_all_garment_4_epoch"


uv run scripts/train.py \
  pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent \
  --exp-name cotrain_base_future_latent_robot_only_polish8 \
  --fsdp-devices 1 \
  --overwrite



 


hf download huggingaccounttest/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_10_epoch \
  --repo-type model \
  --local-dir "~/LEHOME/lehome-openpi/pretrain_base_4_epoch_robot_ft_both_with_state_all_garment_10_epoch" 


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


XLA_PYTHON_CLIENT_MEM_FRACTION=0.98 uv run scripts/train_weighted.py \
  pi05_lehome_camera_cv_multi_cotrain_robot_finetune \
  --exp-name multicotrain_pretrain_base_real_sim_human_polish_1 \
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
  --policy.config pi05_lehome_camera_cv_multi_cotrain_robot_finetune \
  --policy.dir /home/ubuntu/LEHOME/lehome-openpi/pretrain_multidata_cotrain_base_with_state_hum_sim_rob_3_epoch



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



  uv run scripts/multi_serve_policy.py \
  --num-servers 2 \
  --start-port 8002 \
  --gpu-id 0 \
  --total-gpu-fraction 0.93 \
  --total-ppo-gpu-fraction 0.04 \
  --xla-preallocate \
  --stagger-seconds 2.0 \
  --log-dir logs/multi_serve_policy_ppo_gpu1 \
  -- \
  policy:checkpoint \
  --policy.config pi05_lehome_trained_vla_with_ppo_heads \
  --policy.ppo-device cuda:0

  uv run scripts/multi_serve_policy.py \
  --num-servers 2 \
  --start-port 8000 \
  --gpu-id 1 \
  --total-gpu-fraction 0.93 \
  --total-ppo-gpu-fraction 0.04 \
  --xla-preallocate \
  --stagger-seconds 2.0 \
  --log-dir logs/multi_serve_policy_ppo_gpu1 \
  -- \
  policy:checkpoint \
  --policy.config pi05_lehome_trained_vla_with_ppo_heads \
  --policy.ppo-device cuda:0




python "future_latent_predictor/Dataset creation/generate_future_latent_dataset.py" \
--config-name pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent \
--params-path "/home/ubuntu/LEHOME/lehome-openpi/pretrain_multidata_cotrain_base_with_state_hum_sim_rob_3_epoch/params" \
--output-dir "./future_latent_predictor/Dataset creation/output" \
--future-offset 5 \
--batch-size 4 \
--shard-size 512


uv run python "future_latent_predictor/Dataset creation/generate_future_latent_dataset.py" \
  --config-name pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent \
  --params-path "/home/ubuntu/LEHOME/lehome-openpi/pretrain_multidata_cotrain_base_with_state_hum_sim_rob_3_epoch/params" \
  --output-dir "./future_latent_predictor/Dataset creation/output" \
  --future-offset 5 \
  --batch-size 8 \
  --shard-size 128


uv run python "future_latent_predictor/Dataset creation/generate_future_latent_dataset.py" \
  --config-name pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent \
  --params-path "/scratch/pretrain_multidata_cotrain_base_with_state_hum_sim_rob_3_epoch/params" \
  --output-dir "/scratch/real_robot_generated_dataset/output" \
  --future-offset 5 \
  --batch-size 8 \
  --shard-size 128


uv run torchrun --standalone --nproc_per_node=4 \
  future_latent_predictor/resampler_autoencoder/train_resampler_autoencoder.py \
  --config future_latent_predictor/resampler_autoencoder/configs/v100s_4gpu_resampler_autoencoder.json \
  --data-root /scratch/real_robot_generated_dataset/output \
  --output-dir /scratch/future_latent_runs/resampler_autoencoder_v1


uv run torchrun --standalone --nproc_per_node=4 \
  future_latent_predictor/future_predictor/train_future_predictor.py \
  --config future_latent_predictor/future_predictor/configs/v100s_4gpu_future_predictor.json \
  --data-root /scratch/real_robot_generated_dataset/output \
  --resampler-checkpoint /scratch/future_latent_runs/resampler_autoencoder_v1/resampler_encoder_latest.pt \
  --output-dir /scratch/future_latent_runs/future_predictor_v1


uv run python "future_latent_predictor/Dataset creation/generate_future_latent_dataset.py" \
  --config-name pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent \
  --params-path "/scratch/pretrain_multidata_cotrain_base_with_state_hum_sim_rob_3_epoch/params" \
  --output-dir "/scratch2/sim_robot_generated_dataset/output" \
  --future-offset 5 \
  --batch-size 8 \
  --shard-size 128 \
  --debug-start-image-count 5

uv run torchrun --standalone --nproc_per_node=4   future_latent_predictor/future_predictor/train_future_predictor.py   --config future_latent_predictor/future_predictor/configs/v100s_4gpu_future_predictor.json   --data-root /scratch/future_latent_embedding_roots   --resampler-checkpoint /scratch2/future_latent_dependency_weights_data/resampler_autoencoder_sim_real_run_v1/resampler_encoder_latest.pt   --output-dir /scratch/sim_future_latent__run/future_predictor_v1