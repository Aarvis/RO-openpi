git clone --recurse-submodules https://github.com/Aarvis/lehome-openpi

cd lehome-openpi

git submodule update --init --recursive

curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc        

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

pip install huggingface_hub


mkdir -p /ephemeral/cache/openpi
export OPENPI_DATA_HOME=/ephemeral/cache/openpi
export HF_HOME=/ephemeral/.hf_home
export HUGGINGFACE_HUB_CACHE=/ephemeral/.hf_home/hub
export HF_LEROBOT_HOME=/ephemeral/.hf_home/lerobot
unset TRANSFORMERS_CACHE   # removes the deprecation warning path usage

mkdir -p /ephemeral/tmp
export HF_DATASETS_CACHE=/ephemeral/.hf_home/datasets
export TMPDIR=/ephemeral/tmp

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

hf auth login
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

hf download huggingaccounttest/lehome_all_garment_data \
  --repo-type dataset \
  --local-dir "${HF_LEROBOT_HOME}/local/lehome_all_garment_data"


$env:CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

python "future_latent_predictor\Dataset creation\generate_future_latent_dataset.py" `
  --config-name pi05_lehome_camera_cv_robot_finetune_future_latent `
  --output-dir "\ephemeral\future_latent_predictor\generated_sim_root\output" `
  --future-offset 10 `
  --batch-size 8 `
  --shard-size 512 `
  --read-workers 2 `
  --read-prefetch-batches 2 `
  --max-pending-shard-writes 2 `
  --embedding-dtype float16 `
  --num-gpu-workers auto `
  --worker-launch-stagger-seconds 8 `
  --debug-start-image-count 5



