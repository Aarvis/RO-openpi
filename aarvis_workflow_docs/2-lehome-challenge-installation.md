# Manual Installation Guide

This guide provides step-by-step instructions for manually installing the LeHome Challenge environment.

## Clear Project

deactivate 2>/dev/null || true
rm -rf .venv
uv venv --python 3.11
uv sync
UV_NO_BUILD_ISOLATION=1 ./third_party/IsaacLab/isaaclab.sh -i none


## Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) package manager
- GPU driver and CUDA supporting IsaacSim5.1.0.

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Aarvis/lehome-challenge.git

# git clone https://github.com/lehome-official/lehome-challenge.git
cd lehome-challenge
```

### 2. Install Dependencies with uv

```bash
uv sync
```

This will create a virtual environment and install all required dependencies.

### 3. Clone and Configure IsaacLab

```bash
cd third_party
git clone https://github.com/lehome-official/IsaacLab.git
cd ..
```

### 4. Install IsaacLab

Activate the virtual environment and install IsaacLab:

```bash
source .venv/bin/activate
./third_party/IsaacLab/isaaclab.sh -i none
```

flatdict issue
```bash
cd third_party/IsaacLab

# 1) Confirm where flatdict is pinned
grep -n "flatdict" source/isaaclab/setup.py

# 2) Patch 4.0.1 -> 4.0.0 (same as IsaacLab PR #4581)
sed -i 's/flatdict==4\.0\.1/flatdict==4.0.0/g' source/isaaclab/setup.py

# 3) Verify the patch
grep -n "flatdict" source/isaaclab/setup.py

cd ..

cd ..
```

```bash
# ensure flatdict 4.0.1 isn't lingering
uv pip uninstall flatdict || true
uv pip install -U "flatdict==4.0.0"

# clear uv build cache (you already did, but keep it)
rm -rf ~/.cache/uv/builds-v0

# rerun install
UV_NO_BUILD_ISOLATION=1 ./third_party/IsaacLab/isaaclab.sh -i none
```

### 5. Install LeHome Package

Finally, install the LeHome package in development mode:

```bash
uv pip install -e ./source/lehome
```

---
###
If you are using a server, please download the system dependencies.

```bash
    #step 1
    sudo kill -9 12440
    sudo dpkg --configure -a
    sudo apt -f install
    sudo apt update
    sudo apt install -y \
    libglu1-mesa \
    libgl1 \
    libegl1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxext6 \
    libx11-6
    #step 2
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
```


#### Download Simulation Assets

Download the required simulation assets (scenes, objects, robots) from HuggingFace:

```bash
# This creates the Assets/ directory with all required simulation resources
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
```

#### Download Example Dataset

We provide demonstrations for four types of garments. Download from HuggingFace:

```bash
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

If you need depth information or individual data for each garment. Download from HuggingFace:

```bash
hf download lehome/dataset_challenge --repo-type dataset --local-dir Datasets/example
```

#### Install Open-pi Client for running Policy Eval
```bash
uv pip install -e ~/LEHOME/lehome-openpi/packages/openpi-client
```

