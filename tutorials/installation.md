# Installation

## Requirements
The code is built and tested with **Python 3.10**, **CUDA 12.8**, and **PyTorch 2.7.1**.

## Preparation

### 1. Clone the repository
```bash
git clone https://github.com/InternRobotics/InternVLA-A1.git
cd InternVLA-A1
```

### 2. Create Conda Environment

```bash
conda create -y -n internvla_a1 python=3.10
conda activate internvla_a1

pip install --upgrade pip
```

### 3. Install System Dependencies

We use FFmpeg for video encoding/decoding and SVT-AV1 for efficient storage.

```bash
conda install -c conda-forge ffmpeg=7.1.1 svt-av1 -y
```

### 4. Install PyTorch (CUDA 12.8)

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

### 5. Install Python Dependencies

```bash
pip install torchcodec numpy scipy transformers==4.57.1 mediapy loguru pytest omegaconf
pip install -e .
```

### 6. Patch HuggingFace Transformers

We replace the default implementations of several model modules
(e.g., **π0**, **InternVLA_A1_3B**, **InternVLA_A1_2B**) to support custom architectures for robot learning.

```bash
TRANSFORMERS_DIR=${CONDA_PREFIX}/lib/python3.10/site-packages/transformers/

cp -r src/lerobot/policies/pi0/transformers_replace/models        ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/InternVLA_A1_3B/transformers_replace/models  ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/InternVLA_A1_2B/transformers_replace/models  ${TRANSFORMERS_DIR}
```

Make sure the target directory exists—otherwise create it manually.

### 7. Configure Environment Variables

```bash
export HF_TOKEN=your_token  # for downloading hf models, tokenizers, or processors
export HF_HOME=path_to_huggingface   # default: ~/.cache/huggingface
```

### 8. Link Local HuggingFace Cache

```bash
ln -s ${HF_HOME}/lerobot data
```

This allows the repo to access datasets via `./data/`.