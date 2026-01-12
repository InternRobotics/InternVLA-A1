<div align="center">

# InternVLA-A1: Unifying Understanding, Generation, and Action for Robotic Manipulation​

</div>

![image description](https://internrobotics.github.io/internvla-a1.github.io/imgs/method.jpg)

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/pdf/2601.02456)
[![Data](https://img.shields.io/badge/Model-HuggingFace-blue?logo=huggingface)](https://huggingface.co/InternRobotics/InternVLA-A1-3B)
[![Data](https://img.shields.io/badge/Data-HuggingFace-blue?logo=huggingface)](https://huggingface.co/datasets/InternRobotics/InternData-A1)
[![Website](https://img.shields.io/badge/Website-Pages-blue.svg)](https://internrobotics.github.io/internvla-a1.github.io/)

## 🔥 Highlights
**InternVLA-A1** unifies scene ***understanding***, visual foresight ***generation***, and ***action*** execution into a single framework.

- 🔮 *The Core: Synergizes MLLM's semantic understanding with world-model-style dynamic prediction, enabling it to "imagine" the future and guide adaptive actions.*
- 🚀 *The Fuel: Empowered by high-fidelity synthetic data ([InternData-A1](https://huggingface.co/datasets/InternRobotics/InternData-A1)).*
- ⚡ *The Output: Tackles highly dynamic scenarios with effortless mastery.*

<table width="100%">
  <tr>
    <td width="33%" style="border:0; padding: 5px;">
      <video src="https://github.com/user-attachments/assets/cb14f0d3-f1ff-49ca-bb63-59204f0a0eeb" autoplay loop muted playsinline style="width: 100%;"></video>
    </td>
    <td width="33%" style="border:0; padding: 5px;">
      <video src="https://github.com/user-attachments/assets/8b44781a-5dd1-4d4b-af78-5fbdb441bc5f" autoplay loop muted playsinline style="width: 100%;"></video>
    </td>
    <td width="33%" style="border:0; padding: 5px;">
      <video src="https://github.com/user-attachments/assets/52a9dd1e-ef94-4886-8d6a-a1276cac0b3f" autoplay loop muted playsinline style="width: 100%;"></video>
    </td>
  </tr>
  <tr>
    <td width="33%" style="border:0; padding: 5px;">
      <video src="https://github.com/user-attachments/assets/67adfe28-1f77-4441-a239-3dcbc28f34dc" autoplay loop muted playsinline style="width: 100%;"></video>
    </td>
    <td width="33%" style="border:0; padding: 5px;">
      <video src="https://github.com/user-attachments/assets/76fcef23-de3b-4acc-ba52-0bdef7c6c67a" autoplay loop muted playsinline style="width: 100%;"></video>
    </td>
    <td width="33%" style="border:0; padding: 5px;">
      <video src="https://github.com/user-attachments/assets/482e607c-8380-496b-8d5a-34a7ee0b2655" autoplay loop muted playsinline style="width: 100%;"></video>
    </td>
  </tr>
</table>

## 📅 TODO List
- [x] Release InternVLA-A1-3B
- [x] Add quick-start for fine-tuning on `lerobot/pusht`
- [ ] Release InternVLA-A1-2B
- [ ] Release guideline of large-scale dataset pretraining

## 📑 Table of Contents
- [Installation](#section-Installation)
- [Playground](#section-Playground)
- [Fine-tuning](#section-Finetuning)

<span id="section-Installation"></span>
## 🛠️ Installation

This repository has been tested on **Python 3.10** and **CUDA 12.8**.
We recommend using **conda** to create an isolated environment.

### 1. Create Conda Environment

```bash
conda create -y -n internvla_a1 python=3.10
conda activate internvla_a1

pip install --upgrade pip
```

### 2. Install System Dependencies

We use FFmpeg for video encoding/decoding and SVT-AV1 for efficient storage.

```bash
conda install -c conda-forge ffmpeg=7.1.1 svt-av1 -y
```

### 3. Install PyTorch (CUDA 12.8)

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install Python Dependencies

```bash
pip install torchcodec numpy scipy transformers==4.57.1 mediapy loguru pytest omegaconf
pip install -e .
```

### 5. Patch HuggingFace Transformers

We replace the default implementations of several model modules
(e.g., **π0**, **InternVLA_A1_3B**, **InternVLA_A1_2B**) to support custom architectures for robot learning.

```bash
TRANSFORMERS_DIR=${CONDA_PREFIX}/lib/python3.10/site-packages/transformers/

cp -r src/lerobot/policies/pi0/transformers_replace/models        ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/InternVLA_A1_3B/transformers_replace/models  ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/InternVLA_A1_2B/transformers_replace/models  ${TRANSFORMERS_DIR}
```

Make sure the target directory exists—otherwise create it manually.

### 6. Configure Environment Variables

```bash
export HF_TOKEN=your_token  # for downloading hf models, tokenizers, or processors
export HF_HOME=path_to_huggingface   # default: ~/.cache/huggingface
```

### 7. Link Local HuggingFace Cache

```bash
ln -s ${HF_HOME}/lerobot data
```

This allows the repo to access datasets via `./data/`.

---

<span id="section-Playground"></span>
## 🕹️ Playground

### Quick start with `lerobot/pusht`

#### One-line command

```bash
bash launch/internvla_a1_3b_finetune.sh lerobot/pusht abs false
```

Here, **`abs`** indicates using **absolute actions**, and **`false`** means that the training
script will use the **statistics file (`stats.json`) provided by `lerobot/pusht` itself**.

---
<span id="section-Finetuning"></span>
## 🎯 Fine-tuning

This section provides a tutorial for fine-tuning InternVLA-A1-3B with InternData-A1 real dataset:
**download a dataset → convert it to v3.0 format → fine-tune InternVLA-A1-3B on the A2D Pick-Pen task.**

---

### 1. Prepare the post-training dataset

In this example, we use the **A2D Pick-Pen** task from the **Genie-1 real-robot dataset**.

#### Step 1.1 Download the dataset from Hugging Face

```bash
hf download \
  InternRobotics/InternData-A1 \
  real/genie1/Put_the_pen_from_the_table_into_the_pen_holder.tar.gz \
  --repo-type dataset \
  --local-dir data
```

---

#### Step 1.2 Extract and organize the dataset

Extract the downloaded archive, clean up intermediate files, and rename the dataset to follow the A2D naming convention:

```bash
tar -xzf data/real/genie1/Put_the_pen_from_the_table_into_the_pen_holder.tar.gz -C data

rm -rf data/real

mkdir -p data/v21
mv data/set_0 data/v21/a2d_pick_pen
```

After this step, the dataset directory structure should be:

```text
data/
└── v21/
    └── a2d_pick_pen/
        ├── data/
        ├── meta/
        └── videos/
```

---

### 2. Convert the dataset from v2.1 to v3.0 format

The original dataset is stored in **LeRobot v2.1** format.
This project requires **LeRobot v3.0**, so a format conversion is required.

Run the following command to convert the dataset:

```bash
python src/lerobot/datasets/v30/convert_my_dataset_v21_to_v30.py \
    --old-repo-id v21/a2d_pick_pen \
    --new-repo-id v30/a2d_pick_pen
```

After conversion, the dataset will be available at:

```text
data/v30/a2d_pick_pen/
```

---

### 3. Compute normalization statistics for relative actions (required)

This project fine-tunes policies using **relative (delta) actions**.
Therefore, you must compute per-dataset **normalization statistics** (e.g., mean/std) for the action stream before training.

Run the following command to compute statistics for `v30/a2d_pick_pen`:

```bash
python util_scripts/compute_norm_stats_single.py \
  --action_mode delta \
  --chunk_size 50 \
  --repo_id v30/a2d_pick_pen
```

This script will write a `stats.json` file under ```${HF_HOME}/lerobot/stats/delta/v30/a2d_pick_pen/stats.json```.

---

### 4. Fine-tune InternVLA-A1-3B on `v30/a2d_pick_pen`

#### One-line command

```bash
bash launch/internvla_a1_3b_finetune.sh v30/a2d_pick_pen delta true
```

`v30/a2d_pick_pen` specifies the dataset, `delta` indicates that **relative (delta) actions** are used, and `true` means that **external normalization statistics** are loaded instead of using the dataset’s built-in `stats.json`.


#### ⚠️ Important Note

Before running `launch/internvla_a1_3b_finetune.sh`, **make sure to replace the environment variables inside the script with your own settings**, including but not limited to:

* `HF_HOME`
* `WANDB_API_KEY`
* `CONDA_ROOT`
* CUDA / GPU-related environment variables
* Paths to your local dataset and output directories

<!-- ## 🌐 Pre-Training -->


## License and Citation
All the code within this repo are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please consider citing our project if it helps your research.

```BibTeX
@article{contributors2026internvla_a1,
  title={InternVLA-A1: Unifying Understanding, Generation and Action for Robotic Manipulation},
  author={InternVLA-A1 contributors},
  journal={arXiv preprint arXiv:2601.02456},
  year={2026}
}
```

## ❤️ Acknowledgments

- [Lerobot](https://github.com/huggingface/lerobot)
- [openpi](https://github.com/Physical-Intelligence/openpi)
- [InternVL](https://github.com/OpenGVLab/InternVL)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [COSMOS](https://github.com/nvidia-cosmos)

