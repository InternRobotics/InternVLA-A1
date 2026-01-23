# Evaluation on RoboTwin 2.0 Benchmark

## Table of Contents

- [Installation](#installation)
- [Running Evaluation](#running-evaluation)
- [Viewing Results](#viewing-results)

---

## Installation

### Step 1: Install Basic Environment

Install Vulkan drivers for graphics rendering:

```bash
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
```

### Step 2: Clone RoboTwin Repository

**RoboTwin 2.0 Repository:** [https://github.com/RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)

```bash
cd InternVLA-A1
git submodule update --init third_party/RoboTwin
```

### Step 3: Install Dependencies

Replace the requirements.txt and install dependencies:

```bash
cp -r evaluation/RoboTwin/requirements.txt third_party/RoboTwin/script/requirements.txt
cd third_party/RoboTwin
bash script/_install.sh
bash script/_download_assets.sh
cd ../../
```


> **Note:** For more details, please refer to the [official RoboTwin installation tutorial](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).



## Running Evaluation

```bash
bash evaluation/RoboTwin/eval.sh
```

> You can modify the following variables in `evaluation/RoboTwin/eval.sh`:

- **`PRETRAINED_CKPT`**: The model checkpoint to load.
- **`TASK_CONFIG`**: The RoboTwin evaluation settting "demo_clean/demo_randomized"
  - Example: `demo_clean`
- **`TASK_IDX`**: The task index to evaluate (index into the `TASK_NAMES` list inside `inference.py`).
  - Example: `0  # adjust_bottle` (the comment is only a human hint for the corresponding task name).
- **`BASE_OUTPUT_PATH`**: The root directory for evaluation outputs.
  - The final output directory is `${BASE_OUTPUT_PATH}/${TASK_CONFIG}/${TASK_IDX}` (the script variable `OUTPUT_PATH`).

## Viewing Results

> Videos are saved to `OUTPUT_PATH` and named `<success|failure>_<test_num>.mp4`.