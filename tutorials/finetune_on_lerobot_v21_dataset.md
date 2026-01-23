# Tutorial: Fine-tuning on LeRobot V2.1 Dataset

This tutorial provides a tutorial for fine-tuning InternVLA-A1-3B with real-world dataset in Lerobot V2.1 format:
**download a dataset → convert it to v3.0 format → fine-tune InternVLA-A1-3B on the A2D Pick-Pen task.**

---

## 1. Prepare the post-training dataset

In this example, we use the **A2D Pick-Pen** task from the **Genie-1 real-robot dataset**.

### Step 1.1 Download the dataset from Hugging Face

```bash
hf download \
  InternRobotics/InternData-A1 \
  real/genie1/Put_the_pen_from_the_table_into_the_pen_holder.tar.gz \
  --repo-type dataset \
  --local-dir data
```

---

### Step 1.2 Extract and organize the dataset

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

## 2. Convert the dataset from v2.1 to v3.0 format

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

## 3. Compute normalization statistics for relative actions (required)

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

## 4. Fine-tune InternVLA-A1-3B on `v30/a2d_pick_pen`

### One-line command

```bash
bash launch/internvla_a1_3b_finetune.sh v30/a2d_pick_pen delta true
```

`v30/a2d_pick_pen` specifies the dataset, `delta` indicates that **relative (delta) actions** are used, and `true` means that **external normalization statistics** are loaded instead of using the dataset’s built-in `stats.json`.


### ⚠️ Important Note

Before running `launch/internvla_a1_3b_finetune.sh`, **make sure to replace the environment variables inside the script with your own settings**, including but not limited to:

* `HF_HOME`
* `WANDB_API_KEY`
* `CONDA_ROOT`
* CUDA / GPU-related environment variables
* Paths to your local dataset and output directories