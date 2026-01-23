
# Tutorial: Training on RoboTwin 2.0 Dataset

This tutorial explains how to use this codebase to fine-tune the pre-trained model on a RoboTwin dataset.

## 0. Link Local HuggingFace Cache

If you haven't linked HF_HOME, do it first
```bash
ln -s ${HF_HOME}/lerobot data
```


---

## 1. Download the preprocessed RoboTwin Dataset

First, download the preprocessed RoboTwin dataset in Lerobot v3.0 format from Hugging Face:

```bash
hf download hxma/RoboTwin-LeRobot-v3.0 \
  --repo-type dataset \
  --local-dir data/robotwin
```

This will place the dataset under `data/robotwin`.

---

## 2. Add Feature Name Remapping


This remapping is implemented in:

```
src/lerobot/transforms/constants.py
```

Specifically, you need to update the following dictionaries:

* `MASK_MAPPING`
* `FEATURE_MAPPING`
* `IMAGE_MAPPING`

with dataset-specific entries.

`robotwin` uses the robot type `"aloha"` and has been set in our codebase, so no changes are needed in this step.


## 3. Compute Relative Action Statistics

This codebase uses **relative (delta) actions**.
Therefore, we must compute **normalization statistics** for the delta-action representation of the dataset.

Run the following command:

```bash
DATASET_REPO_ID="$(
  find -L "data/robotwin" -mindepth 2 -maxdepth 2 -type d -name "aloha-agilex*" 2>/dev/null \
  | while read -r d; do
        if [[ -d "$d/meta" && -d "$d/videos" ]]; then
            echo "${d#data/}"
        fi
    done \
  | sort -u
)"

echo "DATASET_REPO_ID: ${DATASET_REPO_ID}"
```


```bash
python util_scripts/compute_norm_stats_multi.py \
  --action_mode delta \
  --chunk_size 50 \
  --repo_id ${DATASET_REPO_ID}
```

The resulting statistics will be saved to:

```
$HF_HOME/lerobot/stats/aloha/delta/<agg_xxxxxx>/stats.json
```
```bash
cp $HF_HOME/lerobot/stats/aloha/delta/<agg_xxxxxx>/stats.json $HF_HOME/lerobot/stats/aloha/delta/
```

These statistics are required for correct action normalization during training.

---

## 4. Fine-tune on `robotwin`

With all configurations in place, you can now fine-tune the model on `robotwin`:

```bash
bash launch/internvla_a1_3b_finetune_robotwin.sh
```

This will launch the fine-tuning job using the dataset-specific mappings and normalization statistics defined above.