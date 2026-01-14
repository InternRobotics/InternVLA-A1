# Tutorial: Pretraining **InternVLA-A1-3B** with **InternData-A1**

This tutorial explains how to use the **InternData-A1** dataset to pretrain **InternVLA-A1-3B**, step by step.

---

## 1. Download [InternData-A1](https://huggingface.co/datasets/InternRobotics/InternData-A1)

First, download the dataset to the local `data/` directory.  

```bash
hf download InternRobotics/InternData-A1 \
  --repo-type dataset \
  --local-dir data \
  --include "sim_updated_lerobotv30/**"
```

---

## 2. Extract All Datasets

Recursively extract all `.tar.gz` files under `data/sim`, then remove the compressed files to save space.

```bash
find data/sim -type f -name "*.tar.gz" -print0 \
| while IFS= read -r -d '' f; do
    dir="$(dirname "$f")"
    echo "Extracting $f -> $dir"
    tar -xzf "$f" -C "$dir" && rm -f "$f"
done
```

After extraction, move the data to the v2.1 layout:

```bash
mv data/sim_updated_lerobotv30 data/a1
```

---

## 3. Compute Normalization Statistics (Per Embodiment)

InternData-A1 contains data from **five robot embodiments**:

* `Franka`  # franka
* `Genie-1`  # genie1
* `ARX Lift-2`  # lift2
* `AgileX Split Aloha`  # split_aloha
* `ARX AC One`  # acone

For each robot type, we compute **relative (delta) normalization statistics** and store them at:

```
${HF_HOME}/lerobot/stats/<robot_type>/delta/stats.json
```

### Example: Computing Stats for **AgileX Split Aloha**

```bash
ROBOT_TYPE="AgileX Split Aloha"
KEY_STRING="split_aloha"

DATASET_REPO_ID="$(
  find -L data/a1 -type d -name data 2>/dev/null \
  | while read -r d; do
      root="$(dirname "$d")"
      if [[ -d "$root/meta" && -d "$root/videos" ]]; then
        echo "${root#data/}"
      fi
    done \
  | grep -Ei "(^|/)${KEY_STRING}[^/]*/"
)"
```

Then run:

```bash
python util_scripts/compute_norm_stats_multi.py \
  --action_mode delta \
  --chunk_size 50 \
  --repo_ids ${DATASET_REPO_ID} \
  --output_dir data/stats/${ROBOT_TYPE}/delta
```

Repeat this process for each robot type (`Franka`, `Genie-1`, `ARX Lift-2`, `ARX AC One`).

---

## 5. Launch Pretraining

Once **all datasets are converted** and **all statistics are computed**, you are ready to start pretraining.

For **InternVLA-A1-3B**, simply run:

```bash
bash launch/internvla_a1_3b_pretrain.sh
```

This will start the joint pretraining over InternData-A1 using the prepared datasets and normalization statistics.

---

## Summary

1. Download InternData-A1 (simulation data only)
2. Extract all `.tar.gz` files
3. Convert datasets from v2.1 → v3.0
4. Compute per-robot normalization statistics (delta mode)
5. Launch InternVLA-A1-3B pretraining

You are now ready to pretrain InternVLA-A1 models on InternData-A1 🚀
