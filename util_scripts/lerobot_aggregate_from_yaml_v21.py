#!/usr/bin/env python
"""
LeRobot Dataset Merger — No-Feature-Change Version

Usage example:

python lerobot_aggregate_from_yaml_v21.py \
    --yaml_path /path/to/yaml_root \
    --output /path/to/out_root \
    --num_workers 4

Description:
- `/path/to/yaml_root` contains several *.yaml files.
  Each YAML file lists multiple v2.1 LeRobot dataset directories.
- The script will process each YAML file independently and merge
  all datasets specified inside that YAML into a single new dataset.
- The merged dataset will be written to:
      /path/to/out_root/<yaml_filename_without_extension>/
"""

import argparse
import json
import logging
import os
import yaml
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("merger.no_feature_change")

# ---------------- Helpers ----------------

def load_jsonl(file_path: str) -> List[dict]:
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                data.append(json.loads(s))
            except json.JSONDecodeError:
                # Accept array-form episodes_stats, fallback
                try:
                    arr = json.loads(f"[{s}]")
                    if isinstance(arr, list):
                        data.extend(arr)
                except Exception:
                    logger.warning(f"Skip bad JSONL line in {file_path}")
    return data


def save_jsonl(data: List[dict], file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def read_info(folder: str) -> dict:
    p = os.path.join(folder, "meta", "info.json")
    if not os.path.exists(p):
        return {}
    with open(p, "r") as f:
        return json.load(f)


def shapes_from_features(features: dict) -> Dict[str, Tuple]:
    out = {}
    for k, v in (features or {}).items():
        shp = tuple(v.get("shape", [])) if isinstance(v, dict) else tuple()
        out[k] = shp
    return out


def same_feature_shapes(infos: List[dict]) -> bool:
    if not infos:
        return True
    base = shapes_from_features(infos[0].get("features", {}))
    for inf in infos[1:]:
        cur = shapes_from_features(inf.get("features", {}))
        if base != cur:
            return False
    return True


# ---------------- Core ops (no padding) ----------------

def process_parquet_file_no_pad(
    source_path: str,
    dest_path: str,
    old_index: int,
    new_index: int,
    episode_to_frame_index: Optional[Dict[int, int]] = None,
    folder_task_mapping: Optional[Dict[str, Dict[int, int]]] = None,
    old_folder: Optional[str] = None,
    chunks_size: int = 1000,
) -> bool:
    """Copy parquet while updating minimal identifiers; DO NOT touch vector shapes/values."""
    try:
        df = pd.read_parquet(source_path)

        # Update episode_index
        if "episode_index" in df.columns:
            df["episode_index"] = new_index

        # Update global index if present
        if "index" in df.columns:
            if episode_to_frame_index and new_index in episode_to_frame_index:
                first_index = episode_to_frame_index[new_index]
            else:
                first_index = int(df["index"].min()) if len(df) else 0
            df["index"] = np.arange(
                first_index,
                first_index + len(df),
                dtype=df["index"].dtype if df["index"].dtype.kind in "iu" else None,
            )

        # Remap task_index if mapping provided
        if (
            "task_index" in df.columns
            and folder_task_mapping is not None
            and old_folder is not None
            and old_folder in folder_task_mapping
        ):
            m = folder_task_mapping[old_folder]
            # Only remap values that exist in mapping; keep others untouched
            df["task_index"] = df["task_index"].apply(lambda x: m.get(int(x), x))

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        df.to_parquet(dest_path, index=False)
        return True
    except Exception as e:
        logger.error(f"process_parquet_file_no_pad failed for {source_path}: {e}")
        return False


def find_episode_parquet(folder: str, episode_idx: int) -> Optional[str]:
    candidates = [
        os.path.join(folder, "parquet", f"episode_{episode_idx:06d}.parquet"),
        os.path.join(folder, "data", f"episode_{episode_idx:06d}.parquet"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # fallback recursive search
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            if fn.endswith(".parquet") and f"episode_{episode_idx:06d}" in fn:
                return os.path.join(root, fn)
    return None


def copy_videos_no_modify(
    first_info: dict,
    episode_mapping: List[Tuple[str, int, int]],
    output_folder: str,
    num_workers: Optional[int] = None,
) -> None:
    if num_workers is None:
        num_workers = min(4, mp.cpu_count())

    video_path_tpl = first_info.get("video_path", "")
    chunks_size = first_info.get("chunks_size", 1000)

    # Discover video keys from features (dtype=="video")
    video_keys = []
    for name, spec in (first_info.get("features", {}) or {}).items():
        if isinstance(spec, dict) and spec.get("dtype") == "video":
            video_keys.append(name)

    def _copy_one(args):
        old_folder, old_idx, new_idx = args
        ok_any = False
        old_chunk = old_idx // chunks_size
        new_chunk = new_idx // chunks_size

        for vkey in video_keys:
            src = os.path.join(
                old_folder,
                video_path_tpl.format(
                    episode_chunk=old_chunk,
                    video_key=vkey,
                    episode_index=old_idx,
                ),
            )
            if not os.path.exists(src):
                # Try alternative patterns
                alt = [
                    os.path.join(
                        old_folder,
                        f"videos/chunk-{old_chunk:03d}/{vkey}/episode_{old_idx:06d}.mp4",
                    ),
                ]
                src = next((p for p in alt if os.path.exists(p)), None)

            if src and os.path.exists(src):
                dst = os.path.join(
                    output_folder,
                    video_path_tpl.format(
                        episode_chunk=new_chunk,
                        video_key=vkey,
                        episode_index=new_idx,
                    ),
                )
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                ok_any = True
            else:
                logger.warning(
                    f"Video missing for {vkey}, episode {old_idx} in {old_folder}"
                )
        return ok_any

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_copy_one, m) for m in episode_mapping]
        ok = 0
        for fu in tqdm(as_completed(futures), total=len(futures), desc="Copy videos"):
            try:
                ok += 1 if fu.result() else 0
            except Exception as e:
                logger.error(f"Video copy failed: {e}")
        logger.info(
            f"Copied videos for {ok}/{len(episode_mapping)} episodes (where present)"
        )


def copy_data_files_no_pad(
    episode_mapping: List[Tuple[str, int, int]],
    output_folder: str,
    episode_to_frame_index: Optional[Dict[int, int]],
    folder_task_mapping: Optional[Dict[str, Dict[int, int]]],
    chunks_size: int = 1000,
    num_workers: Optional[int] = None,
) -> None:
    if num_workers is None:
        num_workers = min(4, mp.cpu_count())

    tasks = []
    for old_folder, old_index, new_index in episode_mapping:
        src = find_episode_parquet(old_folder, old_index)
        if not src:
            logger.warning(f"Missing parquet for episode {old_index} in {old_folder}")
            continue
        chunk_idx = new_index // chunks_size
        dst = os.path.join(
            output_folder,
            "data",
            f"chunk-{chunk_idx:03d}",
            f"episode_{new_index:06d}.parquet",
        )
        tasks.append(
            (
                src,
                dst,
                old_index,
                new_index,
                episode_to_frame_index,
                folder_task_mapping,
                old_folder,
                chunks_size,
            )
        )

    ok = 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(process_parquet_file_no_pad, *t): t for t in tasks}
        for fu in tqdm(as_completed(futures), total=len(futures), desc="Copy parquet"):
            try:
                ok += 1 if fu.result() else 0
            except Exception as e:
                logger.error(f"Parquet copy failed: {e}")
    logger.info(f"Parquet files copied: {ok}/{len(tasks)}")


# ---------------- Merger Class ----------------

class DatasetMergerNoFeatureChange:
    def __init__(
        self,
        source_folders: List[str],
        output_folder: str,
        num_workers: Optional[int] = None,
        copy_images: bool = False,
        copy_videos: bool = True,
        strict_features_match: bool = False,
    ) -> None:
        self.source_folders = source_folders
        self.output_folder = output_folder
        self.num_workers = num_workers
        self.copy_images_flag = copy_images
        self.copy_videos_flag = copy_videos
        self.strict = strict_features_match

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, "meta"), exist_ok=True)

        self.first_info = read_info(self.source_folders[0])
        if not self.first_info:
            raise RuntimeError("First dataset missing meta/info.json")

    def merge(self) -> None:
        logger.info(
            f"Merging {len(self.source_folders)} datasets into {self.output_folder} (no feature change)…"
        )

        infos = [read_info(f) for f in self.source_folders]
        if self.strict and not same_feature_shapes(infos):
            raise RuntimeError(
                "Feature shapes differ across sources under --strict_features_match"
            )

        episode_mapping, meta = self._build_mapping_and_meta()

        # Save concatenated meta files
        save_jsonl(
            meta["all_episodes"],
            os.path.join(self.output_folder, "meta", "episodes.jsonl"),
        )
        save_jsonl(
            meta["all_episodes_stats"],
            os.path.join(self.output_folder, "meta", "episodes_stats.jsonl"),
        )
        save_jsonl(
            meta["all_tasks"],
            os.path.join(self.output_folder, "meta", "tasks.jsonl"),
        )

        # Merge stats.json conservatively
        self._merge_stats_json_conservative()

        # Write info.json WITHOUT altering features
        self._write_info_without_feature_change(meta)

        # Copy videos (no re-encode)
        if self.copy_videos_flag:
            copy_videos_no_modify(
                self.first_info,
                episode_mapping,
                self.output_folder,
                self.num_workers,
            )

        # Copy parquet data w/ minimal updates (no padding)
        copy_data_files_no_pad(
            episode_mapping=episode_mapping,
            output_folder=self.output_folder,
            episode_to_frame_index=meta["episode_to_frame_index"],
            folder_task_mapping=meta["folder_task_mapping"],
            chunks_size=meta["chunks_size"],
            num_workers=self.num_workers,
        )

        # Optionally copy images tree (no changes)
        if self.copy_images_flag:
            self._copy_images_tree(episode_mapping)

        logger.info(
            f"Done. Output={self.output_folder} Total episodes={meta['total_episodes']} frames={meta['total_frames']}"
        )

    # ---------- internals ----------

    def _build_mapping_and_meta(self) -> Tuple[List[Tuple[str, int, int]], dict]:
        chunks_size = int(self.first_info.get("chunks_size", 1000))

        all_tasks: List[dict] = []
        task_desc_to_new_idx: Dict[str, int] = {}
        folder_task_mapping: Dict[str, Dict[int, int]] = {}

        all_episodes: List[dict] = []
        all_episodes_stats: List[dict] = []
        episode_mapping: List[Tuple[str, int, int]] = []

        total_frames = 0
        total_episodes = 0
        cumulative_frame_index = 0
        episode_to_frame_index: Dict[int, int] = {}

        for folder in tqdm(self.source_folders, desc="Scan datasets"):
            # tasks
            tasks = load_jsonl(os.path.join(folder, "meta", "tasks.jsonl"))
            folder_task_mapping[folder] = {}
            for t in tasks:
                desc = t.get("task")
                old_idx = t.get("task_index")
                if desc not in task_desc_to_new_idx:
                    task_desc_to_new_idx[desc] = len(all_tasks)
                    all_tasks.append(
                        {"task_index": task_desc_to_new_idx[desc], "task": desc}
                    )
                folder_task_mapping[folder][int(old_idx)] = task_desc_to_new_idx[desc]

            # episodes & stats
            eps = load_jsonl(os.path.join(folder, "meta", "episodes.jsonl"))
            eps_stats = load_jsonl(
                os.path.join(folder, "meta", "episodes_stats.jsonl")
            )
            stats_map = {s.get("episode_index"): s for s in eps_stats}

            for ep in eps:
                old_ep_idx = ep["episode_index"]
                new_ep_idx = total_episodes

                # append remapped episode meta
                ep_new = dict(ep)
                ep_new["episode_index"] = new_ep_idx
                all_episodes.append(ep_new)

                # remap per-episode stats only on index (leave stats payload untouched)
                if old_ep_idx in stats_map:
                    s = dict(stats_map[old_ep_idx])
                    s["episode_index"] = new_ep_idx
                    all_episodes_stats.append(s)

                episode_mapping.append((folder, old_ep_idx, new_ep_idx))

                # counters
                episode_to_frame_index[new_ep_idx] = cumulative_frame_index
                total_episodes += 1
                total_frames += int(ep.get("length", 0))
                cumulative_frame_index += int(ep.get("length", 0))

        meta = {
            "all_tasks": all_tasks,
            "all_episodes": all_episodes,
            "all_episodes_stats": all_episodes_stats,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "episode_to_frame_index": episode_to_frame_index,
            "folder_task_mapping": folder_task_mapping,
            "chunks_size": int(self.first_info.get("chunks_size", 1000)),
        }
        return episode_mapping, meta

    def _merge_stats_json_conservative(self) -> None:
        """Merge meta/stats.json only for features with identical shapes across sources.
        Otherwise, keep the first dataset's stats for that feature.
        """
        stats_paths = [os.path.join(f, "meta", "stats.json") for f in self.source_folders]
        stats_list = []
        for p in stats_paths:
            if os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        stats_list.append(json.load(f))
                except Exception as e:
                    logger.warning(f"Skip bad stats.json {p}: {e}")

        if not stats_list:
            logger.info("No stats.json found; skip writing merged stats.json")
            return

        first_info = self.first_info
        base_shapes = shapes_from_features(first_info.get("features", {}))

        merged: Dict[str, dict] = {}
        for feat in stats_list[0].keys():
            # Only merge features that exist in features dict and shapes align across infos
            if feat not in base_shapes:
                continue
            ok_all = True
            for inf in [self.first_info] + [
                read_info(f) for f in self.source_folders[1:]
            ]:
                shp = shapes_from_features(inf.get("features", {})).get(feat, None)
                if shp != base_shapes[feat]:
                    ok_all = False
                    break
            if not ok_all:
                # keep first dataset's stats for this feature
                merged[feat] = stats_list[0].get(feat, {})
                continue

            # Merge numeric fields conservatively where shapes match (mean/std/min/max/count)
            # Fall back to first if anything goes wrong
            try:
                fields = ["mean", "std", "min", "max", "count"]
                merged[feat] = {}
                # Gather arrays
                arrs = [{k: v.get(feat, {}).get(k) for k in fields} for v in stats_list]

                # Weighted mean/std by count if available
                counts = [a.get("count", [0])[0] if a.get("count") else 0 for a in arrs]
                total_cnt = sum(counts)

                for fld in ["min", "max"]:
                    if all(a.get(fld) is not None for a in arrs):
                        stacked = np.stack(
                            [np.array(a[fld]) for a in arrs], axis=0
                        )
                        if fld == "min":
                            merged[feat][fld] = np.min(stacked, axis=0).tolist()
                        else:
                            merged[feat][fld] = np.max(stacked, axis=0).tolist()

                if all(a.get("mean") is not None for a in arrs):
                    if total_cnt > 0 and all(a.get("count") for a in arrs):
                        weighted = [
                            np.array(a["mean"]) * (c / total_cnt)
                            for a, c in zip(arrs, counts)
                        ]
                        merged[feat]["mean"] = np.sum(weighted, axis=0).tolist()
                    else:
                        merged[feat]["mean"] = np.mean(
                            np.stack(
                                [np.array(a["mean"]) for a in arrs],
                                axis=0,
                            ),
                            axis=0,
                        ).tolist()

                if all(a.get("std") is not None for a in arrs):
                    if total_cnt > 0 and all(a.get("count") for a in arrs):
                        vars_ = [np.array(a["std"]) ** 2 for a in arrs]
                        weighted = [
                            v * (c / total_cnt) for v, c in zip(vars_, counts)
                        ]
                        merged[feat]["std"] = np.sqrt(
                            np.sum(weighted, axis=0)
                        ).tolist()
                    else:
                        merged[feat]["std"] = np.mean(
                            np.stack(
                                [np.array(a["std"]) for a in arrs],
                                axis=0,
                            ),
                            axis=0,
                        ).tolist()

                if any(a.get("count") for a in arrs):
                    merged[feat]["count"] = [int(total_cnt)]
            except Exception as e:
                logger.warning(f"Stats merge fallback for {feat}: {e}")
                merged[feat] = stats_list[0].get(feat, {})

        out_p = os.path.join(self.output_folder, "meta", "stats.json")
        with open(out_p, "w") as f:
            json.dump(merged, f, indent=2)
        logger.info("Wrote conservative merged stats.json")

    def _write_info_without_feature_change(self, meta: dict) -> None:
        info = dict(self.first_info)  # clone
        # Do NOT touch info["features"] at all
        info["total_episodes"] = int(meta["total_episodes"])
        info["total_frames"] = int(meta["total_frames"])
        info["total_tasks"] = int(len(meta["all_tasks"]))
        chunks_size = int(meta["chunks_size"])
        info["total_chunks"] = (info["total_episodes"] + chunks_size - 1) // chunks_size
        # keep splits simple, no assumptions
        info["splits"] = {"train": f"0:{info['total_episodes']}"}

        # total_videos best-effort: sum if present, else keep original
        tv = 0
        for f in self.source_folders:
            inf = read_info(f)
            tv += int(inf.get("total_videos", 0))
        if tv > 0:
            info["total_videos"] = tv

        out_p = os.path.join(self.output_folder, "meta", "info.json")
        with open(out_p, "w") as f:
            json.dump(info, f, indent=2)
        logger.info("Wrote info.json without altering features")

    def _copy_images_tree(self, episode_mapping: List[Tuple[str, int, int]]) -> None:
        # Shallow copy of images directory tree, remapping episode folder names only
        # This is optional and may be large; use with care
        img_root = os.path.join(self.output_folder, "images")
        os.makedirs(img_root, exist_ok=True)

        # Discover video keys from features (dtype==video)
        video_keys = []
        for name, spec in (self.first_info.get("features", {}) or {}).items():
            if isinstance(spec, dict) and spec.get("dtype") == "video":
                video_keys.append(name)

        def _copy_one(args):
            old_folder, old_idx, new_idx = args
            copied = 0
            for vkey in video_keys:
                src_dir = os.path.join(
                    old_folder, "images", vkey, f"episode_{old_idx:06d}"
                )
                if not os.path.exists(src_dir):
                    continue
                dst_dir = os.path.join(
                    self.output_folder, "images", vkey, f"episode_{new_idx:06d}"
                )
                os.makedirs(dst_dir, exist_ok=True)
                for fn in os.listdir(src_dir):
                    if fn.endswith(".png"):
                        shutil.copy2(
                            os.path.join(src_dir, fn), os.path.join(dst_dir, fn)
                        )
                        copied += 1
            return copied

        with ThreadPoolExecutor(
            max_workers=self.num_workers or min(4, mp.cpu_count())
        ) as ex:
            futures = [ex.submit(_copy_one, m) for m in episode_mapping]
            total = 0
            for fu in tqdm(
                as_completed(futures), total=len(futures), desc="Copy images"
            ):
                try:
                    total += fu.result()
                except Exception as e:
                    logger.error(f"Image copy failed: {e}")
        logger.info(f"Copied {total} images (if present)")


def extract_paths_from_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    base_path = data.get("path")
    task_ids = data.get("task_ids", [])

    if base_path is None:
        return []

    base_path = base_path.rstrip("/*")

    return [str(Path(base_path) / str(t)) for t in task_ids]


def load_all_yaml_paths(root_dir):
    """Recursively search all .yaml files under root_dir and return {yaml_path: [dataset_path1, ...]}"""
    yaml_files = glob(os.path.join(root_dir, "**/*.yaml"), recursive=True)

    result = {}

    for y in yaml_files:
        print(y)
        paths = extract_paths_from_yaml(y)
        result[y] = paths

    return result


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="LeRobot Dataset Merger (No Feature Change)")
    ap.add_argument(
        "--yaml_path",
        required=True,
        help="Root dir containing yaml files (each yaml lists multiple dataset folders)",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output root dir; each yaml will produce one merged dataset under a subfolder",
    )
    ap.add_argument("--num_workers", type=int, default=None, help="Parallel workers")
    ap.add_argument(
        "--copy_images", action="store_true", help="Copy images directory (optional)"
    )
    ap.add_argument(
        "--no_copy_videos", action="store_true", help="Do not copy videos"
    )
    ap.add_argument(
        "--strict_features_match",
        action="store_true",
        help="Fail if feature shapes differ across sources",
    )

    args = ap.parse_args()

    all_paths = load_all_yaml_paths(args.yaml_path)
    if not all_paths:
        logger.error(f"No yaml files found under {args.yaml_path}")
        return

    os.makedirs(args.output, exist_ok=True)

    for yaml_file, sources in all_paths.items():
        if not sources:
            logger.warning(f"No dataset paths found in YAML {yaml_file}; skip.")
            continue

        subdir_name = Path(yaml_file).stem
        out_dir = os.path.join(args.output, subdir_name)

        logger.info(
            f"=== Merging {len(sources)} datasets for YAML {yaml_file} "
            f"into {out_dir} ==="
        )

        merger = DatasetMergerNoFeatureChange(
            source_folders=sources,
            output_folder=out_dir,
            num_workers=args.num_workers,
            copy_images=args.copy_images,
            copy_videos=not args.no_copy_videos,
            strict_features_match=args.strict_features_match,
        )

        try:
            merger.merge()
            logger.info(f"Merge for {yaml_file} completed successfully.")
        except Exception as e:
            logger.error(f"Merge for {yaml_file} failed: {e}")
            traceback.print_exc()

    logger.info("All yaml merges finished.")


if __name__ == "__main__":
    main()
