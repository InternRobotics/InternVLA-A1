#!/usr/bin/env python

import argparse
import hashlib
import multiprocessing as mp
from pathlib import Path

import tqdm
import torch
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transforms.constants import MASK_MAPPING, FEATURE_MAPPING
from lerobot.utils.constants import OBS_STATE, ACTION, HF_LEROBOT_HOME
from lerobot.datasets.utils import write_json


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute (and aggregate) normalization statistics for LeRobot datasets"
    )

    p.add_argument(
        "--action_mode",
        type=str,
        choices=["abs", "delta"],
        required=True,
        help="Action mode used to compute statistics (abs or delta).",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        required=True,
        help="Chunk size used for delta action computation (episodes shorter than chunk_size are skipped).",
    )
    p.add_argument(
        "--repo_ids",
        type=str,
        nargs="+",
        required=True,
        help="One or more LeRobotDataset repo ids (must share the same robot_type and feature schema).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes (repo-level parallelism).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output root directory. If not set, uses HF_LEROBOT_HOME/stats/...",
    )

    return p.parse_args()


class RunningStats:
    """Running stats for vectors: keeps count, mean, mean_sq, min, max."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None

    def update(self, batch: torch.Tensor) -> None:
        batch = batch.to(torch.float32)

        if batch.ndim == 1:
            batch = batch[:, None]
        if batch.ndim > 1:
            batch = batch.reshape(-1, batch.shape[-1])

        count = batch.shape[0]
        mean = batch.mean(dim=0)
        mean_sq = (batch ** 2).mean(dim=0)
        min_ = batch.min(dim=0).values
        max_ = batch.max(dim=0).values

        if self._count == 0:
            self._count = count
            self._mean = mean
            self._mean_of_squares = mean_sq
            self._min = min_
            self._max = max_
            return

        total = self._count + count
        w_old = self._count / total
        w_new = count / total

        self._mean = w_old * self._mean + w_new * mean
        self._mean_of_squares = w_old * self._mean_of_squares + w_new * mean_sq
        self._min = torch.minimum(self._min, min_)
        self._max = torch.maximum(self._max, max_)
        self._count = total

    def merge(self, other: "RunningStats") -> None:
        """Merge another RunningStats (exact for mean/mean_sq/min/max)."""
        if other._count == 0:
            return
        if self._count == 0:
            self._count = other._count
            self._mean = other._mean.clone()
            self._mean_of_squares = other._mean_of_squares.clone()
            self._min = other._min.clone()
            self._max = other._max.clone()
            return

        total = self._count + other._count
        w_self = self._count / total
        w_other = other._count / total

        self._mean = w_self * self._mean + w_other * other._mean
        self._mean_of_squares = w_self * self._mean_of_squares + w_other * other._mean_of_squares
        self._min = torch.minimum(self._min, other._min)
        self._max = torch.maximum(self._max, other._max)
        self._count = total

    def to_payload(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        if self._count == 0:
            # Keep empty stats explicit
            return {
                "count": 0,
                "mean": None,
                "mean_sq": None,
                "min": None,
                "max": None,
            }
        return {
            "count": int(self._count),
            "mean": self._mean.detach().cpu().tolist(),
            "mean_sq": self._mean_of_squares.detach().cpu().tolist(),
            "min": self._min.detach().cpu().tolist(),
            "max": self._max.detach().cpu().tolist(),
        }

    @staticmethod
    def from_payload(p: dict) -> "RunningStats":
        rs = RunningStats()
        if p["count"] == 0:
            return rs
        rs._count = int(p["count"])
        rs._mean = torch.tensor(p["mean"], dtype=torch.float32)
        rs._mean_of_squares = torch.tensor(p["mean_sq"], dtype=torch.float32)
        rs._min = torch.tensor(p["min"], dtype=torch.float32)
        rs._max = torch.tensor(p["max"], dtype=torch.float32)
        return rs

    def get_statistics(self) -> dict:
        """Return mean, std, min, max, count."""
        if self._count == 0:
            raise ValueError("No data has been added yet.")
        var = self._mean_of_squares - self._mean ** 2
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return {
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "mean": self._mean.tolist(),
            "std": std.tolist(),
            "count": [int(self._count)],
        }


def _compute_one_repo(repo_id: str, action_mode: str, chunk_size: int) -> dict:
    """Worker: compute stats for one repo, return serializable payload."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = LeRobotDataset(repo_id)
    robot_type = dataset.meta.robot_type

    mask = MASK_MAPPING[robot_type]
    mapping = FEATURE_MAPPING[robot_type]

    keys = list(dataset.meta.features.keys())
    for k in dataset.meta.video_keys + dataset.meta.image_keys:
        if k in keys:
            keys.remove(k)

    # Capture schema for consistency checks
    shapes = {k: dataset.meta.features[k]["shape"] for k in keys}

    stats = {k: RunningStats() for k in keys}
    total_frames = 0
    skipped_episodes = 0

    from_ids = np.asarray(dataset.meta.episodes["dataset_from_index"])
    to_ids = np.asarray(dataset.meta.episodes["dataset_to_index"])
    total_episodes = dataset.num_episodes

    for from_idx, to_idx in zip(from_ids, to_ids):
        ep_len = int(to_idx - from_idx)
        total_frames += ep_len

        if ep_len < chunk_size:
            skipped_episodes += 1
            continue

        curr_episode = dataset.hf_dataset.select(np.arange(from_idx, to_idx))

        # Non-action stats always update; action stats depend on mode
        for key in keys:
            if action_mode == "abs" or key not in mapping[ACTION]:
                val = torch.stack(curr_episode[key][:])
                stats[key].update(val)

        if action_mode == "delta":
            action = [torch.stack(curr_episode[k][:]) for k in mapping[ACTION]]
            action = [a if a.ndim > 1 else a[:, None] for a in action]
            action = torch.cat(action, dim=-1)

            state = [torch.stack(curr_episode[k][:]) for k in mapping[OBS_STATE]]
            state = [s if s.ndim > 1 else s[:, None] for s in state]
            state = torch.cat(state, dim=-1)

            truncated_state = state[0 : (ep_len - chunk_size + 1)]
            action_chunk = action.unfold(dimension=0, size=chunk_size, step=1).permute(0, 2, 1)
            delta_action = action_chunk - torch.where(mask, truncated_state, 0)[:, None]

            sid, eid = 0, 0
            for action_key in mapping[ACTION]:
                eid += dataset.meta.features[action_key]["shape"][0]
                stats[action_key].update(delta_action[..., sid:eid])
                sid = eid

    payload = {k: stats[k].to_payload() for k in keys}

    return {
        "repo_id": repo_id,
        "robot_type": robot_type,
        "keys": keys,
        "shapes": shapes,
        "payload": payload,
        "total_frames": int(total_frames),
        "skipped_episodes": int(skipped_episodes),
        "total_episodes": int(total_episodes),
    }


def _make_group_name(repo_ids: list[str]) -> str:
    """Short stable name for a repo set."""
    joined = "|".join(repo_ids)
    h = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return f"agg_{len(repo_ids)}repos_{h}"


def _normalize_visual_stats(visual_stats: dict) -> dict:
    """Convert numpy/torch entries to lists."""
    out = {}
    for k, v in visual_stats.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy().tolist()
        else:
            out[k] = v
    return out


def compute_norm_stats_multi(cfg):
    repo_ids = cfg.repo_ids
    action_mode = cfg.action_mode
    chunk_size = cfg.chunk_size

    print(f"---------- aggregate stats for {len(repo_ids)} datasets ----------")
    for rid in repo_ids:
        print(f"  - {rid}")

    # Repo-level parallelism
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=cfg.num_workers) as pool:
        results = list(
            tqdm.tqdm(
                pool.starmap(
                    _compute_one_repo,
                    [(rid, action_mode, chunk_size) for rid in repo_ids],
                ),
                total=len(repo_ids),
                desc="Computing per-repo stats",
            )
        )

    # Consistency checks
    robot_types = {r["robot_type"] for r in results}
    if len(robot_types) != 1:
        raise ValueError(f"repo_ids must share the same robot_type, got: {sorted(robot_types)}")
    robot_type = results[0]["robot_type"]

    keys0 = results[0]["keys"]
    shapes0 = results[0]["shapes"]
    for r in results[1:]:
        if r["keys"] != keys0:
            raise ValueError(f"Feature keys mismatch between repos: {results[0]['repo_id']} vs {r['repo_id']}")
        if r["shapes"] != shapes0:
            raise ValueError(f"Feature shapes mismatch between repos: {results[0]['repo_id']} vs {r['repo_id']}")

    # Merge numeric stats
    global_stats = {k: RunningStats() for k in keys0}
    total_frames = 0
    total_episodes = 0
    skipped_episodes = 0

    for r in results:
        total_frames += r["total_frames"]
        total_episodes += r["total_episodes"]
        skipped_episodes += r["skipped_episodes"]
        for k in keys0:
            tmp = RunningStats.from_payload(r["payload"][k])
            global_stats[k].merge(tmp)

    output_dict = {k: global_stats[k].get_statistics() for k in keys0}

    # Visual stats: take from the first repo for simplicity
    first_ds = LeRobotDataset(repo_ids[0])
    for k in first_ds.meta.video_keys + first_ds.meta.image_keys:
        output_dict[k] = _normalize_visual_stats(first_ds.meta.stats[k])

    # Output path
    group_name = _make_group_name(repo_ids)
    if cfg.output_dir:
        out_root = Path(cfg.output_dir)
    else:
        out_root = HF_LEROBOT_HOME / "stats"

    output_dir = out_root / robot_type / action_mode / group_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dict, output_dir / "stats.json")

    print("---------- done ----------")
    print(f"robot_type: {robot_type}")
    print(f"action_mode: {action_mode}")
    print(f"chunk_size: {chunk_size}")
    print(f"group_name: {group_name}")
    print(f"output: {output_dir / 'stats.json'}")
    print(f"total_frames (sum of episode lengths): {total_frames}")
    print(f"total_episodes: {total_episodes} (skipped: {skipped_episodes} episodes with len < chunk_size)")


if __name__ == "__main__":
    compute_norm_stats_multi(parse_args())
