#!/usr/bin/env python

import logging
import tqdm
import argparse
from pathlib import Path

import torch
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transforms.constants import MASK_MAPPING, FEATURE_MAPPING
from lerobot.utils.constants import OBS_STATE, ACTION, HF_LEROBOT_HOME
from lerobot.datasets.utils import write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for LeRobot datasets"
    )

    parser.add_argument(
        "--action_mode",
        type=str,
        choices=["abs", "delta"],
        required=True,
        help="Action mode used to compute statistics (abs or delta).",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        required=True,
        help="Chunk size used for delta action computation.",
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="LeRobotDataset repo id.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. If not set, use default HF_LEROBOT_HOME path.",
    )

    return parser.parse_args()


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None

    def update(self, batch: torch.Tensor) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            batch (torch.Tensor): shape [..., d]
        """
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
        else:
            total = self._count + count
            w_old = self._count / total
            w_new = count / total

            self._mean = w_old * self._mean + w_new * mean
            self._mean_of_squares = w_old * self._mean_of_squares + w_new * mean_sq
            self._min = torch.minimum(self._min, min_)
            self._max = torch.maximum(self._max, max_)
            self._count = total

    def get_statistics(self) -> dict:
        """Return mean, std, min, max."""
        if self._count == 0:
            raise ValueError("No data has been added yet.")
        var = self._mean_of_squares - self._mean ** 2
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return {
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "mean": self._mean.tolist(),
            "std": std.tolist(),
            "count": [self._count],
        }


def compute_norm_stats(cfg):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"---------- compute statistics for dataset: {cfg.repo_id} ----------")
    dataset = LeRobotDataset(cfg.repo_id)

    from_ids = np.asarray(dataset.meta.episodes['dataset_from_index'])
    to_ids = np.asarray(dataset.meta.episodes['dataset_to_index'])
    total_episodes = dataset.num_episodes
    
    action_mode = cfg.action_mode
    chunk_size = cfg.chunk_size
    robot_type = dataset.meta.robot_type

    mask = MASK_MAPPING[robot_type]
    mapping = FEATURE_MAPPING[robot_type]

    keys = list(dataset.meta.features.keys())
    [keys.remove(video_key) for video_key in dataset.meta.video_keys]
    [keys.remove(image_key) for image_key in dataset.meta.image_keys]
    stats = {key: RunningStats() for key in keys}

    total_frame = 0

    for from_idx, to_idx in tqdm.tqdm(zip(from_ids, to_ids), total=total_episodes, desc="Computing stats"):
        ep_len = to_idx - from_idx
        total_frame += ep_len
        if ep_len < chunk_size:
            continue
        curr_episode = dataset.hf_dataset.select(np.arange(from_idx, to_idx))
        for key in keys:
            if action_mode == 'abs' or key not in mapping[ACTION]:
                val = torch.stack(curr_episode[key][:])
                stats[key].update(val)
        if action_mode == 'delta':
            action = [torch.stack(curr_episode[key][:]) for key in mapping[ACTION]]
            action = [a if a.ndim > 1 else a[:, None] for a in action]
            action = torch.cat(action, dim=-1)
            state = [torch.stack(curr_episode[key][:]) for key in mapping[OBS_STATE]]
            state = [s if s.ndim > 1 else s[:, None] for s in state]
            state = torch.cat(state, dim=-1)
            truncated_state = state[0:(ep_len - chunk_size + 1)]
            action_chunk = action.unfold(dimension=0, size=chunk_size, step=1).permute(0, 2, 1)
            delta_action = action_chunk - torch.where(mask, truncated_state, 0)[:, None]
            sid, eid = 0, 0
            for action_key in mapping[ACTION]:
                eid += dataset.meta.features[action_key]['shape'][0]
                stats[action_key].update(delta_action[..., sid:eid])
                sid = eid
        
    output_dict = {key: stats[key].get_statistics() for key in keys}
    for key in dataset.meta.video_keys + dataset.meta.image_keys:
        dataset.meta.stats[key]
        visual_stats = dataset.meta.stats[key]
        for stat_key in visual_stats:
            if isinstance(visual_stats[stat_key], np.ndarray):
                visual_stats[stat_key] = visual_stats[stat_key].tolist()
            elif isinstance(visual_stats[stat_key], torch.Tensor):
                visual_stats[stat_key] = visual_stats[stat_key].cpu().numpy().tolist()
        output_dict[key] = visual_stats
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)/action_mode/cfg.repo_id
    else:
        output_dir = HF_LEROBOT_HOME/'stats'/action_mode/cfg.repo_id
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    write_json(output_dict, output_dir/"stats.json")

    print(f"total_frame: {total_frame}")


if __name__ == "__main__":
    compute_norm_stats(parse_args())
