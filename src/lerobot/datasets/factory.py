#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from pprint import pformat
from pathlib import Path

import math
import torch
import torch.distributed as dist

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.utils import load_json, cast_stats_to_numpy
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transformed_dataset import (
    TransformedLeRobotDataset, 
    TransformedStreamingLeRobotDataset, 
    MultiLeRobotDataset, 
    MultiStreamingLeRobotDataset, 
)
from lerobot.transforms.constants import FEATURE_MAPPING, IMAGE_MAPPING
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD, OBS_STATE
from lerobot.utils.constants import HF_LEROBOT_HOME

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def get_rank_and_world_size() -> tuple[int, int]:
    """Get the global rank and world_size.

    If torch.distributed is not initialized, fall back to (0, 1).
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def assign_repo_ids_for_rank(
    repo_ids: list[str],
    rank: int,
    world_size: int,
) -> list[str]:
    """
    Given the global list of repo_ids and a global rank, return the subset of
    repo_ids that this rank should load.

    Rules:
    - If the number of repo_ids >= world_size:
        Perform a disjoint contiguous split so that each rank gets a unique,
        non-overlapping chunk.
    - If the number of repo_ids < world_size:
        Allow repetition and assign repo_ids in a round-robin (modulo) manner,
        ensuring every rank receives at least one repo_id and no rank is empty.
    """
    n = len(repo_ids)
    if n == 0:
        raise ValueError("assign_repo_ids_for_rank: repo_ids is empty.")

    # Case 1: Enough repo_ids to distribute without overlap.
    if n >= world_size:
        return [rid for i, rid in enumerate(repo_ids) if i % world_size == rank]

    # Case 2: Fewer repo_ids than ranks → use round-robin assignment.
    idx = rank % n
    return [repo_ids[idx]]


def find_info_json_path_for_repo(cfg: TrainPipelineConfig, repo_id: str) -> Path | None:
    if cfg.dataset.root is not None: 
        root = Path(cfg.dataset.root)
        return root / repo_id / "meta" / "info.json"
    else:
        return HF_LEROBOT_HOME / repo_id / "meta" / "info.json"


def load_info_for_repos(
    cfg: TrainPipelineConfig,
    repo_ids: list[str],
) -> dict[str, int]:
    frames_map: dict[str, int] = {}
    episodes_map: dict[str, int] = {}

    for rid in repo_ids:
        info_path = find_info_json_path_for_repo(cfg, rid)
        info = load_json(info_path)
        frames_map[rid] = int(info["total_frames"])
        episodes_map[rid] = int(info["total_episodes"])

    return frames_map, episodes_map


def compute_balanced_repo_assignment(
    repo_ids: list[str],
    frames_map: dict[str, int],
    world_size: int,
) -> list[list[str]]:
    """
    Compute a balanced assignment of repo_ids to ranks based on total_frames.

    Goals:
    - Every rank gets at least one repo_id.
    - The total_frames sum per rank is as close as possible across ranks.
    - For len(repo_ids) >= world_size:
        * Each repo_id is used at most once (no duplication).
    - For len(repo_ids) < world_size:
        * Allow duplication of repo_ids across ranks to avoid empty ranks.

    Strategy:
    - Use a greedy LPT-style algorithm:
        1) Sort repo_ids by descending total_frames (ties broken by repo_id string).
        2) Repeatedly assign the next repo_id to the rank with the smallest current load.
        3) If len(repo_ids) < world_size, keep cycling over repo_ids until we have
           assigned at least one repo_id to every rank.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive.")

    n = len(repo_ids)
    if n == 0:
        raise ValueError("compute_balanced_repo_assignment: repo_ids is empty.")

    # Initialize per-rank containers
    rank_to_repos: list[list[str]] = [[] for _ in range(world_size)]
    rank_loads: list[int] = [0 for _ in range(world_size)]

    # Sort repo_ids by descending total_frames, tie-breaker by repo_id for determinism
    def frames_key(rid: str) -> tuple[int, str]:
        # Use negative so that larger total_frames come first
        return (-frames_map.get(rid, 0), rid)

    sorted_repos = sorted(repo_ids, key=frames_key)

    if n >= world_size:
        # Case A: enough repos to avoid duplication.
        # Greedy: always assign the next repo to the rank with the smallest current load.
        for rid in sorted_repos:
            min_rank = min(range(world_size), key=lambda r: (rank_loads[r], r))
            rank_to_repos[min_rank].append(rid)
            rank_loads[min_rank] += frames_map.get(rid, 0)
    else:
        # Case B: fewer repos than ranks -> we must allow duplication
        #
        # Strategy:
        # - Still use LPT greedily, but we keep "expanding" sorted_repos in cycles
        #   until every rank has at least one repo.
        # - This keeps total_frames per rank roughly balanced, even with repetition.
        assignments_done = 0
        idx = 0
        while assignments_done < world_size:
            rid = sorted_repos[idx % n]
            min_rank = min(range(world_size), key=lambda r: (rank_loads[r], r))
            rank_to_repos[min_rank].append(rid)
            rank_loads[min_rank] += frames_map.get(rid, 0)

            assignments_done += 1
            idx += 1

    logging.info(
            f"total_frames={sum(rank_loads)}"
        )
    for r in range(world_size):
        logging.info(
            f"[dist_loading] rank {r}: "
            f"num_frames={rank_loads[r]}"
        )

    return rank_to_repos


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        elif key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        elif key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]
        elif key in FEATURE_MAPPING[ds_meta.robot_type][ACTION] and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        
        if key in IMAGE_MAPPING[ds_meta.robot_type].keys() and hasattr(cfg, "image_delta_indices") and cfg.image_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.image_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def _build_single_dataset(
    cfg: TrainPipelineConfig,
    repo_id: str,
    image_transforms,
    seed_offset: int, 
):
    """
    Build one dataset (single robot) including:
    - metadata
    - delta timestamps
    - LeRobotDataset (or streaming version)
    - ImageNet stats substitution
    - external stats loading (if enabled)
    - TransformedLeRobotDataset wrapping

    Returns:
        transformed_dataset,
        stats_copy,
        robot_type
    """

    # Load metadata + determine delta timestamps
    ds_meta = LeRobotDatasetMetadata(
        repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    if cfg.dataset.streaming:
        root = cfg.dataset.root if cfg.dataset.root is not None else HF_LEROBOT_HOME / repo_id
        base_ds = StreamingLeRobotDataset(
            repo_id=repo_id,
            root=root,          
            episodes=cfg.dataset.episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            revision=cfg.dataset.revision,
            force_cache_sync=False,         
            streaming=True,                 
            buffer_size=cfg.dataset.buffer_size,               
            max_num_shards=cfg.num_workers,
            seed=cfg.seed + seed_offset,
            rng=None,
            shuffle=True,
        )
        transformed_ds = TransformedStreamingLeRobotDataset.from_base(
            base_ds,
            cfg.dataset.data_transforms.inputs,
        )
    else:

        # Create the actual LeRobot dataset (non-streaming recommended for multi-robot)
        base_ds = LeRobotDataset(
            repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            # tolerance_s=0.0002, 
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
        )
        transformed_ds = TransformedLeRobotDataset.from_base(
            base_ds,
            cfg.dataset.data_transforms.inputs,
        )

    # Optional: override stats using ImageNet norm
    if cfg.dataset.use_imagenet_stats:
        for key in base_ds.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                base_ds.meta.stats[key][stats_type] = torch.tensor(
                    stats, dtype=torch.float32
                )

    robot_type = base_ds.meta.robot_type

    # Optional: load aggregated external stats
    if cfg.dataset.use_external_stats:
        if cfg.dataset.external_stats_path is not None:
            stat_path = Path(cfg.dataset.external_stats_path)
        else:
            action_mode = cfg.dataset.action_mode
            stat_path = HF_LEROBOT_HOME / f"stats/{robot_type}/{action_mode}/stats.json"
        # stat_path = HF_LEROBOT_HOME / f"stats/{robot_type}/{action_mode}/{repo_id}/stats.json"

        if stat_path.exists():
            ext_stats = cast_stats_to_numpy(load_json(stat_path))
            logging.info(f"Using external stats from {stat_path}")
            base_ds.meta.stats.update(ext_stats)
        else:
            raise FileNotFoundError(
                f"use_external_stats=True but no file found at {stat_path}."
            )

    stats_copy = base_ds.meta.stats.copy()

    return transformed_ds, stats_copy, robot_type


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | StreamingLeRobotDataset | MultiLeRobotDataset | MultiStreamingLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    all_data_stats = {}
    all_repo_ids = cfg.dataset.repo_id.split(' ')
    logging.info(
        f"[make_dataset] all_repo_ids={all_repo_ids}"
    )
    rank, world_size = get_rank_and_world_size()
    if cfg.dataset.dist_loading:
        # Try to balance by total_frames first.
        try:
            frames_map, episodes_map = load_info_for_repos(cfg, all_repo_ids)
            rank_to_repos = compute_balanced_repo_assignment(
                all_repo_ids,
                frames_map,
                world_size,
            )
            repo_ids = rank_to_repos[rank]
            logging.info(
                f"[make_dataset] dist_loading=True, using total_frames-balanced "
                f"assignment."
            )
        except Exception as e:
            # Fallback to the simple deterministic assignment
            logging.warning(
                f"[make_dataset] total_frames-based balancing failed with error: {e}. "
                "Falling back to simple rank-based assignment."
            )
            repo_ids = assign_repo_ids_for_rank(all_repo_ids, rank, world_size)
    else:
        repo_ids = all_repo_ids

    print(
        f"rank={rank}/{world_size}, repo_ids_for_this_rank:\n"
        + "\n".join(f"[rank {rank}] repo_id = {rid}" for rid in repo_ids)
    )

    if len(repo_ids) == 1:
        repo_id = repo_ids[0]

        transformed_ds, stats_copy, robot_type = _build_single_dataset(
            cfg, repo_id, image_transforms, rank
        )
        all_data_stats[robot_type] = stats_copy

        return transformed_ds, all_data_stats
    
    transformed_datasets = []

    for rid, repo_id in enumerate(repo_ids):
        transformed_ds, stats_copy, robot_type = _build_single_dataset(
            cfg, repo_id, image_transforms, rank * 128 + rid
        )
        transformed_datasets.append(transformed_ds)
        all_data_stats[robot_type] = stats_copy  # TODO: If multiple repos share robot_type, last one overwrites.

    if not cfg.dataset.streaming:
        multi_ds = MultiLeRobotDataset(transformed_datasets)
    else:
        multi_ds = MultiStreamingLeRobotDataset(transformed_datasets, seed=cfg.seed)

    return multi_ds, all_data_stats
