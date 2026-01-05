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
from collections.abc import Iterator

import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Iterator, List, Optional

from lerobot.datasets.transformed_dataset import MultiLeRobotDataset


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


class DistributedEpisodeAwareSampler(Sampler[int]):
    def __init__(
        self,
        dataset_from_indices: List[int],
        dataset_to_indices: List[int],
        episode_indices_to_use: Optional[List[int]] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
    ):
        """Distributed sampler that incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            drop_last: Whether to drop tail samples so the number of samples is divisible by world size.
            seed: Random seed for shuffling, ensures different order across epochs when combined with set_epoch.
        """
        # --- build flat indices respecting episode boundaries ---
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                lo = start_index + drop_n_first_frames
                hi = end_index - drop_n_last_frames
                if hi > lo:
                    indices.extend(range(lo, hi))

        self._base_indices = indices
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # --- distributed info ---
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        indices = self._base_indices

        # 1) global shuffle (all ranks use the same permutation)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]

        # 2) align to world_size (pad or truncate)
        if self.drop_last:
            # truncate so that the length is divisible by world_size
            n = (len(indices) // self.world_size) * self.world_size
            indices = indices[:n]
        else:
            # pad by repeating from the beginning until divisible
            pad = (-len(indices)) % self.world_size
            if pad:
                indices = indices + indices[:pad]

        # 3) shard by rank (round-robin split)
        per_rank_indices = indices[self.rank::self.world_size]
        # print(f"[rank {self.rank}/{self.world_size}] first indices: {per_rank_indices[:10]}")
        return iter(per_rank_indices)

    def __len__(self) -> int:
        n = len(self._base_indices)
        if self.world_size <= 1:
            return n
        if self.drop_last:
            return n // self.world_size
        else:
            return math.ceil(n / self.world_size)


class MultiLeRobotWeightedSampler(Sampler[int]):
    """
    A sampler that chooses which underlying dataset to sample from according
    to `dataset.dataset_weights`, then uniformly samples a frame inside it,
    and finally returns the corresponding global index.

    This keeps the Dataset logic simple and delegates sampling strategy here.
    """

    def __init__(
        self,
        dataset: MultiLeRobotDataset,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if not isinstance(dataset, MultiLeRobotDataset):
            raise TypeError("MultiLeRobotWeightedSampler requires a MultiLeRobotDataset.")

        self.dataset = dataset
        self.replacement = replacement
        self.generator = generator

        # Number of indices the sampler will produce per epoch
        self.num_samples = num_samples if num_samples is not None else len(dataset)

        # Cache for convenience
        self._lengths = dataset._lengths
        self._cum_lengths = dataset._cum_lengths
        self._weights = dataset.dataset_weights

    def __iter__(self) -> Iterator[int]:
        g = self.generator if self.generator is not None else torch.Generator()
        # Ensure some randomness (you can also set externally for reproducibility)
        if self.generator is None:
            g.manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())

        for _ in range(self.num_samples):
            # 1) Sample dataset index according to weights
            ds_idx = torch.multinomial(self._weights, 1, replacement=True, generator=g).item()

            # 2) Sample a local index uniformly from that dataset
            local_len = self._lengths[ds_idx]
            local_idx = torch.randint(high=local_len, size=(1,), generator=g).item()

            # 3) Convert to global index
            global_start = 0 if ds_idx == 0 else self._cum_lengths[ds_idx - 1]
            global_idx = global_start + local_idx

            yield global_idx

    def __len__(self) -> int:
        return self.num_samples
