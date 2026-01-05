from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional

import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.transforms.core import (
    DataTransformFn, 
    DataDict, 
    compose, 
    hydrate_normalize_transform, 
    hydrate_compose_field_transform, 
    hydrate_delta_action_transform, 
    hydrate_remap_image_key_transform, 
    filter_image_features, 
)


class TransformedLeRobotDataset(LeRobotDataset):

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use TransformedLeRobotDataset.from_base(...) or .from_repo(...).")

    @classmethod
    def from_base(
        cls,
        base: LeRobotDataset,
        transforms: Sequence[DataTransformFn] | None = None,
        *,
        share_dict: bool = True,
    ) -> TransformedLeRobotDataset:
        obj = cls.__new__(cls)  
        obj.__dict__ = (base.__dict__ if share_dict else base.__dict__.copy())

        transforms = hydrate_normalize_transform(transforms, obj)
        transforms = hydrate_compose_field_transform(transforms, obj)
        transforms = hydrate_delta_action_transform(transforms, obj)
        transforms = hydrate_remap_image_key_transform(transforms, obj)
        filter_image_features(obj)

        obj._transform = compose(transforms)
        obj._wrapped_base_cls = base.__class__.__name__
        obj._is_transformed_wrapper = True
        return obj

    @classmethod
    def from_repo(
        cls,
        repo_id: str,
        *,
        root=None,
        episodes=None,
        image_transforms=None,
        delta_timestamps=None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        transforms: Sequence[DataTransformFn] | None = None,
        share_dict: bool = True,
    ) -> TransformedLeRobotDataset:
        base = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
        )
        return cls.from_base(base, transforms=transforms, share_dict=share_dict)

    def __getitem__(self, idx: int) -> DataDict:
        sample = super().__getitem__(idx)
        return self._transform(sample)

    def __repr__(self) -> str:
        base = super().__repr__().rstrip("\n")
        return base + f" (Transformed from {getattr(self, '_wrapped_base_cls', 'LeRobotDataset')})\n"


@dataclass
class CombinedMeta:
    """Lightweight metadata container for MultiLeRobotDataset.

    Only stores what we actually need: flattened episode_from/to indices.
    """
    episodes: Dict[str, List[int]]


class MultiLeRobotDataset(Dataset):
    """
    A concatenation dataset that merges multiple TransformedLeRobotDataset instances.

    Assumptions:
    - Each underlying dataset already applies its own DataTransformFn pipeline.
    - All datasets return aligned keys (i.e., they are transform-aligned).
    
    This class only:
    - Locates which sub-dataset corresponds to a global index.
    - Calls the appropriate __getitem__.
    - Optionally attaches metadata such as dataset_index and repo_id.

    It is intentionally minimal so it can be used directly inside a PyTorch DataLoader.
    """

    def __init__(
        self, 
        datasets: Sequence[TransformedLeRobotDataset], 
        dataset_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()

        if not datasets:
            raise ValueError("MultiLeRobotDataset requires at least one dataset.")

        # List of transformed datasets (one per robot / repo)
        self.datasets = list(datasets)

        # Pre-compute lengths for fast index routing
        self._lengths = [ds.num_frames for ds in self.datasets]

        # Cumulative lengths for O(N_datasets) lookup
        self._cum_lengths = []
        running = 0
        for length in self._lengths:
            running += length
            self._cum_lengths.append(running)
        
        self.meta = self._build_combined_metadata()

        if dataset_weights is None:
            self.dataset_weights = None
            # Default: sample datasets proportional to their lengths
            # w = torch.tensor(self._lengths, dtype=torch.float32)
        else:
            if len(dataset_weights) != len(self.datasets):
                raise ValueError(
                    f"dataset_weights must have length {len(self.datasets)}, "
                    f"got {len(dataset_weights)}."
                )
            w = torch.tensor(dataset_weights, dtype=torch.float32)

            if (w < 0).any():
                raise ValueError("dataset_weights must be non-negative.")

            if w.sum() == 0:
                raise ValueError("At least one dataset weight must be positive.")

            self.dataset_weights = w / w.sum()
    
    @property
    def num_frames(self) -> int:
        """Total number of frames across all robots."""
        return self._cum_lengths[-1]

    @property
    def num_episodes(self) -> int:
        """Total number of episodes across all underlying datasets."""
        return sum(ds.meta.total_episodes for ds in self.datasets)

    def _build_combined_metadata(self) -> CombinedMeta:
        """
        Construct a lightweight metadata object that mimics:
            dataset.meta.episodes["dataset_from_index"]
            dataset.meta.episodes["dataset_to_index"]

        Behavior:
        - Episodes from all robots are concatenated in order:
            robot1 episodes → robot2 episodes → ...
        - Frame indexing is continuous across robots.
        """
        episodes = {
            "dataset_from_index": [],
            "dataset_to_index": [],
        }

        running_frame = 0

        for ds in self.datasets:
            dataset_from_index = np.asarray(ds.meta.episodes["dataset_from_index"]) + running_frame
            dataset_to_index = np.asarray(ds.meta.episodes["dataset_to_index"]) + running_frame

            episodes["dataset_from_index"].extend(dataset_from_index.tolist())
            episodes["dataset_to_index"].extend(dataset_to_index.tolist())

            running_frame += ds.num_frames

        return CombinedMeta(episodes=episodes)


    def __len__(self):
        return self.num_frames

    def _locate_dataset(self, idx: int) -> tuple[int, int]:
        """
        Convert a global index into (dataset_index, local_index).

        Example:
            If dataset lengths = [1000, 800],
            global idx 1200 → dataset #1, local_idx 200.
        """
        if idx < 0:
            idx = len(self) + idx

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range.")

        start = 0
        for ds_idx, length in enumerate(self._lengths):
            end = start + length
            if idx < end:
                return ds_idx, idx - start
            start = end

        # Should not happen
        raise RuntimeError("Index resolution failed in MultiLeRobotDataset._locate_dataset")

    def __getitem__(self, idx: int) -> dict:
        """
        Forward a global index to the correct robot-dataset.
        """
        ds_idx, local_idx = self._locate_dataset(idx)
        ds = self.datasets[ds_idx]

        # Already transformed sample
        sample = ds[local_idx]

        # Add metadata identifying which robot this sample came from
        # sample["dataset_index"] = torch.tensor(ds_idx, dtype=torch.long)
        # sample["repo_id"] = ds.repo_id

        return sample

    def __repr__(self):
        return (
            f"MultiLeRobotDataset(\n"
            f"  Robots: {len(self.datasets)} → {[ds.repo_id for ds in self.datasets]},\n"
            f"  Total frames: {self.num_frames},\n"
            f"  Total episodes: {self.num_episodes}\n"
            f")"
        )


class TransformedStreamingLeRobotDataset(IterableDataset):

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use .from_base(...)")

    @classmethod
    def from_base(
        cls,
        base: StreamingLeRobotDataset,
        transforms: Sequence[DataTransformFn] | None = None,
    ):
        obj = cls.__new__(cls)

        obj._base = base

        obj.meta = base.meta
        obj.stats = base.meta.stats
        obj.robot_type = base.meta.robot_type

        transforms = hydrate_normalize_transform(transforms, obj)
        transforms = hydrate_compose_field_transform(transforms, obj)
        transforms = hydrate_delta_action_transform(transforms, obj)
        transforms = hydrate_remap_image_key_transform(transforms, obj)
        filter_image_features(obj)

        obj._transform = compose(transforms)
        return obj

    def __iter__(self):
        for x in self._base:
            yield self._transform(x)

    @property
    def num_frames(self):
        return self._base.num_frames
    
    @property
    def num_episodes(self):
        return self._base.num_episodes

    @property
    def fps(self):
        return self._base.fps

    @property
    def camera_keys(self):
        return self._base.meta.camera_keys

    def __repr__(self):
        return f"TransformedStreamingLeRobotDataset(base={self._base})"


class MultiStreamingLeRobotDataset(IterableDataset):
    """
    Streaming version of MultiLeRobotDataset.

    This dataset merges multiple StreamingLeRobotDataset or
    TransformedStreamingLeRobotDataset instances into a single
    IterableDataset for large-scale streaming training.

    Notes:
    - This class does NOT support random indexing (__getitem__).
    - It supports two merging modes:
        (1) Simple concatenation (no dataset_weights)
        (2) Weighted multi-stream sampling (with dataset_weights)
    - For extremely large datasets (100M+ frames), this avoids
      loading into memory and fully supports PyTorch multi-worker
      dataloading.
    """

    def __init__(
        self,
        datasets: Sequence[TransformedStreamingLeRobotDataset],
        *,
        dataset_weights: Optional[Sequence[float]] = None,
        seed: int = 42,
        add_dataset_index: bool = True,
    ) -> None:
        super().__init__()

        if not datasets:
            raise ValueError("MultiStreamingLeRobotDataset requires at least one dataset.")

        # Store datasets (each is iterable)
        self.datasets: List[TransformedStreamingLeRobotDataset] = list(datasets)
        self.add_dataset_index = add_dataset_index

        # Aggregate minimal metadata (episode boundaries only)
        self.meta = self._build_combined_metadata()

        # Handle optional dataset sampling weights
        if dataset_weights is None:
            # No sampling weights → simple concatenation
            self.dataset_weights = np.asarray([ds.num_frames for ds in self.datasets])
            self.dataset_weights = self.dataset_weights / self.dataset_weights.sum()
        else:
            if len(dataset_weights) != len(self.datasets):
                raise ValueError(
                    f"dataset_weights must have length {len(self.datasets)}, "
                    f"got {len(dataset_weights)}."
                )

            w = np.asarray(dataset_weights, dtype=np.float64)

            if (w < 0).any():
                raise ValueError("dataset_weights must be non-negative.")
            if w.sum() == 0:
                raise ValueError("At least one dataset weight must be positive.")

            # Normalize weights
            self.dataset_weights = (w / w.sum()).tolist()

        self.seed = seed

    # ----------------------------------------------------------
    # Properties (to match MultiLeRobotDataset)
    # ----------------------------------------------------------
    @property
    def num_frames(self) -> int:
        """Total frames across all datasets (approximate)."""
        return sum(ds.num_frames for ds in self.datasets)

    @property
    def num_episodes(self) -> int:
        """Total number of episodes across datasets."""
        return sum(ds.num_episodes for ds in self.datasets)

    @property
    def fps(self) -> Optional[int]:
        """FPS is assumed consistent across datasets."""
        return self.datasets[0].fps

    def __len__(self):
        """IterableDataset normally has no __len__, but we return num_frames for compatibility."""
        return self.num_frames

    # ----------------------------------------------------------
    # Main Streaming Iterator
    # ----------------------------------------------------------
    def __iter__(self):
        """
            Main streaming logic:

            Case 1 — No dataset_weights:
                Simple sequential concatenation.
                Each dataset is fully exhausted before moving to the next one:

                    ds0 → ds0 → ds0 → ... → ds0
                                            ↓
                    ds1 → ds1 → ds1 → ... → ds1
                                            ↓
                    ds2 → ds2 → ds2 → ... → ds2

            Case 2 — With dataset_weights:
                Weighted multi-stream sampling (with replacement).

                At each step, one dataset is sampled according to dataset_weights,
                and one sample is yielded from that dataset. When a dataset iterator
                is exhausted, it is automatically restarted (wrap-around).

                This produces an effectively infinite mixed stream whose long-run
                sampling frequency converges to the specified weights:

                    step:   1     2     3     4     5     6     ...
                    choice: ds0   ds0   ds1   ds0   ds0   ds1  ...
                            ↑           ↑
                        p≈0.9       p≈0.1

                (Example: dataset_weights = [0.9, 0.1])

                Epoch length is therefore not defined by dataset exhaustion and must
                be controlled externally (e.g. via max_steps or steps_per_epoch).

            Multi-worker support:
                - Each worker uses a different RNG seed for dataset selection.
                - Dataset iterators are independent across workers.
        """
        worker_info = get_worker_info()

        # Each worker uses a different RNG seed for sampling
        if worker_info is None:
            rng_seed = self.seed
        else:
            rng_seed = self.seed + worker_info.id

        rng = np.random.default_rng(rng_seed)

        # Create iterators for each dataset
        iterators = {i: iter(ds) for i, ds in enumerate(self.datasets)}
        active_ids = list(iterators.keys())

        # ------------------------------------------------------
        # 1) Simple concatenation mode
        # ------------------------------------------------------
        if self.dataset_weights is None:
            for ds_idx in active_ids:
                it = iterators[ds_idx]
                for sample in it:
                    if self.add_dataset_index and isinstance(sample, dict):
                        sample["dataset_index"] = torch.tensor(ds_idx, dtype=torch.long)
                    yield sample
            return

        # ------------------------------------------------------
        # 2) Weighted multi-stream merging
        # ------------------------------------------------------
        weights = np.asarray(self.dataset_weights, dtype=np.float64)

        while True:  
            cur_weights = weights / weights.sum()
            ds_idx = rng.choice(len(self.datasets), p=cur_weights)

            it = iterators[ds_idx]
            try:
                sample = next(it)
            except StopIteration:
                it = iter(self.datasets[ds_idx])
                iterators[ds_idx] = it
                sample = next(it) 

            if self.add_dataset_index and isinstance(sample, dict):
                sample["dataset_index"] = torch.tensor(ds_idx, dtype=torch.long)
            yield sample

    # ----------------------------------------------------------
    # Metadata concatenation
    # ----------------------------------------------------------
    def _build_combined_metadata(self) -> CombinedMeta:
        """
        Concatenate dataset episode boundaries into unified metadata.

        This replicates MultiLeRobotDataset's CombinedMeta but for streaming.
        It does NOT store heavy feature definitions—only lightweight indices.

        Behavior:
        - Episodes are concatenated in order:
              ds0 episodes → ds1 episodes → ...
        - Frame indexing becomes continuous across datasets.
        """
        episodes = {
            "dataset_from_index": [],
            "dataset_to_index": [],
        }

        running_frame = 0

        for ds in self.datasets:
            from_index = np.asarray(ds.meta.episodes["dataset_from_index"]) + running_frame
            to_index = np.asarray(ds.meta.episodes["dataset_to_index"]) + running_frame

            episodes["dataset_from_index"].extend(from_index.tolist())
            episodes["dataset_to_index"].extend(to_index.tolist())

            running_frame += ds.num_frames

        return CombinedMeta(episodes=episodes)

    def __repr__(self) -> str:
        return (
            f"MultiStreamingLeRobotDataset(\n"
            f"  Num datasets: {len(self.datasets)},\n"
            f"  Num frames (approx): {self.num_frames},\n"
            f"  Num episodes (approx): {self.num_episodes},\n"
            f"  Has weights: {self.dataset_weights is not None},\n"
            f")"
        )
