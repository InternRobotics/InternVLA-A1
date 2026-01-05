from __future__ import annotations

from typing import Any, Optional, runtime_checkable
from dataclasses import dataclass, field, replace

import abc
import draccus
import logging
import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.transforms.utils import resize_with_pad, resize_center_crop
from lerobot.utils.constants import OBS_IMAGE, OBS_IMAGES, OBS_STATE, ACTION
from .constants import FEATURE_MAPPING, MASK_MAPPING, IMAGE_MAPPING


DataDict = dict[str, Any]


class DataTransformFn(draccus.ChoiceRegistry, abc.ABC):
    @abc.abstractmethod
    def __call__(self, data: DataDict) -> DataDict: ...


@dataclass(frozen=True)
class TransformGroup:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: list[DataTransformFn] = field(default_factory=list)

    # Transforms that are applied to the model output data.
    outputs: list[DataTransformFn] = field(default_factory=list)

    def push(self, 
             *, 
             inputs: list[DataTransformFn] = None, 
             outputs: list[DataTransformFn] = None) -> TransformGroup:
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        if inputs is None: inputs = []
        if outputs is None: outputs = []
        return TransformGroup(
            inputs=[*self.inputs, *inputs],
            outputs=[*outputs, *self.outputs],
        )


@DataTransformFn.register_subclass("composite")
@dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: list[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: list[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@DataTransformFn.register_subclass("identity")
@dataclass(frozen=True)
class IdentityTransformFn(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict: 
        return data
    

@DataTransformFn.register_subclass("pad_state_and_action")
@dataclass
class PadStateAndActionTransformFn(DataTransformFn):
    max_state_dim: int = 32
    max_action_dim: int = 32

    def __call__(self, data: DataDict) -> DataDict: 
        data[OBS_STATE] = self._pad_vector(data[OBS_STATE], self.max_state_dim)
        data[ACTION] = self._pad_vector(data[ACTION], self.max_action_dim)
        return data

    def _pad_vector(self, vector: torch.Tensor, new_dim: int):
        if vector.shape[-1] >= new_dim:
            return vector
        return F.pad(vector, (0, new_dim - vector.shape[-1]))


@DataTransformFn.register_subclass("totensor")
@dataclass
class ToTensorTransformFn(DataTransformFn):
    def __post_init__(self):
        self.img2tensor_fn = torchvision.transforms.ToTensor()
    
    def __call__(self, data: DataDict) -> DataDict: 
        for key in data.keys():
            if key.startswith(OBS_IMAGES) or key == OBS_IMAGE or "image" in key:
                data[key] = self.img2tensor_fn(data[key])
            elif isinstance(data[key], list):
                data[key] = torch.tensor(data[key])
            elif isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(data[key])
        return data


@DataTransformFn.register_subclass("resize_with_pad")
@dataclass
class ResizeImagesWithPadFn(DataTransformFn):
    height: int
    width: int
    mode: str = "bilinear"

    def __call__(self, data: DataDict) -> DataDict:
        
        for k, v in data.items():
            if "is_pad" in k:
                continue
            if k.startswith(OBS_IMAGES) or k == OBS_IMAGE or "image" in k:
                 data[k] = resize_with_pad(v, self.height, self.width, self.mode)
        return data


@DataTransformFn.register_subclass("resize_center_crop")
@dataclass
class ResizeShortestCenterCropFn(DataTransformFn):
    height: int
    width: int
    mode: str = "bilinear"

    def __call__(self, data: DataDict) -> DataDict:
        for k, v in data.items():
            if k.startswith(OBS_IMAGES) or k == OBS_IMAGE or "image" in k:
                data[k] = resize_center_crop(v, self.height, self.width, self.mode)
        return data


@DataTransformFn.register_subclass("compose_fields")
@dataclass
class ComposeFieldsTransform(DataTransformFn):
    """
    Merge multiple keys' values into a single new key.

    Example:
        mapping = {
            "observation.state": [
                "observation.states.joint.position",
                "observation.states.effector.position",
            ]
            "action": [
                "actions.joint.position", 
                "actions.effector.position", 
            ]
        }
    """
    mapping: dict[str, list[str]] = field(default_factory=dict)

    def __call__(self, data: DataDict) -> DataDict:
        for new_key, src_keys in self.mapping.items():
            if len(src_keys) == 1 and src_keys[0] == new_key:
                continue
            # Concatenate along the last dimension
            merge_list = self._align_for_cat([data[k] for k in src_keys])
            merged = torch.cat(merge_list, dim=-1)
            data[new_key] = merged
            for k in src_keys: data.pop(k, None)
        return data
    
    def _align_for_cat(self, tensors: list[torch.Tensor], dim=-1) -> list[torch.Tensor]:
        max_ndim = max((t.ndim for t in tensors))
        out = []
        for t in tensors:
            t = t if t.ndim == max_ndim else t.unsqueeze(dim)
            out.append(t)
        return out


@DataTransformFn.register_subclass("remap_image_key")
@dataclass
class RemapImageKeyTransformFn(DataTransformFn):
    """
    Remap image keys to new key names.
    Example:
        mapping = {
            "images.rgb.head": f"{OBS_IMAGES}.image0", 
            "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
            "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
        }
    """
    mapping: dict[str, str] = field(default_factory=dict)

    def __call__(self, data: DataDict) -> DataDict: 
        for old_key, new_key in self.mapping.items():
            data[new_key] = data.pop(old_key)
            data[f"{new_key}_mask"] = torch.tensor(True)
        # create missing keys if necessary
        if len(self.mapping) < 3:
            data[f"{OBS_IMAGES}.image2"] = torch.ones_like(data[f"{OBS_IMAGES}.image0"])
            data[f"{OBS_IMAGES}.image2_mask"] = torch.tensor(False)
        if len(self.mapping) < 2:
            data[f"{OBS_IMAGES}.image1"] = torch.ones_like(data[f"{OBS_IMAGES}.image0"])
            data[f"{OBS_IMAGES}.image1_mask"] = torch.tensor(False)
        return data


@DataTransformFn.register_subclass("normalize")
@dataclass
class NormalizeTransformFn(DataTransformFn):
    """
    Normalize specified keys in a DataDict using precomputed statistics.

    Args:
        selected_keys: list of keys to normalize (e.g. ["observation.state", "actions"]).
            If None, will normalize all keys that exist in norm_stats.
        mode: normalization mode ("mean_std" or "min_max").
        norm_stats: dictionary containing normalization parameters.

    Example:
        norm_stats = {
            "observation.state": {"mean": ..., "std": ..., "min": ..., "max": ...},
            "action": {"mean": ..., "std": ..., "min": ..., "max": ...},
        }
    """

    selected_keys: Optional[list[str]] = None
    mode: str = "mean_std"  # "mean_std" or "min_max"
    norm_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __call__(self, data: DataDict) -> DataDict:
        eps = 1e-6

        keys = self.selected_keys if self.selected_keys is not None else list(self.norm_stats.keys())

        for key in keys:
            if key not in data:
                logging.warning(
                    f"[NormalizeTransformFn] Key '{key}' not found in data — skipping normalization."
                )
                continue
            if key not in self.norm_stats:
                logging.warning(
                    f"[NormalizeTransformFn] No normalization stats found for key '{key}' — skipping."
                )
                continue

            x = data[key]
            stats = self.norm_stats[key]

            if self.mode == "mean_std":
                mean = torch.from_numpy(stats["mean"]).to(x)
                std = torch.from_numpy(stats["std"]).to(x)
                x = ((x - mean) / (std + eps))
            elif self.mode == "min_max":
                min_v = torch.from_numpy(stats["min"]).to(x)
                max_v = torch.from_numpy(stats["max"]).to(x)
                x = (x - min_v) / (max_v - min_v + eps)
            else:
                raise ValueError(f"Unknown normalization mode: {self.mode}")

            data[key] = x
        return data
    

@DataTransformFn.register_subclass("unnormalize")
@dataclass
class UnNormalizeTransformFn(DataTransformFn):
    """
    Unnormalize specified keys in a DataDict using precomputed statistics.

    Args:
        selected_keys: list of keys to unnormalize (e.g. ["observation.state", "actions"]).
            If None, will unnormalize all keys that exist in norm_stats.
        mode: unnormalization mode ("mean_std" or "min_max").
        norm_stats: dictionary containing unnormalization parameters.

    Example:
        norm_stats = {
            "observation.state": {"mean": ..., "std": ..., "min": ..., "max": ...},
            "action": {"mean": ..., "std": ..., "min": ..., "max": ...},
        }
    """

    selected_keys: Optional[list[str]] = None
    mode: str = "mean_std"  # "mean_std" or "min_max"
    norm_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __call__(self, data: DataDict) -> DataDict:
        eps = 1e-6

        keys = self.selected_keys if self.selected_keys else list(self.norm_stats.keys())

        for key in keys:
            if key not in data:
                logging.warning(
                    f"[UnNormalizeTransformFn] Key '{key}' not found in data — skipping unnormalization."
                )
                continue
            if key not in self.norm_stats:
                logging.warning(
                    f"[UnNormalizeTransformFn] No stats found for key '{key}' — skipping unnormalization."
                )
                continue

            x = data[key]
            stats = self.norm_stats[key]

            if self.mode == "mean_std":
                mean = torch.from_numpy(stats["mean"]).to(x)
                std = torch.from_numpy(stats["std"]).to(x)
                x = x * (std + eps) + mean
            elif self.mode == "min_max":
                min_v = torch.from_numpy(stats["min"]).to(x)
                max_v = torch.from_numpy(stats["max"]).to(x)
                x = x * (max_v - min_v + eps) + min_v
            else:
                raise ValueError(f"Unknown unnormalization mode: {self.mode}")

            data[key] = x

        return data


@DataTransformFn.register_subclass("delta_action")
@dataclass
class DeltaActionTransformFn(DataTransformFn):

    mask: Optional[list[bool]] = None
    mapping: dict[str, list[str]] = field(default_factory=dict)

    def __call__(self, data: DataDict) -> DataDict:
        # only extrat OBS_STATE and ACTION
        state_keys = self.mapping[OBS_STATE]
        state_list, _ = self._align_for_cat([data[k] for k in state_keys])
        state = torch.cat(state_list, dim=-1)
        action_keys = self.mapping[ACTION]
        action_list, size = self._align_for_cat([data[k] for k in action_keys])
        action = torch.cat(action_list, dim=-1)
        mask = self.mask if self.mask is not None else torch.tensor([True] * state.shape[-1])
        action -= torch.where(mask, state, 0)[None]
        sid, eid = 0, 0
        for i, key in enumerate(action_keys):
            eid += size[i]
            data[key] = action[..., sid:eid]
            sid = eid
        return data
    
    def _align_for_cat(self, tensors: list[torch.Tensor], dim=-1) -> list[torch.Tensor]:
        max_ndim = max((t.ndim for t in tensors))
        out, size = [], []
        for t in tensors:
            t = t if t.ndim == max_ndim else t.unsqueeze(dim)
            out.append(t)
            size.append(t.shape[-1])
        return out, size


def hydrate_normalize_transform(
    transforms: list[DataTransformFn],
    dataset: LeRobotDataset | StreamingLeRobotDataset,
) -> list[DataTransformFn]:
    hydrated: list[DataTransformFn] = []
    for t in transforms:
        # if hasattr(t, "norm_stats") and hasattr(t, "selected_keys"):
        if isinstance(t, NormalizeTransformFn):
            robot_type = dataset.meta.robot_type
            EMBODIMENT_SPEC = FEATURE_MAPPING[robot_type]
            selected_keys = EMBODIMENT_SPEC[OBS_STATE] + EMBODIMENT_SPEC[ACTION]
            print(
                f"Hydrating transform {t.__class__.__name__} "
                f"with dataset.meta.stats (robot_type={robot_type}) "
                f"and selected_keys (selected_keys={selected_keys})"
            )
            t = replace(t, norm_stats=dataset.meta.stats, selected_keys=selected_keys)
        hydrated.append(t)
    return hydrated


def hydrate_compose_field_transform(
    transforms: list[DataTransformFn],
    dataset: LeRobotDataset | StreamingLeRobotDataset,
) -> list[DataTransformFn]:
    hydrated: list[DataTransformFn] = []
    for t in transforms:
        # if hasattr(t, "mapping"):
        if isinstance(t, ComposeFieldsTransform):
            print(
                f"Hydrating transform {t.__class__.__name__} "
                f"with mapping (robot_type={dataset.meta.robot_type})"
            )
            t = replace(t, mapping=FEATURE_MAPPING[dataset.meta.robot_type])
        hydrated.append(t)
    return hydrated


def hydrate_delta_action_transform(
    transforms: list[DataTransformFn],
    dataset: LeRobotDataset | StreamingLeRobotDataset,
) -> list[DataTransformFn]:
    hydrated: list[DataTransformFn] = []
    for t in transforms:
        # if hasattr(t, "action_state_pairs") and hasattr(t, "mask"):
        if isinstance(t, DeltaActionTransformFn):
            robot_type = dataset.meta.robot_type
            print(
                f"Hydrating transform {t.__class__.__name__} "
                f"with mapping and mask (robot_type={robot_type})"
            )
            t = replace(t, mapping=FEATURE_MAPPING[robot_type], mask=MASK_MAPPING[robot_type])
        hydrated.append(t)
    return hydrated


def hydrate_remap_image_key_transform(
    transforms: list[DataTransformFn],
    dataset: LeRobotDataset | StreamingLeRobotDataset, 
) -> list[DataTransformFn]:
    hydrated: list[DataTransformFn] = []
    for t in transforms:
        # if hasattr(t, "action_state_pairs") and hasattr(t, "mask"):
        if isinstance(t, RemapImageKeyTransformFn):
            robot_type = dataset.meta.robot_type
            print(
                f"Hydrating transform {t.__class__.__name__} "
                f"with mapping (robot_type={robot_type})"
            )
            t = replace(t, mapping=IMAGE_MAPPING[robot_type])
        hydrated.append(t)
    return hydrated


def filter_image_features(
    dataset: LeRobotDataset | StreamingLeRobotDataset, 
) -> None:
    robot_type = dataset.meta.robot_type
    mapping=IMAGE_MAPPING[robot_type]
    for key in dataset.meta.video_keys:
        if key not in mapping:
            dataset.meta.features.pop(key)
