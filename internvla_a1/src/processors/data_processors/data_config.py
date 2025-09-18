import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from pathlib import Path
from omegaconf import OmegaConf
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from transformers.configuration_utils import PretrainedConfig
from src.utils.image_tools import (
    build_transform,
    dynamic_preprocess,
)
from src.utils.utils import convert_ds_stats_to_dict
import src.utils.transforms as _transforms
from src.configs.data_config import (
    LeRobotLiberoDataConfig, 
    LeRobotBridgeV2DataConfig, 
    LeRobotAgiBotWorldDataConfig,
    LeRobotFractalDataConfig,
    LeRobotSplitAlohaDataConfig,
    LeRobotARXLift2DataConfig,
)

from torchvision.transforms import transforms

from functools import partial

from PIL import Image, ImageOps


DATA_CONFIG_MAP = {
    "libero": LeRobotLiberoDataConfig,
    "bridge": LeRobotBridgeV2DataConfig,
    "agibot": LeRobotAgiBotWorldDataConfig,
    "fractal": LeRobotFractalDataConfig,
    "split_aloha": LeRobotSplitAlohaDataConfig,
    "arx_lift2": LeRobotARXLift2DataConfig,
}

def pad_to_square(img, fill=0):
    w, h = img.size
    pad_w = max(h - w, 0)
    pad_h = max(w - h, 0)
    padding = (pad_w, pad_h, 0, 0) #  (left, top, right, bottom)
    return ImageOps.expand(img, padding, fill=fill)

def create_data_config(data_config: dict, policy_config: PretrainedConfig, exp_config: dict):
    # create the transforms for the und expert
    und_transform = build_transform(data_config.image_size)
    und_expert_obs_transforms = [
        partial(dynamic_preprocess, image_size=data_config.image_size.height, max_num=1),
        lambda x: [und_transform(y) for y in x],
    ]

    # create the transforms for the generation expert
    # consistent_random_crop_stages = data_config.transforms.consistent_random_crop_stages
    # crop_method = _transforms.ConsistentRandomCrop if exp_config.stage in consistent_random_crop_stages else transforms.CenterCrop
    # FINAL_RESO = 256
    # gen_expert_obs_transforms = [
    #     crop_method((FINAL_RESO, FINAL_RESO)),
    #     transforms.ToTensor(), 
    # ]

    gen_expert_obs_transforms = [
        transforms.Lambda(lambda img: pad_to_square(img, fill=0)),
        transforms.Resize((256, 256)),
        transforms.ToTensor(), 
    ]

    # gen_expert_obs_transforms = [
    #     _transforms.ResizeImages(
    #         height=data_config.image_height_for_gen_expert, 
    #         width=data_config.image_width_for_gen_expert, 
    #         # suffix=data_config.world_model_suffix,
    #         suffix="physical",
    #     ),
    # ]

    # create the data config for each dataset
    configs = {}
    pbar = tqdm(data_config.train_dir.items(), total=len(data_config.train_dir))
    for key, value in pbar:
        pbar.set_description(f"[Creating dataset config]: {key}")

        dataset_type = value.type
        assert dataset_type in DATA_CONFIG_MAP, f"Data config type {dataset_type} not found in {DATA_CONFIG_MAP.keys()}"
        local_paths = []
        # recognize the data path and build the dataset metadata
        if hasattr(value, 'task_config_files'): #value.task_config_files is not None:
            task_config_files = value.task_config_files
            for task_config_file in task_config_files:
                task_config = OmegaConf.load(Path(task_config_file))
                value = task_config
                if isinstance(value.path, str) and "*" not in value.path or \
                    isinstance(value.path, list) and len(value.path) == 1 and "*" not in value.path[0]:
                    # to handle the case that the data path is a single path
                    if value.get("norm_stats_path", None) is None:
                        ds_meta = LeRobotDatasetMetadata(value.path)
                        ds_stats = ds_meta.stats
                    else:
                        with open(value.norm_stats_path, "r") as f:
                            ds_stats = json.load(f)
                    local_paths = local_paths + [value.path]

                elif isinstance(value.path, str) and "*" in value.path or isinstance(value.path, list):
                    # to handle the case that the data path includes wildcards '*'
                    all_local_paths = natsorted(glob(value.path))
                    if value.task_ids is not None:
                        local_paths = local_paths + [
                            p for p in all_local_paths
                            if any(str(task_id) in p for task_id in value.task_ids)
                        ]
                    else:
                        local_paths = local_paths + all_local_paths            

                    if value.get("norm_stats_path", None) is None:
                        sub_ds_stats = []
                        for local_path in local_paths:
                            ds_meta = LeRobotDatasetMetadata(local_path)
                            sub_ds_stats.append(ds_meta.stats)
                        ds_stats = aggregate_stats(sub_ds_stats)
                    else:
                        with open(value.norm_stats_path, "r") as f:
                            ds_stats = json.load(f)
                else:
                    raise ValueError(f"Invalid data config path: {value.path}")
        else:
            if isinstance(value.path, str) and "*" not in value.path or \
                isinstance(value.path, list) and len(value.path) == 1 and "*" not in value.path[0]:
                # to handle the case that the data path is a single path
                if value.get("norm_stats_path", None) is None:
                    ds_meta = LeRobotDatasetMetadata(value.path)
                    ds_stats = ds_meta.stats
                else:
                    with open(value.norm_stats_path, "r") as f:
                        ds_stats = json.load(f)
                local_paths = local_paths + [value.path]

            elif isinstance(value.path, str) and "*" in value.path or isinstance(value.path, list):
                # to handle the case that the data path includes wildcards '*'
                all_local_paths = natsorted(glob(value.path))
                if value.task_ids is not None:
                    local_paths = local_paths + [
                        p for p in all_local_paths
                        if any(str(task_id) in p for task_id in value.task_ids)
                    ]
                else:
                    local_paths = local_paths + all_local_paths            

                if value.get("norm_stats_path", None) is None:
                    sub_ds_stats = []
                    for local_path in local_paths:
                        ds_meta = LeRobotDatasetMetadata(local_path)
                        sub_ds_stats.append(ds_meta.stats)
                    ds_stats = aggregate_stats(sub_ds_stats)
                else:
                    with open(value.norm_stats_path, "r") as f:
                        ds_stats = json.load(f)
            else:
                raise ValueError(f"Invalid data config path: {value.path}")

        worker_idx = int(os.environ.get("MLP_ROLE_INDEX", 0))
        local_rank_idx = int(os.environ.get('LOCAL_RANK', -1))
        if worker_idx == 0 and local_rank_idx in [-1, 0]:
            dump_ds_stats = convert_ds_stats_to_dict(ds_stats)
            with open(f"{exp_config.training_args.output_dir}/{key}_stats.json", "w") as f:
                json.dump(dump_ds_stats, f, indent=4)

        # build the data config
        data_config_class = DATA_CONFIG_MAP[dataset_type]()
        configs[key] = data_config_class.initialize(
            policy_config=policy_config, 
            camera_keys=value.camera_keys, 
            state_keys=value.state_keys, 
            action_keys=value.action_keys, 
            predict_img_keys=value.predict_img_keys,
            gripper_action_keys=value.gripper_action_keys,
            image_size=data_config.image_size,
            gen_obs_suffix=data_config.world_model_suffix,
            local_path=local_paths, 
            action_mode=value.action_mode,
            ds_stats=ds_stats,
            do_normalize=True,
            norm_method=value.norm_method,
            weight=value.weight,
            n_obs_img_steps=value.n_obs_img_steps, 
            n_pred_img_steps=value.n_pred_img_steps, 
            obs_img_stride=value.obs_img_stride,
            num_indices=value.num_indices,
            und_expert_obs_transforms=und_expert_obs_transforms,
            gen_obs_transforms=gen_expert_obs_transforms,
        )

    return configs
