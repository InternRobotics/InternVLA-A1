#!/usr/bin/env python

import sys
import os
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import imageio
import numpy as np
import torch
import tyro
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import load_json
from lerobot.policies.InternVLA_A1_3B.modeling_internvla_a1 import QwenA1Config, QwenA1Policy
from lerobot.policies.InternVLA_A1_3B.transform_internvla_a1 import Qwen3_VLProcessorTransformFn
from lerobot.transforms.core import (
    NormalizeTransformFn,
    ResizeImagesWithPadFn,
    UnNormalizeTransformFn,
    RemapImageKeyTransformFn,
    compose,
)
from lerobot.utils.constants import OBS_IMAGES

ROOT_PATH = Path(__file__).resolve().parents[2]
# RoboTwin dependencies
sys.path.extend(
    [
        str(ROOT_PATH),
        str(ROOT_PATH / "third_party" / "RoboTwin"),
        str(ROOT_PATH / "third_party" / "RoboTwin" / "policy"),
        str(ROOT_PATH / "third_party" / "RoboTwin" / "description" / "utils"),
    ]
)

from envs import CONFIGS_PATH 
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions
import image_tools


def resolve_ckpt_dir(ckpt_path: Union[str, Path]) -> Path:
    """
    Resolve a checkpoint path to a local directory.

    Supports:
    - Local directory path
    - HuggingFace repo id (e.g., "org/repo"), downloaded to HF cache via snapshot_download
    """
    ckpt_str = str(ckpt_path)
    local_dir = Path(ckpt_str).expanduser()
    if local_dir.exists():
        return local_dir.resolve()

    snapshot_dir = snapshot_download(repo_id=ckpt_str)
    return Path(snapshot_dir)


# Task list matching eval_robotwin.py
TASK_NAMES = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]


def get_embodiment_config(robot_file: str):
    """Load robot embodiment configuration from YAML file."""
    robot_config_file = Path(robot_file) / "config.yml"
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return OmegaConf.load(f)


def class_decorator(task_name: str):
    """Dynamically import and instantiate task environment class."""
    import importlib

    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def build_task_args(task_config: str, task_name: str):
    """Build task arguments from configuration files."""
    task_cfg_file = ROOT_PATH / "third_party" / "RoboTwin" / "task_config" / f"{task_config}.yml"
    with open(task_cfg_file, "r", encoding="utf-8") as f:
        task_args = OmegaConf.to_container(OmegaConf.load(f), resolve=True)

    with open(CONFIGS_PATH + "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiment_types = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_cfg = OmegaConf.to_container(OmegaConf.load(f), resolve=True)

    def get_embodiment_file(embodiment_type):
        robot_file = embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files found")
        return robot_file

    embodiment_type = task_args["embodiment"]
    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = camera_cfg[head_camera_type]["h"]
    task_args["head_camera_w"] = camera_cfg[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        robot_file = str(ROOT_PATH / "third_party" / "RoboTwin" / get_embodiment_file(embodiment_type[0]))
        task_args["left_robot_file"] = robot_file
        task_args["right_robot_file"] = robot_file
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = str(
            ROOT_PATH / "third_party" / "RoboTwin" / get_embodiment_file(embodiment_type[0])
        )
        task_args["right_robot_file"] = str(
            ROOT_PATH / "third_party" / "RoboTwin" / get_embodiment_file(embodiment_type[1])
        )
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise RuntimeError(f"Invalid embodiment type length: {len(embodiment_type)}, expected 1 or 3")

    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])
    task_args["task_name"] = task_name
    task_args["task_config"] = task_config
    task_args["eval_mode"] = True
    return task_args


def build_policy_and_transforms(ckpt_path: Union[str, Path], stats_key: str, resize_size: int, dtype: torch.dtype):
    """Load policy and build input/output transforms."""
    ckpt_dir = resolve_ckpt_dir(ckpt_path)
    config = PreTrainedConfig.from_pretrained(ckpt_dir)
    if not isinstance(config, QwenA1Config):
        raise ValueError(f"Expected QwenA1Config, got {type(config)}")
    
    policy = QwenA1Policy.from_pretrained(config=config, pretrained_name_or_path=ckpt_dir)
    policy.cuda().to(dtype).eval()

    stats = load_json(ckpt_dir / "stats.json")[stats_key]
    stat_keys = ["min", "max", "mean", "std"]

    state_concat = {k: np.asarray(stats["observation.state"][k]) for k in stat_keys}
    state_stat = {"observation.state": state_concat}

    action_concat = {k: np.asarray(stats["action"][k]) for k in stat_keys}
    action_stat = {"action": action_concat}

    unnormalize_fn = UnNormalizeTransformFn(
        selected_keys=["action"],
        mode="mean_std",
        norm_stats=action_stat,
    )

    image_keys = [f"{OBS_IMAGES}.image{i}" for i in range(3)]

    input_transforms = compose(
        [
            ResizeImagesWithPadFn(height=resize_size, width=resize_size),
            RemapImageKeyTransformFn(mapping={k: k for k in image_keys}),
            Qwen3_VLProcessorTransformFn(),
            NormalizeTransformFn(selected_keys=["observation.state"], norm_stats=state_stat),
        ]
    )

    return policy, input_transforms, unnormalize_fn


@dataclass
class InferenceArgs:
    """Configuration arguments for inference."""

    task_idx: int = 0
    task_config: str = "demo_clean"
    instruction_type: str = "unseen"
    seed: int = 0
    ckpt_path: Union[str, Path] = "InternRobotics/InternVLA-A1-3B"
    stats_key: str = "aloha"
    resize_size: int = 224
    image_history_interval: int = 15
    action_mode: str = "delta"  # delta | abs
    dtype: str = "float32"  # float32 | bfloat16
    video_dir: Path = Path("videos")
    fps: int = 30
    decode_image_flag: bool = False
    debug: bool = False
    log_level: str = "WARNING"  # DEBUG | INFO | WARNING | ERROR
    infer_horizon: int = 30
    action_horizon_size: int = 50
    test_num: int = 100
    robot_type: tuple[int, ...] = (6, 1, 6, 1)



def infer_once(args: InferenceArgs):
    """Run inference on a single task."""
    task_name = TASK_NAMES[args.task_idx]
    task_args = build_task_args(args.task_config, task_name)
    TASK_ENV = class_decorator(task_args["task_name"])

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    policy, input_transforms, unnormalize_fn = build_policy_and_transforms(
        args.ckpt_path, args.stats_key, args.resize_size, dtype
    )

    logging.info("=" * 80)
    logging.info("Initializing environment...")
    logging.info(f"Task: {task_name}, seed: {args.seed}")

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    expert_check = True

    now_id = 0
    succ_seed = 0
    seed = args.seed
    st_seed = 100000 * (1 + seed)
    now_seed = st_seed
    test_num = args.test_num
    clear_cache_freq = task_args["clear_cache_freq"]
    task_args["eval_mode"] = True
    succ_seeds = list(range(st_seed, st_seed * 2))

    while succ_seed < test_num:
        render_freq = task_args["render_freq"]
        task_args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(
                    now_ep_num=now_id, seed=succ_seeds[now_seed - st_seed], is_test=True, **task_args
                )
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except (UnStableError, Exception):
                TASK_ENV.close_env()
                now_seed += 1
                task_args["render_freq"] = render_freq
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
        else:
            now_seed += 1
            task_args["render_freq"] = render_freq
            continue

        task_args["render_freq"] = render_freq

        TASK_ENV.setup_demo(
            now_ep_num=now_id, seed=succ_seeds[now_seed - st_seed], is_test=True, **task_args
        )
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(task_name, episode_info_list, test_num)
        instruction = np.random.choice(results[0][args.instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        succ = False
        policy.reset()
        action_plan = deque([], maxlen=args.action_horizon_size)
        replay_images = []
        head_color_list = []
        left_wrist_color_list = []
        right_wrist_color_list = []
        image_history_interval = args.image_history_interval
        action_dim = sum(args.robot_type)
        left_gripper_idx = sum(args.robot_type[0:2])-1
        right_gripper_idx = sum(args.robot_type[0:4])-1

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            if len(action_plan) <= image_history_interval:
                observation = TASK_ENV.get_obs()
                img = observation["observation"]["head_camera"]["rgb"]
                left_wrist_img = observation["observation"]["left_camera"]["rgb"]
                right_wrist_img = observation["observation"]["right_camera"]["rgb"]

                head_color_list.append(torch.as_tensor(img).contiguous().cuda().to(dtype) / 255.0)
                left_wrist_color_list.append(torch.as_tensor(left_wrist_img).contiguous().cuda().to(dtype) / 255.0)
                right_wrist_color_list.append(torch.as_tensor(right_wrist_img).contiguous().cuda().to(dtype) / 255.0)

                while len(head_color_list) > image_history_interval + 1:
                    head_color_list.pop(0)
                    left_wrist_color_list.pop(0)
                    right_wrist_color_list.pop(0)

                past_idx = max(len(head_color_list) - image_history_interval - 1, 0)
                image_head_with_history = torch.stack([head_color_list[past_idx], head_color_list[-1]], dim=0)
                image_hand_left_with_history = torch.stack(
                    [left_wrist_color_list[past_idx], left_wrist_color_list[-1]], dim=0
                )
                image_hand_right_with_history = torch.stack(
                    [                    right_wrist_color_list[past_idx], right_wrist_color_list[-1]], dim=0
                )

            if not action_plan:
                init_action = torch.as_tensor(observation["joint_action"]["vector"][None]).contiguous().cuda()
                state = torch.from_numpy(observation["joint_action"]["vector"]).float().cuda()
                task = TASK_ENV.get_instruction()

                sample = {
                    f"{OBS_IMAGES}.image0": image_head_with_history,
                    f"{OBS_IMAGES}.image1": image_hand_left_with_history,
                    f"{OBS_IMAGES}.image2": image_hand_right_with_history,
                    "observation.state": state,
                    "task": task,
                }
                for key in sample.keys():
                    if OBS_IMAGES in key and "mask" not in key:
                        image = sample[key].permute(0, 3, 1, 2)
                        sample[key] = image

                sample = input_transforms(sample)
                transformed_image_to_save = sample[f"{OBS_IMAGES}.image0"][0]  # (3, 224, 224)
                replay_images.append(image_tools.convert_to_uint8(transformed_image_to_save.cpu().numpy()))

                inputs = {}
                for key in sample.keys():
                    if key == "task":
                        inputs[key] = [sample[key]]
                    elif sample[key].dtype == torch.int64:
                        inputs[key] = sample[key][None].cuda()
                    else:
                        inputs[key] = sample[key][None].cuda().to(dtype=dtype)

                inputs.update({
                    f"{OBS_IMAGES}.image0_mask": torch.tensor([True]).cuda(),
                    f"{OBS_IMAGES}.image1_mask": torch.tensor([True]).cuda(),
                    f"{OBS_IMAGES}.image2_mask": torch.tensor([True]).cuda(),
                })

                with torch.no_grad():
                    action_pred, _ = policy.predict_action_chunk(inputs, decode_image=args.decode_image_flag)

                action_pred = action_pred[0, : args.infer_horizon, :action_dim]
                action_pred = unnormalize_fn({"action": action_pred})["action"]

                if args.action_mode == "delta":
                    init_action[:, left_gripper_idx] = 0.0
                    init_action[:, right_gripper_idx] = 0.0
                    action_pred += init_action
                action_plan.extend(action_pred.cpu().numpy())

            action = action_plan.popleft()
            action[left_gripper_idx] = 0 if action[left_gripper_idx] < 0.5 else 1
            action[right_gripper_idx] = 0 if action[right_gripper_idx] < 0.5 else 1
            TASK_ENV.take_action(action, action_type="qpos")

            if TASK_ENV.eval_success:
                succ = True
                break

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        # Save a replay video of the episode
        args.video_dir.mkdir(parents=True, exist_ok=True)
        suffix = "success" if succ else "failure"
        imageio.mimwrite(
            args.video_dir / f"{suffix}_{succ_seed}.mp4",
            [np.asarray(x).transpose(1, 2, 0) for x in replay_images],
            fps=args.fps,
        )

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m |  \033[92m{task_args['task_config']}\033[0m \033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
            f"\033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1


def main(args: InferenceArgs):
    """Main entry point for inference."""
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = log_level_map.get(args.log_level.upper(), logging.INFO)
    if args.debug:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Suppress curobo INFO logs
    logging.getLogger("curobo").setLevel(logging.WARNING)

    logging.info("=" * 80)
    logging.info("Starting inference...")
    logging.info(f"Debug mode: {args.debug}, Log level: {args.log_level}")
    logging.info(f"Task index: {args.task_idx}, Checkpoint: {args.ckpt_path}")

    infer_once(args)


if __name__ == "__main__":
    tyro.cli(main)