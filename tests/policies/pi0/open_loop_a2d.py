from pathlib import Path
from pprint import pp

import time
import pprint
import logging
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi0 import PI0Config, PI0Policy
from lerobot.datasets.utils import write_json
from lerobot.datasets.factory import make_dataset
from lerobot.transforms.core import UnNormalizeTransformFn
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_IMAGES

def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # ckpt_path = Path(f"outputs/pi0/2025_10_30_22_31_43-pi0-jcaiaq_a2d_pick_pen-abs_action-full_layers/checkpoints/last/pretrained_model")
    # ckpt_path = Path(f"outputs/pi0/2025_10_30_23_24_37-pi0-jcaiaq_a2d_pick_pen-abs_action-full_layers-from_scratch/checkpoints/last/pretrained_model")
    # ckpt_path = Path(f"outputs/pi0/2025_11_03_23_47_22-pi0-jcaiaq_a2d_pick_pen-abs-from_scratch-recommend_train_params/checkpoints/last/pretrained_model")
    # ckpt_path = Path(f"outputs/pi0/2025_11_05_23_35_36-pi0-jcaiaq_a2d_pick_pen-abs-finetuned-fix_action_expert/checkpoints/060000/pretrained_model")
    # ckpt_path = Path(f"outputs/pi0/2025_11_06_10_44_21-pi0-jcaiaq_a2d_pick_pen-abs-finetuned-fix_action_expert-init_with_gemma_vlm/checkpoints/last/pretrained_model")
    # ckpt_path = Path(f"outputs/pi0/2025_11_06_11_19_04-pi0-jcaiaq_a2d_pick_pen-abs-finetuned-init_with_gemma_vlm-init_with_pi0_expert/checkpoints/last/pretrained_model")
    # ckpt_path = Path(f"outputs/pi0/2025_11_08_15_49_55-pi0-jcaiaq_a2d_pick_pen-delta-finetuned/checkpoints/last/pretrained_model")  # !!!!!! remember to set _vocab_size to 257216
    ckpt_path = Path(f"outputs/pi0/2025_11_10_10_59_27-pi0-jcaiaq_a2d_pick_pen-delta-from_scratch-60k/checkpoints/last/pretrained_model")  
    ckpt_name = ckpt_path.parts[2]
    config = PreTrainedConfig.from_pretrained(ckpt_path)
    config.compile_model = True
    config.compile_mode = "reduce-overhead"
    # dtype = torch.bfloat16
    dtype = torch.float32
    assert isinstance(config, PI0Config)
    policy = PI0Policy.from_pretrained(
        config=config, 
        pretrained_name_or_path=ckpt_path, 
    )
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"\nTotal parameters: {total_params:,}  ({total_params / 1e9:.2f}B)")
    print(f"PaliGemma params: {sum(p.numel() for p in policy.model.paligemma_with_expert.paligemma.parameters()) / 1e9:.2f}B")
    print(f"Gemma_Expert params: {sum(p.numel() for p in policy.model.paligemma_with_expert.gemma_expert.parameters()) / 1e9:.2f}B")
    policy.cuda()
    policy.to(dtype)
    policy.eval()
    dummy_inputs = {
        OBS_STATE: torch.rand((1, 32), dtype=dtype).cuda(), 
        ACTION: torch.rand((1, 50, 32), dtype=dtype).cuda(), 
        f"{OBS_IMAGES}.image0": torch.rand((1, 3, 224, 224), dtype=dtype).cuda(), 
        f"{OBS_IMAGES}.image1": torch.rand((1, 3, 224, 224), dtype=dtype).cuda(), 
        f"{OBS_IMAGES}.image2": torch.rand((1, 3, 224, 224), dtype=dtype).cuda(), 
        f"{OBS_IMAGES}.image0_mask": torch.tensor([True]).cuda(), 
        f"{OBS_IMAGES}.image1_mask": torch.tensor([True]).cuda(), 
        f"{OBS_IMAGES}.image2_mask": torch.tensor([True]).cuda(), 
        "task": ['Put the pen from the table into the pen holder.'], 
        "observation.language.tokens": torch.randint(0, 7777, (1, 48)).cuda(), 
        "observation.language.attention_mask": torch.ones((1, 48)).cuda(), 
    }
    logger.info("policy warmup ... ")
    with torch.no_grad():
        policy.predict_action_chunk(dummy_inputs)
        policy.predict_action_chunk(dummy_inputs)

    cfg = TrainPipelineConfig.from_pretrained(ckpt_path)
    cfg.dataset.repo_id = "jcaiaq/pick_marker_pen_inference_rollouts_v30"
    # cfg.dataset.repo_id = "jcaiaq/a2d_pick_pen"
    cfg.dataset.use_external_stats = True
    action_mode = cfg.dataset.action_mode

    dataset, _ = make_dataset(cfg)

    stat_keys = ['min', 'max', 'mean', 'std']
    action_stat = {stat_key: np.concatenate([
        dataset.meta.stats["actions.joint.position"][stat_key], 
        dataset.meta.stats["actions.effector.position"][stat_key]
    ], axis=-1) for stat_key in stat_keys}
    state_stat = {stat_key: np.concatenate([
        dataset.meta.stats["observation.states.joint.position"][stat_key], 
        dataset.meta.stats["observation.states.effector.position"][stat_key]
    ], axis=-1) for stat_key in stat_keys}
    act_unnorm_fn = UnNormalizeTransformFn(
        selected_keys=[ACTION], 
        mode="mean_std", 
        norm_stats={ACTION: action_stat}, 
    )
    state_unnorm_fn = UnNormalizeTransformFn(
        selected_keys=[OBS_STATE], 
        mode="mean_std", 
        norm_stats={OBS_STATE: state_stat}, 
    )

    from_ids = np.asarray(dataset.meta.episodes['dataset_from_index']).tolist()
    to_ids = np.asarray(dataset.meta.episodes['dataset_to_index']).tolist()
    num_episodes = dataset.num_episodes

    metric_mse = []
    mse_joint = []
    mse_gripper = []

    output_dir = Path(f"outputs/pi0/{ckpt_name}/open_loop_results/last")
    (output_dir / "plots").mkdir(exist_ok=True, parents=True)

    num_steps = 0
    elapse_time = 0
    
    for ep_id in range(min(num_episodes, 10)):
        print(f"episode: {ep_id}")
        print(f"from_idx: {from_ids[ep_id]}, to_idx: {to_ids[ep_id]}")
        action_gt_list = []
        action_pred_list = []
        state_list = []
        # start_time = time.perf_counter()
        for idx in range(from_ids[ep_id], to_ids[ep_id], config.chunk_size):
            print(f"compute sample {idx}")
            sample = dataset[idx]
            inputs = {}
            for key in sample.keys():
                if key == 'task':
                    inputs[key] = [sample[key]]
                elif sample[key].dtype == torch.int64:
                    inputs[key] = sample[key][None].cuda()
                else:
                    inputs[key] = sample[key][None].cuda().to(dtype=dtype)
            with torch.no_grad():
                start_time = time.perf_counter()
                action_pred = policy.predict_action_chunk(inputs)[0, :, :16]
                end_time = time.perf_counter()
                if num_steps > 2:
                    elapse_time += end_time - start_time
                action_gt = inputs[ACTION][0, :, :16]
                action_gt_list.append(action_gt)
                action_pred_list.append(action_pred.clone())
                state_list.append(inputs[OBS_STATE].clone().repeat(config.chunk_size, 1)[:, :16])
            num_steps += 1
        action_gt_tensor = torch.cat(action_gt_list, dim=0)
        action_gt_tensor = act_unnorm_fn({ACTION: action_gt_tensor})[ACTION]
        action_pred_tensor = torch.cat(action_pred_list, dim=0)
        action_pred_tensor = act_unnorm_fn({ACTION: action_pred_tensor})[ACTION]
        if action_mode == 'delta':
            state_tensor = torch.cat(state_list, dim=0)
            state_tensor = state_unnorm_fn({OBS_STATE: state_tensor})[OBS_STATE]
            action_pred_tensor[:, :14] += state_tensor[:, :14]
            action_gt_tensor[:, :14] += state_tensor[:, :14]
        action_gt_tensor = action_gt_tensor.to(torch.float32)
        action_pred_tensor = action_pred_tensor.to(torch.float32)
        # end_time = time.perf_counter()
        # elapse_time += end_time - start_time
        metric_mse.append(float(F.mse_loss(action_gt_tensor, action_pred_tensor, reduction='mean').detach().cpu().numpy()))
        mse_joint.append(float(F.mse_loss(action_gt_tensor[:, :14], action_pred_tensor[:, :14], reduction='mean').detach().cpu().numpy()))
        mse_gripper.append(float(F.mse_loss(action_gt_tensor[:, 14:], action_pred_tensor[:, 14:], reduction='mean').detach().cpu().numpy()))
        action_gt_numpy = action_gt_tensor.detach().cpu().numpy()
        action_pred_numpy = action_pred_tensor.detach().cpu().numpy()
        fig, axs = plt.subplots(8, 2, figsize=(16, 12))
        axs = axs.ravel()
        num_dimensions = action_gt_numpy.shape[1]
        x_values = np.arange(action_gt_numpy.shape[0])
        for dim in range(num_dimensions):
            axs[dim].plot(x_values, action_gt_numpy[:, dim], label='Ground Truth', color='blue', linewidth=1.5)
            axs[dim].plot(x_values, action_pred_numpy[:, dim], label='Predicted', color='red', linestyle='--', linewidth=1.5)
            axs[dim].set_title(f'Dimension {dim+1}')
            axs[dim].set_xlabel('Time Step / Sample Index')
            axs[dim].set_ylabel(f'Value Dim {dim+1}')
            axs[dim].legend(loc='upper right')
            axs[dim].grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.suptitle('Ground Truth vs Prediction', fontsize=16, y=1.02)
        plt.savefig(output_dir / "plots" / f"pi0_open_loop_ep{ep_id}.jpg")
    log = {
        "MSE": metric_mse, 
        "Average MSE": np.mean(metric_mse), 
        "MSE on joints": mse_joint, 
        "Average MSE on joints": np.mean(mse_joint), 
        "MSE on gripper": mse_gripper, 
        "Average MSE on gripper": np.mean(mse_gripper), 
    }
    write_json(log, output_dir/"log.json")
    pp(log)
    pp(f"elapse time: {elapse_time}s")
    pp(f"fps: {(num_steps - 2) / (elapse_time)}")

if __name__ == '__main__':
    main()
