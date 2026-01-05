from pathlib import Path
from pprint import pp
from torchvision.utils import save_image

import time
import logging
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.InternVLA_A1_3B import QwenA1Config, QwenA1Policy
from lerobot.datasets.utils import write_json
from lerobot.datasets.factory import make_dataset
from lerobot.transforms.core import UnNormalizeTransformFn
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, ACTION

def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ckpt_id = "060000"
    ckpt_path = Path("outputs/qwena1_a2d_400k/2025_12_17_22_53_50-qwena1-a2d_real_a2d_pen_holder-delta-28l-pretrain_400k-finetune_60k/checkpoints/060000/pretrained_model")
    # ckpt_path = Path("outputs/qwena1/2025_12_14_18_54_37-qwena1-jcaiaq_a2d_pick_pen-delta-28l-pretrain_100k-finetune_60k/checkpoints/060000/pretrained_model")
    ckpt_name = ckpt_path.parts[2]
    config = PreTrainedConfig.from_pretrained(ckpt_path)
    config.compile_model = True
    config.compile_mode = "reduce-overhead"
    dtype = torch.bfloat16
    # dtype = torch.float32
    assert isinstance(config, QwenA1Config)
    policy = QwenA1Policy.from_pretrained(
        config=config, 
        pretrained_name_or_path=ckpt_path, 
    )
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"\nTotal parameters: {total_params:,}  ({total_params / 1e9:.2f}B)")
    print(f"Und params: {sum(p.numel() for p in policy.model.qwen3_vl_with_expert.und_expert.parameters()) / 1e9:.2f}B")
    print(f"Gen params: {sum(p.numel() for p in policy.model.qwen3_vl_with_expert.gen_expert.parameters()) / 1e9:.2f}B")
    print(f"Act params: {sum(p.numel() for p in policy.model.qwen3_vl_with_expert.act_expert.parameters()) / 1e9:.2f}B")
    policy.cuda()
    policy.to(dtype)
    policy.eval()

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

    output_dir = Path(f"outputs/qwena1/{ckpt_name}/open_loop_results/last/{cfg.dataset.repo_id}")
    (output_dir / "plots").mkdir(exist_ok=True, parents=True)
    
    num_steps = 0
    elapse_time = 0

    for ep_id in range(min(num_episodes, 10)):
        print(f"episode: {ep_id}")
        print(f"from_idx: {from_ids[ep_id]}, to_idx: {to_ids[ep_id]}")
        action_gt_list = []
        action_pred_list = []
        state_list = []
        for idx in range(from_ids[ep_id], to_ids[ep_id], config.chunk_size):
            print(f"compute sample {idx}")
            sample = dataset[idx]
            inputs = {}
            for key in sample.keys():
                if key == 'task':
                    inputs[key] = [sample[key]]
                elif sample[key].dtype == torch.int64 or sample[key].dtype == torch.bool:
                    inputs[key] = sample[key][None].cuda()
                else:
                    inputs[key] = sample[key][None].cuda().to(dtype=dtype)
            with torch.no_grad():
                start_time = time.perf_counter()
                action_pred, _ = policy.predict_action_chunk(inputs, decode_image=False)
                end_time = time.perf_counter()
                elapse_time += end_time - start_time
                action_pred = action_pred[0, :, :16]
                action_gt = inputs['action'][0, :, :16]
                action_gt_list.append(action_gt)
                action_pred_list.append(action_pred.clone())
                state_list.append(inputs[OBS_STATE].clone().repeat(config.chunk_size, 1)[:, :16])
                # gt_images = torch.cat([inputs[key].clone() for key in inputs.keys() if OBS_IMAGES in key and "mask" not in key], dim=0)
                # gt_images = F.interpolate(gt_images, size=(256, 256), mode='bilinear', align_corners=False)
                # save_images = torch.cat([recon_images[0], gt_images], dim=0)
                # save_image(save_images, output_dir / "recon_images" / f"ep{ep_id}_id{idx}.png", nrow=3)
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
        # loss_recon = float(F.mse_loss(recon_images[0], gt_images))
        # recon_mse.append(loss_recon)
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
        plt.savefig(output_dir / "plots" / f"qwena1_open_loop_ep{ep_id}.jpg")
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
    pp(f"fps: {num_steps / (elapse_time)}")

if __name__ == '__main__':
    main()
