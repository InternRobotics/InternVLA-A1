import safetensors.torch
import os

# ================= 配置区域 =================
# 输入文件路径 (你的原始权重文件)
input_path = "outputs/scratch/internvla_a1_2b/internvla_a1_2b-jcaiaq_a2d_pick_pen-scratch_60k-2025_12_07_02_55_40/checkpoints/060000/pretrained_model/model_bak3.safetensors"

# 输出文件路径 (修复后的文件)
output_path = "outputs/scratch/internvla_a1_2b/internvla_a1_2b-jcaiaq_a2d_pick_pen-scratch_60k-2025_12_07_02_55_40/checkpoints/060000/pretrained_model/model_fixed.safetensors"

def convert_safetensor():
    print(f"正在加载: {input_path}")
    if not os.path.exists(input_path):
        print(f"错误: 文件不存在 -> {input_path}")
        return

    # 加载原始权重
    tensors = safetensors.torch.load_file(input_path)
    new_tensors = {}
    
    print("开始转换键名...")
    
    for key, value in tensors.items():        
        # -------------------------------------------------
        # 1. 处理 und_expert (直接改名)
        # -------------------------------------------------
        if key.startswith("model.internvl_with_expert.internvl"):
            new_key = key.replace(
                "model.internvl_with_expert.internvl", 
                "model.internvl_with_expert.und_expert"
            )
            new_tensors[new_key] = value
        elif key.startswith("model.internvl_with_expert.qwen2_gen_expert"):
            new_key = key.replace(
                "model.internvl_with_expert.qwen2_gen_expert", 
                "model.internvl_with_expert.gen_expert"
            )
            new_tensors[new_key] = value
        elif key.startswith("model.internvl_with_expert.qwen2_expert"):
            new_key = key.replace(
                "model.internvl_with_expert.qwen2_expert", 
                "model.internvl_with_expert.act_expert"
            )
            new_tensors[new_key] = value
        elif "cosmos_in_proj" in key:
            new_key = key.replace(
                "cosmos_in_proj", 
                "gen_in_proj"
            )
            new_tensors[new_key] = value
        elif "cosmos_out_proj" in key:
            new_key = key.replace(
                "cosmos_out_proj", 
                "gen_out_proj"
            )
            new_tensors[new_key] = value
        elif "cosmos_out_layer_norm" in key:
            new_key = key.replace(
                "cosmos_out_layer_norm", 
                "gen_out_layer_norm"
            )
            new_tensors[new_key] = value
        elif "model.cosmos" in key:
            # import pdb; pdb.set_trace()
            new_key = key.replace(
                "model.cosmos", 
                "model.cosmos_tokenizer"
            )
        else:
            new_tensors[key] = value
                    

    print(f"转换完成。原始键数量: {len(tensors)}, 新键数量: {len(new_tensors)}")
    
    # 保存新文件
    print(f"正在保存到: {output_path}")
    safetensors.torch.save_file(new_tensors, output_path)
    print("保存成功！")

if __name__ == "__main__":
    convert_safetensor()
