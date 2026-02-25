import re
import pandas as pd
import os, openpyxl
def extract_log_metrics(log_file_path):
    """
    从 out.log 文件中提取指定的监控指标
    :param log_file_path: log 文件路径（如 "out.log"）
    :return: 包含指标的 DataFrame
    """
    # 1. 定义需要提取的目标指标（与需求一致）
    target_metrics = [
        "image_to_text_mean_rank",
        "image_to_text_R@1",
        "image_to_text_R@2",
        "image_to_text_R@5",
        "image_to_text_R@10",
        "text_to_image_mean_rank",
        "text_to_image_R@1",
        "text_to_image_R@2",
        "text_to_image_R@5",
        "med-zeroshot-val-joint-top1",
        "med-zeroshot-val-joint-top2",
        "med-zeroshot-val-joint-top5",
        "med-zeroshot-val-joint-top10",
        "med-zeroshot-val-category-top1",
        "med-zeroshot-val-category-top2",
        "med-zeroshot-val-category-top5",
        "med-zeroshot-val-location-top1",
        "med-zeroshot-val-location-top2",
        "med-zeroshot-val-location-top5",
        "med-zeroshot-val-location-top10"  # 修正需求中的笔误（原"med-zeroshot-val-location-top"）
    ]
    
    # 2. 初始化结果列表（存储每一轮的指标数据）
    metrics_results = []
    
    # 3. 读取 log 文件并逐行解析
    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 只处理包含 Eval Epoch 的指标行（从日志可知指标均在 Eval Epoch 行中）
            if "Eval Epoch" in line and ("image_to_text" in line or "med-zeroshot" in line):
                # 提取当前 Epoch 号（用于标识每一轮结果）
                epoch_match = re.search(r"Eval Epoch: (\d+)", line)
                epoch = int(epoch_match.group(1)) if epoch_match else None
                
                # 初始化当前轮次的指标字典
                current_metrics = {"epoch": epoch}
                
                # 提取每个目标指标的数值（正则匹配 "指标名: 数值" 格式）
                for metric in target_metrics:
                    # 正则表达式：匹配 "metric_name: 数值"（数值可能包含小数点）
                    metric_pattern = re.compile(f"{metric}: ([0-9]+\.[0-9]+)")
                    match = metric_pattern.search(line)
                    # 若匹配到则存入数值，否则为 None（表示该轮无此指标）
                    current_metrics[metric] = float(match.group(1)) if match else None
                
                # 将当前轮次指标加入结果列表
                metrics_results.append(current_metrics)
    
    # 4. 转换为 DataFrame（便于查看和后续处理）
    metrics_df = pd.DataFrame(metrics_results)
    
    # 5. 清理数据：删除完全无指标的行（若有）
    metrics_df = metrics_df.dropna(how="all", subset=target_metrics)
    
    return metrics_df

# ------------------- 调用函数并查看结果 -------------------
if __name__ == "__main__":
    # 替换为你的 out.log 文件路径（相对路径或绝对路径均可）
    log_dirpath = "/root/MedCLIP-SAMv2-main/biomedclip_finetuning/open_clip/src/logs/2025_11_26-23_26_51-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp"
 
    log_path = os.path.join(log_dirpath,"out.log")
    result_xlsx = os.path.join(log_dirpath, "result.xlsx")
    # 提取指标
    metrics_data = extract_log_metrics(log_path)
    
    # 打印提取的结果（表格形式）
    print("=" * 120)
    print("从 out.log 中提取的监控指标结果：")
    print("=" * 120)
    print(metrics_data.to_string(index=False))  # 不显示索引列，更清晰
    
    # 可选：将结果保存为 Excel 文件（需安装 openpyxl：pip install openpyxl）
    metrics_data.to_excel(result_xlsx, index=False, engine="openpyxl")
    print(f"\n结果已保存到 {result_xlsx}")