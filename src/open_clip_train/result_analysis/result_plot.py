import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
import numpy as np  # 确保numpy正确导入且生效

def load_metrics(file_path):
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"读取文件 {file_path} 出错: {e}")
        return None

def get_metric_groups():
    return {
        "image_to_text_rank": [
            "image_to_text_R@1", "image_to_text_R@2", "image_to_text_R@5", "image_to_text_R@10"
        ],
        "text_to_image_rank": [
            "text_to_image_R@1", "text_to_image_R@2", "text_to_image_R@5"
        ],
        "zeroshot_joint": [
            "med-zeroshot-val-joint-top1", "med-zeroshot-val-joint-top2", "med-zeroshot-val-joint-top5", "med-zeroshot-val-joint-top10"
        ],
        "zeroshot_category": [
            "med-zeroshot-val-category-top1", "med-zeroshot-val-category-top2", "med-zeroshot-val-category-top5"
        ],
        "zeroshot_location": [
            "med-zeroshot-val-location-top1", "med-zeroshot-val-location-top2", "med-zeroshot-val-location-top5", "med-zeroshot-val-location-top10"
        ]
    }

def print_max_values(metric, dfs, labels, file_paths):
    """
    打印每个文件中指定指标的最大值、最后值及对应的epoch
    """
    print(f"\n{'='*80}")
    print(f"指标 {metric} 的统计信息:")
    print(f"{'='*80}")
    
    max_values = []
    last_values = []
    
    for i, (df, label, file_path) in enumerate(zip(dfs, labels, file_paths)):
        if metric not in df.columns:
            print(f"{label:50s} | 缺少指标 {metric}")
            continue

        # 筛选有效数据并按epoch排序
        tmp = df[['epoch', metric]].dropna().sort_values('epoch')
        if tmp.empty:
            print(f"{label:50s} | 无有效数据")
            continue

        # 找到最大值
        max_idx = tmp[metric].idxmax()
        max_epoch = int(tmp.loc[max_idx, 'epoch'])
        max_value = tmp.loc[max_idx, metric]
        
        # 找到最后一个epoch的值
        last_idx = tmp['epoch'].idxmax()
        last_epoch = int(tmp.loc[last_idx, 'epoch'])
        last_value = tmp.loc[last_idx, metric]
        
        max_values.append((label, max_value, max_epoch, file_path))
        last_values.append((label, last_value, last_epoch, file_path))
        
        print(f"{label:50s} | 最大值: {max_value:.4f} (Epoch {max_epoch:3d}) | 最后值: {last_value:.4f} (Epoch {last_epoch:3d}) | 文件: {os.path.basename(file_path)}")
    
    # 找出全局最大值和最佳最后值
    if max_values:
        global_max = max(max_values, key=lambda x: x[1])
        best_last = max(last_values, key=lambda x: x[1])
        
        print(f"{'-'*80}")
        print(f"全局最大值: {global_max[0]} - {global_max[1]:.4f} (Epoch {global_max[2]})")
        print(f"最佳最后值: {best_last[0]} - {best_last[1]:.4f} (Epoch {best_last[2]})")
        print(f"{'='*80}")
        
        return max_values, last_values, global_max, best_last
    else:
        print("未找到任何有效数据")
        return [], [], None, None
    

def plot_single_metric(metric, dfs, labels, output_dir, interpolate=True, method='linear'):
    """
    绘制单个指标（允许稀疏采样）的折线对比图。
    参数：
      metric: 指标名
      dfs: DataFrame 列表（每个文件读入的 DataFrame）
      labels: 对应标签
      output_dir: 输出目录
      interpolate: 是否对缺失 epoch 做插值（True/False）
      method: 插值方法，pandas 支持 'linear', 'time', 'index', 'nearest', 'spline' 等
    """
    plt.figure(figsize=(12, 6))
    os.makedirs(output_dir, exist_ok=True)

    any_plotted = False
    for df, label in zip(dfs, labels):
        if metric not in df.columns:
            print(f"{label} 缺少指标 {metric}，跳过")
            continue

        # 筛选有效数据并按epoch排序
        tmp = df[['epoch', metric]].dropna().sort_values('epoch')
        if tmp.empty:
            print(f"{label} 中 {metric} 无有效数据，跳过")
            continue

        # 确保epoch为整数、指标为浮点数（避免类型错误）
        epochs = tmp['epoch'].astype(int).values
        vals = tmp[metric].astype(float).values

        # 处理插值逻辑（确保np已定义）
        if interpolate and len(epochs) >= 2:
            # 生成连续的epoch序列（依赖numpy的arange）
            full_epochs = np.arange(epochs.min(), epochs.max() + 1)
            # 构造Series用于插值
            s = pd.Series(data=vals, index=epochs)
            s = s.reindex(full_epochs)  # 重索引到连续epoch（缺失处为NaN）
            # 插值填补NaN（两端缺失也填补）
            s = s.interpolate(method=method, limit_direction='both')
            # 绘制插值折线和原始采样点
            plt.plot(full_epochs, s.values, '-', linewidth=1.2, label=label)
            plt.scatter(epochs, vals, s=30, alpha=0.9, edgecolor='black', zorder=5)  # 采样点突出显示
            any_plotted = True
        else:
            # 无插值：直接绘制原始数据（单点显示为散点，多点显示为折线+散点）
            plt.plot(epochs, vals, '-o', markersize=4, linewidth=1.0, label=label, zorder=4)
            any_plotted = True

    if not any_plotted:
        print(f"未绘制任何 {metric} 数据（无有效采样）")
        plt.close()
        return

    # 图表美化与保存
    plt.title(f"{metric} Comparison Across Models", fontsize=14, pad=15)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 仅显示整数epoch
    plt.tight_layout()  # 自动调整布局，避免标签被截断

    # 保存图表（高分辨率）
    out_path = os.path.join(output_dir, f"{metric}_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存图表: {out_path}")

def plot_metric_group(group_name, metrics, dfs, labels, output_dir):
    """绘制一组指标的对比图（需所有文件都包含该指标）"""
    # 筛选所有文件都存在的指标
    existing_metrics = []
    for m in metrics:
        if all(m in df.columns for df in dfs):
            existing_metrics.append(m)
    if not existing_metrics:
        print(f"分组 {group_name} 没有所有文件都存在的共同指标，跳过")
        return

    plt.figure(figsize=(12, 6))
    for metric in existing_metrics:
        for df, label in zip(dfs, labels):
            # 筛选该指标的有效数据
            tmp = df[['epoch', metric]].dropna().sort_values('epoch')
            if not tmp.empty:
                epochs = tmp['epoch'].astype(int).values
                vals = tmp[metric].astype(float).values
                plt.plot(epochs, vals, marker='o', markersize=4, label=f"{label} - {metric}")

    # 图表美化与保存
    plt.title(f"{group_name.replace('_', ' ').title()} Comparison", fontsize=14, pad=15)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{group_name}_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存图表: {out_path}")

def compare_metrics(file_paths, labels, output_dir, mode="group", metrics=None):
    """
    主函数：对比多个Excel文件的指标并绘图
    参数：
      file_paths: 多个xlsx文件路径列表
      labels: 与file_paths一一对应的模型标签
      output_dir: 图表输出目录
      mode: "group"（按预设组绘图）或 "single"（单指标绘图）
      metrics: 单指标模式下的指标列表（mode="single"时必传）
    """
    # 加载所有Excel文件
    dfs = [load_metrics(fp) for fp in file_paths]
    # 检查是否有文件加载失败
    if any(df is None for df in dfs):
        failed_indices = [i for i, df in enumerate(dfs) if df is None]
        failed_paths = [file_paths[i] for i in failed_indices]
        print(f"以下文件加载失败，程序退出：{failed_paths}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 新增：在单指标模式下，打印每个文件的最大值和最后值
    if mode == "single" and metrics:
        for metric in metrics:
            max_values, last_values, global_max, best_last = print_max_values(metric, dfs, labels, file_paths)
            
            # 将统计信息保存到文件
            if max_values:
                stats_file = os.path.join(output_dir, f"{metric}_statistics.txt")
                with open(stats_file, 'w', encoding='utf-8') as f:
                    f.write(f"指标 {metric} 的统计信息\n")
                    f.write("="*80 + "\n")
                    for i, (label, max_val, max_epoch, path) in enumerate(max_values):
                        last_val, last_epoch = last_values[i][1], last_values[i][2]
                        f.write(f"{label:50s} | 最大值: {max_val:.4f} (Epoch {max_epoch:3d}) | 最后值: {last_val:.4f} (Epoch {last_epoch:3d}) | 文件: {os.path.basename(path)}\n")
                    if global_max and best_last:
                        f.write("-"*80 + "\n")
                        f.write(f"全局最大值: {global_max[0]} - {global_max[1]:.4f} (Epoch {global_max[2]})\n")
                        f.write(f"最佳最后值: {best_last[0]} - {best_last[1]:.4f} (Epoch {best_last[2]})\n")
                print(f"统计信息已保存到: {stats_file}")
    
    if mode == "single":
        # 单指标模式：检查metrics参数有效性
        if metrics is None or (isinstance(metrics, (list, tuple)) and len(metrics) == 0):
            print("单指标模式需传入有效的metrics列表")
            return
        if isinstance(metrics, str):
            metrics = [metrics]  # 统一转为列表
        # 逐个指标绘图
        for metric in metrics:
            plot_single_metric(metric, dfs, labels, output_dir)
    else:
        # 分组模式：按预设指标组绘图
        metric_groups = get_metric_groups()
        for group_name, group_metrics in metric_groups.items():
            plot_metric_group(group_name, group_metrics, dfs, labels, output_dir)
    print(f"所有图表已保存到 {output_dir} 目录")

if __name__ == "__main__":
    # 1. 配置文件路径和模型标签（确保一一对应）
    file_paths = [
        #"/root/autodl-fs/clip_log/2025_08_29-15_19_36-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx",
        #"/root/autodl-fs/clip_log/2025_08_31-10_53_29-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx",
        #"/root/autodl-fs/clip_log/2025_08_28-20_56_21-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx",
        #"/root/autodl-fs/clip_log/2025_08_29-22_19_58-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx",
        #"/root/autodl-fs/clip_log/2025_08_30-12_51_00-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx",
        #"/root/autodl-fs/clip_log/2025_08_31-23_04_40-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx",
        #"/root/autodl-fs/clip_log/2025_09_01-10_56_22-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx",
        # "/root/autodl-fs/clip_log/2025_09_19-00_07_41-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_16-j_4-p_amp/result.xlsx",
        # "/root/autodl-fs/clip_log/2025_09_30-23_34_14-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_32-j_4-p_amp/result.xlsx",
        # "/root/MedCLIP-SAMv2-main/biomedclip_finetuning/open_clip/src/logs/2025_10_02-17_08_08-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_32-j_4-p_amp/result.xlsx",
        # "/root/MedCLIP-SAMv2-main/biomedclip_finetuning/open_clip/src/logs/2025_10_05-22_19_18-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_32-j_4-p_amp/result.xlsx",
        # "/root/MedCLIP-SAMv2-main/biomedclip_finetuning/open_clip/src/logs/2025_10_05-21_20_43-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_32-j_4-p_amp/result.xlsx",
        # "/root/MedCLIP-SAMv2-main/biomedclip_finetuning/open_clip/src/logs/2025_10_05-21_13_26-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_32-j_4-p_amp/result.xlsx",
        # "/root/autodl-fs/clip_log/2025_10_03-15_06_49-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_256-j_4-p_amp/result.xlsx",
        # "/root/autodl-fs/clip_log/2025_10_03-21_37_00-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_32-j_4-p_amp/result.xlsx",
        # "/root/autodl-fs/clip_log/2025_10_02-17_08_08-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_32-j_4-p_amp/result.xlsx"
        "/root/autodl-fs/clip_log/2025_10_05-22_19_18-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_32-j_4-p_amp/result.xlsx",
        "/root/autodl-fs/clip_log/2025_10_05-21_20_43-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_32-j_4-p_amp/result.xlsx",
        "/root/autodl-fs/clip_log/2025_10_05-21_13_26-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_32-j_4-p_amp/result.xlsx"
    ]
    labels = [
        #"CLIP",
        #"CLIP_explicit_1",
        #"CLIP_explicit_2",
        #"CLIP_explicit_2_implicit_2_4_8_8",
        #"CLIP_explicit_2_implicit_2_8_16_32",
        #"CLIP_explicit_2_implicit_2_dynamic_8_20epoch",
        #"CLIP_explicit_2_implicit_2_dynamic_8_280epoch",
        # "CLIP_explicit_2_implicit_2_dynamic_8_0epoch_OrthoGradient",
        # "CLIP_explicit_2_implicit_2_dynamic_8_0epoch_OrthoGradient_distill_image",
        # "CLIP_explicit_2_implicit_2_dynamic_8_0epoch_OrthoGradient_dino_distill_double",
        # "CLIP_pmc_oa",
        # "CLIP_explicit_2_implicit_2_dynamic_8_0epoch_OrthoGradient_pmc_oa",
        # "CLIP_explicit_2_implicit_2_dynamic_8_0epoch_OrthoGradient_dino_distill_double_pmc_oa"
        # "CLIP_mimic_cxr",
        # "CLIP_explicit_2_implicit_2_dynamic_8_0epoch_OrthoGradient_mimic_cxr",
        # "CLIP_explicit_2_implicit_2_dynamic_8_0epoch_OrthoGradient_dino_distill_double_mimic_cxr"
        "CLIP_pmc_oa",
        "CLIP_explicit_2_implicit_2_dynamic_8_0epoch_OrthoGradient_pmc_oa",
        "CLIP_explicit_2_implicit_2_dynamic_8_0epoch_OrthoGradient_dino_distill_double_pmc_oa"
        
    ]

    # 2. 配置绘图参数
    # target_metrics = [
    #     "image_to_text_mean_rank",
    #     "image_to_text_R@1",
    #     "image_to_text_R@2",
    #     "image_to_text_R@5",
    #     "image_to_text_R@10",
    #     "text_to_image_mean_rank",
    #     "text_to_image_R@1",
    #     "text_to_image_R@2",
    #     "text_to_image_R@5",
    #     "text_to_image_R@10"
    #     "med-zeroshot-val-joint-top1",
    #     "med-zeroshot-val-joint-top2",
    #     "med-zeroshot-val-joint-top5",
    #     "med-zeroshot-val-joint-top10",
    #     "med-zeroshot-val-category-top1",
    #     "med-zeroshot-val-category-top2",
    #     "med-zeroshot-val-category-top5",
    #     "med-zeroshot-val-location-top1",
    #     "med-zeroshot-val-location-top2",
    #     "med-zeroshot-val-location-top5",
    #     "med-zeroshot-val-location-top10"
    # ]
    target_metric = "med-zeroshot-val-location-top2"  # 目标指标
    output_root_dir = "/root/MedCLIP-SAMv2-main/biomedclip_finetuning/open_clip/src/result"
    # 最终输出目录（按指标命名，便于区分）
    output_dir = os.path.join(output_root_dir, target_metric)
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存到: {output_dir}")

    # 3. 执行单指标对比绘图
    compare_metrics(
        file_paths=file_paths,
        labels=labels,
        output_dir=output_dir,
        mode="single",
        metrics=[target_metric]  # 传入列表形式的指标
    )

    # （可选）若需要分组对比，取消以下注释
    # compare_metrics(
    #     file_paths=file_paths,
    #     labels=labels,
    #     output_dir=output_dir,
    #     mode="group"
    # )