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
            "text_to_text_R@1", "text_to_image_R@2", "text_to_image_R@5"
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


def plot_single_metric(metric, dfs, labels, output_dir, interpolate=True, method='linear'):
    """
    绘制单个指标（允许稀疏采样）的折线对比图，新增显示曲线最大值功能。
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

        # 计算当前曲线的最大值及对应epoch
        max_val = np.max(vals)
        max_epoch_idx = np.argmax(vals)
        max_epoch = epochs[max_epoch_idx]

        # 处理插值逻辑（依赖numpy的arange）
        if interpolate and len(epochs) >= 2:
            # 生成连续的epoch序列
            full_epochs = np.arange(epochs.min(), epochs.max() + 1)
            # 构造Series用于插值
            s = pd.Series(data=vals, index=epochs)
            s = s.reindex(full_epochs)  # 重索引到连续epoch（缺失处为NaN）
            # 插值填补NaN（两端缺失也填补）
            s = s.interpolate(method=method, limit_direction='both')
            # 绘制插值折线和原始采样点
            plt.plot(full_epochs, s.values, '-', linewidth=1.2, label=label)
            plt.scatter(epochs, vals, s=30, alpha=0.9, edgecolor='black', zorder=5)  # 采样点突出显示
            # 标注最大值（插值后若需更精准最大值，可使用s.values计算）
            plt.scatter(max_epoch, max_val, s=80, c='red', marker='*', zorder=10, 
                        label=f'{label} Max' if any_plotted is False else "")
            # 添加最大值文本标注
            plt.annotate(f'Max: {max_val:.4f}\nEpoch: {max_epoch}',
                         xy=(max_epoch, max_val),
                         xytext=(5, 5),  # 文本偏移量
                         textcoords='offset points',
                         fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                         zorder=15)
            any_plotted = True
        else:
            # 无插值：直接绘制原始数据（单点显示为散点，多点显示为折线+散点）
            plt.plot(epochs, vals, '-o', markersize=4, linewidth=1.0, label=label, zorder=4)
            # 标注最大值
            plt.scatter(max_epoch, max_val, s=80, c='red', marker='*', zorder=10,
                        label=f'{label} Max' if any_plotted is False else "")
            # 添加最大值文本标注
            plt.annotate(f'Max: {max_val:.4f}\nEpoch: {max_epoch}',
                         xy=(max_epoch, max_val),
                         xytext=(5, 5),  # 文本偏移量
                         textcoords='offset points',
                         fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                         zorder=15)
            any_plotted = True

    if not any_plotted:
        print(f"未绘制任何 {metric} 数据（无有效采样）")
        plt.close()
        return

    # 图表美化与保存
    plt.title(f"{metric} Comparison Across Models (With Max Values)", fontsize=14, pad=15)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 仅显示整数epoch
    plt.tight_layout()  # 自动调整布局，避免标签被截断

    # 保存图表（高分辨率）
    out_path = os.path.join(output_dir, f"{metric}_comparison_with_max.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存带最大值标注的图表: {out_path}")

def plot_metric_group(group_name, metrics, dfs, labels, output_dir):
    """绘制一组指标的对比图（需所有文件都包含该指标），新增显示曲线最大值功能"""
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
                
                # 计算当前曲线的最大值及对应epoch
                max_val = np.max(vals)
                max_epoch_idx = np.argmax(vals)
                max_epoch = epochs[max_epoch_idx]

                plt.plot(epochs, vals, marker='o', markersize=4, label=f"{label} - {metric}")
                
                # 标注最大值（散点+文本）
                plt.scatter(max_epoch, max_val, s=60, c='red', marker='*', zorder=10)
                plt.annotate(f'Max: {max_val:.4f}\nEpoch: {max_epoch}',
                             xy=(max_epoch, max_val),
                             xytext=(5, 5),
                             textcoords='offset points',
                             fontsize=8,
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6),
                             zorder=15)

    # 图表美化与保存
    plt.title(f"{group_name.replace('_', ' ').title()} Comparison (With Max Values)", fontsize=14, pad=15)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{group_name}_comparison_with_max.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存带最大值标注的分组图表: {out_path}")

def compare_metrics(file_paths, labels, output_dir, mode="group", metrics=None):
    """
    主函数：对比多个Excel文件的指标并绘图（支持显示最大值）
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
    print(f"所有带最大值标注的图表已保存到 {output_dir} 目录")

if __name__ == "__main__":
    # 1. 配置文件路径和模型标签（确保一一对应）
    file_paths = [
        "/root/autodl-fs/clip_log/2025_10_17-00_47_54-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_200-j_4-p_amp/result.xlsx", #clip
        #"/root/autodl-fs/clip_log/2025_08_29-15_19_36-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx", #clip
        #"/root/autodl-fs/clip_log/2025_08_31-10_53_29-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx",
        #"/root/autodl-fs/clip_log/2025_08_28-20_56_21-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx",
        #"/root/autodl-fs/clip_log/2025_08_29-22_19_58-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx",
        #"/root/autodl-fs/clip_log/2025_08_30-12_51_00-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx",
        #"/root/autodl-fs/clip_log/2025_08_31-23_04_40-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx",
        #"/root/autodl-fs/clip_log/2025_09_01-10_56_22-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx",
        #"/root/autodl-fs/clip_log/2025_09_19-00_07_41-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_16-j_4-p_amp/result.xlsx",
        #"/root/autodl-fs/clip_log/2025_09_30-21_35_55-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_32-j_4-p_amp/result.xlsx",
        #"/root/autodl-fs/clip_log/2025_10_01-11_24_01-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_32-j_4-p_amp/result.xlsx",
        "/root/MedCLIP-SAMv2-main/biomedclip_finetuning/open_clip/src/logs/2025_11_26-23_26_51-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx"
    ]
    labels = [
        "CLIP",
        #"CLIP_explicit_1",
        #"CLIP_explicit_2",
        #"CLIP_explicit_2_implicit_2_4_8_8",
        #"CLIP_explicit_2_implicit_2_8_16_32",
        #"CLIP_explicit_2_implicit_2_dynamic_8_20epoch",
        #"CLIP_explicit_2_implicit_2_dynamic_8_280epoch",
        #"CLIP_explicit_2_implicit_2_dynamic_8_0epoch_OrthoGradient",
        #"CLIP_explicit_2_implicit_2_dynamic_8_0epoch_OrthoGradient_distill_image",
        #"CLIP_explicit_2_implicit_2_dynamic_8_0epoch_OrthoGradient_double_distill",
        "CLIP_explicit_2_implicit_2_dynamic_8_0epoch_OrthoGradient",
        ""
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
    target_metric = "image_to_text_R@1"  # 目标指标
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