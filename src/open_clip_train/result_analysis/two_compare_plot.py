import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator

# 设置中文显示
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def load_metrics(file_path):
    """加载指标Excel文件"""
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"读取文件 {file_path} 出错: {e}")
        return None

def get_metric_groups():
    """定义指标分组，每组指标将绘制在同一张图上"""
    return {
        "image_to_text_rank": [
            #"image_to_text_mean_rank",
            "image_to_text_R@1",
            "image_to_text_R@2",
            "image_to_text_R@5",
            "image_to_text_R@10"
        ],
        "text_to_image_rank": [
            #"text_to_image_mean_rank",
            "text_to_image_R@1",
            "text_to_image_R@2",
            "text_to_image_R@5"
        ],
        "zeroshot_joint": [
            "med-zeroshot-val-joint-top1",
            "med-zeroshot-val-joint-top2",
            "med-zeroshot-val-joint-top5",
            "med-zeroshot-val-joint-top10"
        ],
        "zeroshot_category": [
            "med-zeroshot-val-category-top1",
            "med-zeroshot-val-category-top2",
            "med-zeroshot-val-category-top5"
        ],
        "zeroshot_location": [
            "med-zeroshot-val-location-top1",
            "med-zeroshot-val-location-top2",
            "med-zeroshot-val-location-top5",
            "med-zeroshot-val-location-top10"
        ]
    }

def plot_metric_group(group_name, metrics, df1, df2, label1, label2, output_dir):
    """绘制一组指标的对比图"""
    # 筛选存在的指标
    existing_metrics = [m for m in metrics if m in df1.columns and m in df2.columns]
    if not existing_metrics:
        print(f"警告: 分组 {group_name} 中没有共同的指标，跳过绘图")
        return
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 获取最大的epoch数，用于统一x轴范围
    max_epoch = max(df1['epoch'].max(), df2['epoch'].max())
    
    # 为每个指标绘制曲线
    for metric in existing_metrics:
        # 绘制第一个文件的数据
        if not df1[metric].isna().all():
            plt.plot(df1['epoch'], df1[metric], marker='o', label=f"{label1} - {metric}")
        
        # 绘制第二个文件的数据
        if not df2[metric].isna().all():
            plt.plot(df2['epoch'], df2[metric], marker='s', label=f"{label2} - {metric}")
    
    # 设置图表属性
    plt.title(f'{group_name.replace("_", " ").title()} comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Indicator')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在图表右侧
    plt.tight_layout()  # 自动调整布局
    
    # 设置x轴为整数
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 保存图表
    output_path = os.path.join(output_dir, f'{group_name}_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表: {output_path}")

def compare_metrics(file1, file2, label1="模型A", label2="模型B", output_dir="metric_comparisons"):
    """对比两个指标Excel文件并生成可视化图表"""
    # 加载数据
    df1 = load_metrics(file1)
    df2 = load_metrics(file2)
    
    if df1 is None or df2 is None:
        print("无法加载数据，程序退出")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取指标分组
    metric_groups = get_metric_groups()
    
    # 为每个分组绘制对比图
    for group_name, metrics in metric_groups.items():
        plot_metric_group(group_name, metrics, df1, df2, label1, label2, output_dir)
    
    print(f"所有图表已保存到 {output_dir} 目录")

if __name__ == "__main__":
    # 两个Excel文件的路径
    file_path1 = "/root/autodl-fs/clip_log/2025_08_29-22_19_58-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx"  # 替换为第一个文件的实际路径
    file_path2 = "/root/MedCLIP-SAMv2-main/biomedclip_finetuning/open_clip/src/logs/2025_09_01-10_56_22-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/result.xlsx"  # 替换为第二个文件的实际路径
    # 为两个文件设置标签（将显示在图表中）
    label1 = "explicit_implicit_supervise_2_4_8_8_20_supervise_2"  # 可根据实际情况修改
    label2 = "explicit_implicit_dynamic_8_280_supervise_2"  # 可根据实际情况修改
    result_path = f"/root/MedCLIP-SAMv2-main/biomedclip_finetuning/open_clip/src/result/{label1}_{label2}"
    
    
    
    # 执行对比并生成图表
    compare_metrics(file_path1, file_path2, label1, label2, result_path)
