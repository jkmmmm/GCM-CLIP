import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import json

# 设置Nature期刊风格的绘图参数
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.constrained_layout.use': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})

# 定义更美观的配色方案（基于Seaborn的Set3调色板，适合分类数据）
beautiful_colors = [
    '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
    '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
    '#ccebc5', '#ffed6f', '#e41a1c', '#377eb8', '#4daf4a',
    '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf',
    '#999999', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
    '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'
]

# 宏平均曲线使用特殊的颜色
macro_color = '#1a1a1a'  # 深黑色，突出显示

# 读取数据
dirname = "/root/MedCLIP-SAMv2-main/similarity_analysis_results/similarity_analysis_20251215_202226"
df = pd.read_csv(os.path.join(dirname, 'raw_similarity_data.csv'))

# 读取JSON文件获取映射
json_path = '/root/autodl-tmp/data/forensic_CT_statistics.json'
with open(json_path, 'r') as f:
    forensic_data = json.load(f)

# 提取疾病名称映射和位置映射
disease_mapping = forensic_data['category']
location_mapping = forensic_data['location']

# 分离health和location数据
health_df = df[df['condition_name'].str.startswith('health_')].copy()
location_df = df[df['condition_name'].str.startswith('location_')].copy()

print(f"总数据行数: {len(df)}")
print(f"health数据行数: {len(health_df)}")
print(f"location数据行数: {len(location_df)}")

# 处理health数据
def process_data(df, data_type='health'):
    """处理数据并计算ROC曲线"""
    if data_type == 'health':
        mapping = disease_mapping
        prefix = 'health_'
    else:  # location
        mapping = location_mapping
        prefix = 'location_'
    
    # 提取基础condition_name（去掉_same和_different后缀）
    df['base_condition'] = df['condition_name'].str.replace('_same$', '', regex=True).str.replace('_different$', '', regex=True)
    
    # 从base_condition中提取数字部分
    df['condition_num'] = df['base_condition'].str.extract(rf'{prefix}(\d+)')[0]
    
    # 处理NaN值
    df['condition_num'] = pd.to_numeric(df['condition_num'], errors='coerce').fillna(0).astype(int)
    
    # 映射到名称（注意：JSON中从1开始，所以需要condition_num+1）
    df['mapped_name'] = df['condition_num'].apply(
        lambda x: mapping.get(str(x+1), f"Unknown_{x}")
    )
    
    return df

# 处理health数据
if not health_df.empty:
    health_df = process_data(health_df, 'health')
    health_base_conditions = health_df['base_condition'].unique()
    print(f"\n找到 {len(health_base_conditions)} 个health基础condition:")
    for i, cond in enumerate(health_base_conditions):
        disease_name = health_df[health_df['base_condition'] == cond]['mapped_name'].iloc[0]
        print(f"  {i+1}. {cond} -> {disease_name}")

# 处理location数据
if not location_df.empty:
    location_df = process_data(location_df, 'location')
    location_base_conditions = location_df['base_condition'].unique()
    print(f"\n找到 {len(location_base_conditions)} 个location基础condition:")
    for i, cond in enumerate(location_base_conditions):
        location_name = location_df[location_df['base_condition'] == cond]['mapped_name'].iloc[0]
        print(f"  {i+1}. {cond} -> {location_name}")

def plot_roc_curves(df, data_type='health'):
    """绘制ROC曲线，包含宏平均AUC"""
    if df.empty:
        print(f"\n没有{data_type}数据，跳过绘制ROC曲线")
        return None, None, None
    
    base_conditions = df['base_condition'].unique()
    
    # 页面大小：183.01 x 137.24 (单位，假设为毫米)
    page_width = 183.01  # 页面宽度
    page_height = 137.24  # 页面高度
    
    # 图层位置：左起12.01%，上起4.01%，宽80%，高60%
    left_margin = 0.1201  # 左起12.01%
    top_margin = 0.0401  # 上起4.01%
    plot_width = 0.8  # 宽度80%
    plot_height = 0.6  # 高度60%
    
    # 计算实际绘图区域在页面上的位置
    left = left_margin
    bottom = 1 - top_margin - plot_height
    width = plot_width
    height = plot_height
    
    # 创建图形
    fig = plt.figure(figsize=(page_width/25.4, page_height/25.4))
    ax = fig.add_axes([left, bottom, width, height])
    
    # 存储每个condition的AUC值
    auc_results = {}
    valid_conditions = []
    
    # 用于存储每个类别的ROC曲线数据（用于计算宏平均）
    fprs_list = []
    tprs_list = []
    
    # 为每个基础condition_name绘制ROC曲线
    for i, base_cond in enumerate(base_conditions):
        # 提取当前基础condition的所有数据
        condition_data = df[df['base_condition'] == base_cond]
        
        # 获取映射名称
        mapped_name = condition_data['mapped_name'].iloc[0]
        
        # 根据comparison_type创建标签：same为1，different为0
        y_true = np.where(condition_data['comparison_type'] == 'same', 1, 0)
        
        # 检查是否同时包含正负样本
        if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
            print(f"警告: {base_cond} ({mapped_name}) 只包含一种类型的样本 (same: {np.sum(y_true)}, different: {len(y_true)-np.sum(y_true)})")
            continue
        
        valid_conditions.append((base_cond, mapped_name))
        
        # 获取相似度值作为预测分数
        y_scores = condition_data['similarity_value'].values
        
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # 计算AUC值
        roc_auc = auc(fpr, tpr)
        auc_results[mapped_name] = roc_auc
        
        # 存储ROC数据用于宏平均
        fprs_list.append(fpr)
        tprs_list.append(tpr)
        
        # 获取当前颜色
        color = beautiful_colors[i % len(beautiful_colors)]
        
        # 绘制ROC曲线 - 使用更细的线宽和适当的透明度
        ax.plot(fpr, tpr, color=color, lw=1.2, alpha=0.7,
                 label=f'{mapped_name} (AUC = {roc_auc:.3f})')
    
    # 如果没有有效的condition，则退出
    if not valid_conditions:
        print(f"没有找到同时包含正负样本的{data_type} condition!")
        plt.close()
        return None, None, None
    
    # ==================== 计算宏平均AUC ====================
    # 宏平均AUC（Macro-average AUC）：直接计算各个类别AUC的平均值
    macro_auc = np.mean(list(auc_results.values()))
    
    # 绘制宏平均ROC曲线（通过插值方法）
    # 定义通用的FPR点
    mean_fpr = np.linspace(0, 1, 100)
    mean_tprs = []
    
    for fpr, tpr in zip(fprs_list, tprs_list):
        # 对每个ROC曲线在mean_fpr点进行插值
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # 确保从(0,0)开始
        mean_tprs.append(interp_tpr)
    
    # 计算平均TPR
    mean_tpr = np.mean(mean_tprs, axis=0)
    mean_tpr[-1] = 1.0  # 确保结束于(1,1)
    
    # ==================== 绘制宏平均曲线 ====================
    # 绘制宏平均ROC曲线 - 使用特殊的颜色和较粗的线宽
    ax.plot(mean_fpr, mean_tpr, color=macro_color, lw=2.5, linestyle='-', alpha=1.0,
             label=f'Macro-average (AUC = {macro_auc:.3f})')
    
    # 添加对角线（随机分类器）- 使用灰色，较细的线
    ax.plot([0, 1], [0, 1], color='#808080', lw=1.2, linestyle=':', alpha=0.8, label='Random (AUC = 0.500)')
    
    # ==================== 设置图形属性 ====================
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    
    if data_type == 'health':
        title = 'ROC Curves for Different Diseases'
    else:
        title = 'ROC Curves for Different Locations'
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # 美化图例
    if data_type == 'health':
        # 疾病类别较多，使用2列图例
        legend = ax.legend(loc='lower right', ncol=2, fontsize=4,
                            frameon=True, fancybox=True, shadow=False,
                            framealpha=0.95, edgecolor='#333333',
                            bbox_to_anchor=(0.98, 0.02),
                            borderpad=0.3,
                            labelspacing=0.2,
                            handlelength=1.2,
                            handletextpad=0.3,
                            columnspacing=0.4)
    else:
        # 位置类别较少，使用1列图例
        legend = ax.legend(loc='lower right', ncol=1, fontsize=8,
                           frameon=True, fancybox=True, shadow=False,
                           framealpha=0.95, edgecolor='#333333')
    
    # 设置网格样式 - 更淡的网格线
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # 保存图片
    if data_type == 'health':
        filename = 'roc_curves_diseases_with_macro_avg.png'
    else:
        filename = 'roc_curves_locations_with_macro_avg.png'
    
    plt.savefig(os.path.join(dirname, filename), dpi=600, bbox_inches='tight')
    print(f"\n{data_type} ROC曲线图已保存为 '{filename}'，包含 {len(valid_conditions)} 个类别")
    
    # 关闭图形，释放内存
    plt.close()
    
    # 返回结果（只包含宏平均AUC）
    return auc_results, valid_conditions, macro_auc

# 绘制health数据的ROC曲线
print("\n" + "="*70)
print("处理Health数据:")
print("="*70)
if not health_df.empty:
    health_auc_results, health_valid_conditions, health_macro_auc = plot_roc_curves(health_df, 'health')
    
    if health_auc_results:
        # 打印AUC结果
        print("\nAUC Values for Each Disease:")
        print("="*70)
        for disease, auc_value in health_auc_results.items():
            print(f"{disease}: {auc_value:.4f}")
        print("="*70)
        
        # 打印宏平均AUC
        print(f"\nMacro-average AUC for Diseases: {health_macro_auc:.4f}")
        print("="*70)
        
        # 按AUC值排序
        sorted_auc = sorted(health_auc_results.items(), key=lambda x: x[1], reverse=True)
        print("\nDiseases Sorted by AUC (descending):")
        print("="*70)
        for disease, auc_value in sorted_auc:
            print(f"{disease}: {auc_value:.4f}")
        print("="*70)
        
        # 保存结果到文件
        results_data = []
        for base_cond, disease_name in health_valid_conditions:
            if disease_name in health_auc_results:
                results_data.append({
                    'Original_Condition': base_cond,
                    'Disease_Name': disease_name,
                    'AUC': health_auc_results[disease_name]
                })
        
        # 添加宏平均AUC到结果文件
        results_df = pd.DataFrame(results_data)
        macro_avg_df = pd.DataFrame([{
            'Original_Condition': 'Average',
            'Disease_Name': 'Macro-average',
            'AUC': health_macro_auc
        }])
        
        # 合并所有结果
        all_results_df = pd.concat([results_df, macro_avg_df], ignore_index=True)
        all_results_df.to_csv(os.path.join(dirname, 'roc_auc_results_diseases_with_macro_avg.csv'), index=False)
        print("\nAUC结果（包含宏平均AUC）已保存到 'roc_auc_results_diseases_with_macro_avg.csv'")

# 绘制location数据的ROC曲线
print("\n" + "="*70)
print("处理Location数据:")
print("="*70)
if not location_df.empty:
    location_auc_results, location_valid_conditions, location_macro_auc = plot_roc_curves(location_df, 'location')
    
    if location_auc_results:
        # 打印AUC结果
        print("\nAUC Values for Each Location:")
        print("="*70)
        for location, auc_value in location_auc_results.items():
            print(f"{location}: {auc_value:.4f}")
        print("="*70)
        
        # 打印宏平均AUC
        print(f"\nMacro-average AUC for Locations: {location_macro_auc:.4f}")
        print("="*70)
        
        # 按AUC值排序
        sorted_auc = sorted(location_auc_results.items(), key=lambda x: x[1], reverse=True)
        print("\nLocations Sorted by AUC (descending):")
        print("="*70)
        for location, auc_value in sorted_auc:
            print(f"{location}: {auc_value:.4f}")
        print("="*70)
        
        # 保存结果到文件
        results_data = []
        for base_cond, location_name in location_valid_conditions:
            if location_name in location_auc_results:
                results_data.append({
                    'Original_Condition': base_cond,
                    'Location_Name': location_name,
                    'AUC': location_auc_results[location_name]
                })
        
        # 添加宏平均AUC到结果文件
        results_df = pd.DataFrame(results_data)
        macro_avg_df = pd.DataFrame([{
            'Original_Condition': 'Average',
            'Location_Name': 'Macro-average',
            'AUC': location_macro_auc
        }])
        
        # 合并所有结果
        all_results_df = pd.concat([results_df, macro_avg_df], ignore_index=True)
        all_results_df.to_csv(os.path.join(dirname, 'roc_auc_results_locations_with_macro_avg.csv'), index=False)
        print("\nAUC结果（包含宏平均AUC）已保存到 'roc_auc_results_locations_with_macro_avg.csv'")

# 打印汇总统计
print("\n" + "="*70)
print("汇总统计:")
print("="*70)

if not health_df.empty:
    health_base_conditions = health_df['base_condition'].unique()
    print(f"Health总基础condition数: {len(health_base_conditions)}")
    if 'health_auc_results' in locals() and health_auc_results:
        print(f"Health有效疾病类别数: {len(health_valid_conditions)}")
        print(f"Health跳过的condition数: {len(health_base_conditions) - len(health_valid_conditions)}")
        if 'health_macro_auc' in locals():
            print(f"Health宏平均AUC: {health_macro_auc:.4f}")

if not location_df.empty:
    location_base_conditions = location_df['base_condition'].unique()
    print(f"Location总基础condition数: {len(location_base_conditions)}")
    if 'location_auc_results' in locals() and location_auc_results:
        print(f"Location有效位置类别数: {len(location_valid_conditions)}")
        print(f"Location跳过的condition数: {len(location_base_conditions) - len(location_valid_conditions)}")
        if 'location_macro_auc' in locals():
            print(f"Location宏平均AUC: {location_macro_auc:.4f}")

# 打印映射表
print("\n" + "="*70)
print("疾病名称映射表:")
print("="*70)
if not health_df.empty:
    print(f"{'原始Condition':<20} {'疾病名称':<40} {'ID映射':<10}")
    print("-"*70)
    for base_cond in health_df['base_condition'].unique():
        condition_data = health_df[health_df['base_condition'] == base_cond]
        if len(condition_data) > 0:
            condition_num = condition_data['condition_num'].iloc[0]
            disease_name = condition_data['mapped_name'].iloc[0]
            print(f"{base_cond:<20} {disease_name:<40} health_{condition_num}->{condition_num+1}")

print("\n" + "="*70)
print("位置名称映射表:")
print("="*70)
if not location_df.empty:
    print(f"{'原始Condition':<20} {'位置名称':<30} {'ID映射':<10}")
    print("-"*70)
    for base_cond in location_df['base_condition'].unique():
        condition_data = location_df[location_df['base_condition'] == base_cond]
        if len(condition_data) > 0:
            condition_num = condition_data['condition_num'].iloc[0]
            location_name = condition_data['mapped_name'].iloc[0]
            print(f"{base_cond:<20} {location_name:<30} location_{condition_num}->{condition_num+1}")

print("\n" + "="*70)
print("处理完成!")
print("="*70)