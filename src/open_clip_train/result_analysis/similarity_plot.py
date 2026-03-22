import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import os  # 新增导入，用于文件路径处理

# -------------------------- 可配置参数 --------------------------
CSV_FILE_PATH = "/root/MedCLIP-SAMv2-main/similarity_analysis_results/similarity_analysis_20251127_162202/raw_similarity_data.csv"  # 替换为你的CSV文件实际路径
TARGET_HEALTH_CATEGORY = "Healthy"  # 指定要分析的health_category类别
# ----------------------------------------------------------------

# 1. 读取CSV文件
df = pd.read_csv(CSV_FILE_PATH)

# 2. 查看所有可用的health_category类别（方便用户核对）
available_categories = df["health_category"].unique()
print(f"文件中可用的health_category类别：{available_categories}")

# 3. 筛选指定health_category的数据，并进行数据校验
filtered_df = df[df["health_category"] == TARGET_HEALTH_CATEGORY]
if filtered_df.empty:
    raise ValueError(f"错误：未找到health_category为【{TARGET_HEALTH_CATEGORY}】的数据！")
print(f"成功筛选出【{TARGET_HEALTH_CATEGORY}】类别的数据，共 {len(filtered_df)} 行")

# 4. 按comparison_type分组提取similarity_value数据，并清理空值
same_data = filtered_df[filtered_df["comparison_type"] == "same"]["similarity_value"].dropna()
different_data = filtered_df[filtered_df["comparison_type"] == "different"]["similarity_value"].dropna()

# 校验分组后的数据是否非空
if same_data.empty:
    raise ValueError(f"【{TARGET_HEALTH_CATEGORY}】类别下无comparison_type为same的数据！")
if different_data.empty:
    raise ValueError(f"【{TARGET_HEALTH_CATEGORY}】类别下无comparison_type为different的数据！")

# 5. 计算两个分布的重叠程度（Overlap）
def calculate_overlap(data1, data2):
    # 拟合KDE概率密度曲线
    kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)
    # 生成x轴采样点（0到1，共2000个点，提高精度）
    x = np.linspace(0, 1, 2000)
    # 计算两个分布在x点上的密度值
    pdf1 = kde1(x)
    pdf2 = kde2(x)
    # 积分重叠区域（梯形积分法）
    overlap = np.trapz(np.minimum(pdf1, pdf2), x)
    return x, pdf1, pdf2, overlap

# 计算KDE数据和重叠值
x_grid, pdf_same, pdf_diff, overlap_value = calculate_overlap(same_data, different_data)

# 6. 绘制概率密度图（参考指定样式）
# 配置颜色方案
# disease_same_color = (66/255, 145/255, 178/255)    # RGB(66,145,178) # 蓝色调
# disease_diff_color = (170/255, 78/255, 126/255)    # RGB(170,78,126)
disease_same_color = (241/255, 153/255, 26/255)      # RGB(241,153,26) # 橙色调
disease_diff_color = (203/255, 80/255, 51/255)       # RGB(203,80,51)
overlap_hatch = '////'
overlap_alpha = 0.3
bg_color = "white"

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(14, 8), facecolor=bg_color)  # 增大图形尺寸
ax = plt.gca()

# 绘制Same Disease的KDE曲线和填充
ax.plot(x_grid, pdf_same, label=f"Same Disease (n={len(same_data)})",
        color=disease_same_color, linewidth=3.0, alpha=0.9)
ax.fill_between(x_grid, pdf_same, alpha=0.4, color=disease_same_color)

# 绘制Different Disease的KDE曲线和填充（虚线）
ax.plot(x_grid, pdf_diff, label=f"Different Disease (n={len(different_data)})",
        color=disease_diff_color, linewidth=3.0, alpha=0.9, linestyle='--')
ax.fill_between(x_grid, pdf_diff, alpha=0.3, color=disease_diff_color)

# 绘制重叠区域（斜线填充）
mask = np.minimum(pdf_same, pdf_diff) > 0
ax.fill_between(x_grid, np.minimum(pdf_same, pdf_diff), where=mask,
                facecolor='gray', alpha=overlap_alpha, hatch=overlap_hatch,
                edgecolor='black', linewidth=0.5)

# 7. 美化图表和坐标轴
# 设置标题
ax.set_title(
    f"Disease Similarity Density — {TARGET_HEALTH_CATEGORY}\nOverlap: {overlap_value:.3f}",
    fontsize=16, fontweight='bold', pad=20
)

# 设置轴标签
ax.set_xlabel("Similarity Value: text-text", fontsize=18, fontweight='bold')
ax.set_ylabel("Probability Density", fontsize=18, fontweight='bold')

# -------------------------- 关键修改：设置刻度数字大小 --------------------------
# 设置x轴和y轴刻度标签的字体大小（可根据需求调整数值，比如16/18/20）
ax.tick_params(axis='x', labelsize=20)  # x轴刻度数字大小
ax.tick_params(axis='y', labelsize=20)  # y轴刻度数字大小
# --------------------------------------------------------------------------------

# 设置轴范围
ax.set_xlim(0, 1)
y_max = max(pdf_same.max(), pdf_diff.max())
ax.set_ylim(0, y_max * 1.1)

# 美化坐标轴（隐藏上、右边框）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# 优化网格
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# 优化图例（放在右上角外部）
legend = ax.legend(frameon=True, fontsize=20, loc='upper left',
                   fancybox=True, shadow=True, framealpha=0.9,
                   bbox_to_anchor=(0.02, 0.98))
legend.get_frame().set_facecolor('white')

# 调整布局
plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # 为图例留出空间

# 8. 保存图片
save_path = os.path.join(os.path.dirname(CSV_FILE_PATH), f"{TARGET_HEALTH_CATEGORY}_density.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=bg_color,
            edgecolor='none', transparent=False)
print(f"图片已保存至：{save_path}")

# 显示图片（可选）
# plt.show()
plt.close()