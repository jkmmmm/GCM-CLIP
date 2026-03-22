#!/bin/bash

# ===================== t-SNE可视化脚本 =====================
# 使用方法: ./run_tsne.sh
# 注意: 请确保tSNEanalysis.py已经修复了matplotlib兼容性问题
# ===========================================================

cd biomedclip_finetuning/open_clip/src || exit  # 增加cd失败退出

# GPU设备设置（可修改）
export CUDA_VISIBLE_DEVICES=0
echo "使用GPU设备: ${CUDA_VISIBLE_DEVICES}"

# 基础参数
MODEL_NAME0="CLIP"
MODEL_NAME1="CLIP_explicit_implicit_8_16_32_supervised"

# 模型检查点路径
CHECKPOINT="/root/autodl-fs/clip_log/2025_08_29-15_19_36-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/checkpoints/epoch_300.pt"
BASE_CHECKPOINT="/root/autodl-fs/clip_log/2025_11_26-23_26_51-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/checkpoints/best_model.pt"

# 数据配置
CONFIG="/root/autodl-tmp/data/forensic_CT_statistics.json"
VAL_DATA="/root/autodl-tmp/data/balanced_samples_4096_method2.csv"
OUTPUT_DIR="/root/MedCLIP-SAMv2-main/biomedclip_finetuning/open_clip/src/result/t_SNE"

# 数据加载参数
DATASET_TYPE="csv"
BATCH_SIZE=128
WORKERS=4
CSV_IMG_KEY="image_path"
CSV_CAPTION_KEY="original_text"
CSV_DISEASE_CATEGORY="category"
CSV_DISEASE_LOCATION="location"
CSV_SEPARATOR=","

# 模型参数
MODEL="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BASE_MODEL="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CLASS_NAMES="category"
FEATURE_TYPE="image"

# t-SNE基础参数（优化聚类的核心配置）
PERPLEXITY=30          # 提升困惑度，平衡局部/全局结构
MAX_ITER=20000         # 延长迭代，确保收敛
RANDOM_STATE=42
N_COMPONENTS=2

echo "====================================================="
echo "t-SNE可视化配置（优化同类聚集）"
echo "====================================================="
echo "模型对比: ${MODEL_NAME0} vs ${MODEL_NAME1}"
echo "特征类型: ${FEATURE_TYPE}"
echo "类别字段: ${CLASS_NAMES}"
echo "输出目录: ${OUTPUT_DIR}"
echo "====================================================="

# ===================== t-SNE调参选项 =====================
# 1. t-SNE计算方法
# TSNE_METHOD="barnes_hut"  # 大数据集使用，快速
TSNE_METHOD="exact"         # 小数据集精确计算

# 2. 学习率和优化参数（核心调参）
LEARNING_RATE=300.0         # 提升学习率，加速聚类收敛
EARLY_EXAGGERATION=15.0     # 提高早期放大因子，强化类间区分
MIN_GRAD_NORM=1e-7
N_ITER_WITHOUT_PROGRESS=500 # 放宽无进展阈值，避免提前终止

# 3. 距离度量方法
METRIC="cosine"             # 余弦距离适配CLIP高维特征

# 4. 初始化方法
INIT="pca"                  # PCA初始化，减少随机噪声

# 5. Barnes-Hut角度参数（仅method=barnes_hut时有效）
ANGLE=0.5

# 6. 特征预处理（关键：PCA降维+L2归一）
NORMALIZE_FEATURES="--normalize-features"  # L2归一化（必须）
STANDARDIZE="--standardize"                # 新增：标准化，统一特征分布
PCA_DIM="--pca-dim 50"                     # 新增：PCA预降维，过滤高维噪声

# 7. 多困惑度比较（可选）
# PERPLEXITY_LIST="--perplexity-list '5,15,30,50,100'"

# 8. 距离度量比较（可选）
# COMPARE_METRICS="--compare-metrics"

# ===================== 可视化选项 =====================
PLOT_STYLE="alpha_scatter"  # 透明度渐变，突出密集聚类
POINT_SIZE=30
ALPHA=0.7
COLORMAP="tab20"

# 图例和评估
SHOW_LEGEND="--show-legend"
LEGEND_LOC="best"
PLOT_CLUSTER_QUALITY="--plot-cluster-quality"

# 异常值检测（注释，避免干扰聚类可视化）
HIGHLIGHT_OUTLIERS=""
OUTLIER_THRESHOLD=""

# 保存嵌入向量
SAVE_EMBEDDINGS="--save-embeddings"
DPI=300

# ===================== 构建命令参数（修复语法错误） =====================
CMD="python3 -m open_clip_train.result_analysis.tSNEanalysis \
    --model-name0 \"${MODEL_NAME0}\" \
    --model-name1 \"${MODEL_NAME1}\" \
    --checkpoint \"${CHECKPOINT}\" \
    --base-checkpoint \"${BASE_CHECKPOINT}\" \
    --class-names \"${CLASS_NAMES}\" \
    --feature-type \"${FEATURE_TYPE}\" \
    --model \"${MODEL}\" \
    --base-model \"${BASE_MODEL}\" \
    --force-CMCLIP \
    --config \"${CONFIG}\" \
    --val-data \"${VAL_DATA}\" \
    --csv-img-key \"${CSV_IMG_KEY}\" \
    --csv-caption-key \"${CSV_CAPTION_KEY}\" \
    --csv-disease-category \"${CSV_DISEASE_CATEGORY}\" \
    --csv-disease-location \"${CSV_DISEASE_LOCATION}\" \
    --csv-separator \"${CSV_SEPARATOR}\" \
    --dataset-type \"${DATASET_TYPE}\" \
    --batch-size \"${BATCH_SIZE}\" \
    --workers \"${WORKERS}\" \
    --output-dir \"${OUTPUT_DIR}\" \
    --perplexity \"${PERPLEXITY}\" \
    --max-iter \"${MAX_ITER}\" \
    --random-state \"${RANDOM_STATE}\" \
    --n-components \"${N_COMPONENTS}\" \
    --tsne-method \"${TSNE_METHOD}\" \
    --learning-rate \"${LEARNING_RATE}\" \
    --early-exaggeration \"${EARLY_EXAGGERATION}\" \
    --min-grad-norm \"${MIN_GRAD_NORM}\" \
    --n-iter-without-progress \"${N_ITER_WITHOUT_PROGRESS}\" \
    --metric \"${METRIC}\" \
    --init \"${INIT}\" \
    --angle \"${ANGLE}\" \
    --plot-style \"${PLOT_STYLE}\" \
    --point-size \"${POINT_SIZE}\" \
    --alpha \"${ALPHA}\" \
    --colormap \"${COLORMAP}\" \
    --dpi \"${DPI}\" \
    --legend-loc \"${LEGEND_LOC}\" \
    ${SHOW_LEGEND} \
    ${PLOT_CLUSTER_QUALITY} \
    ${HIGHLIGHT_OUTLIERS} \
    ${OUTLIER_THRESHOLD} \
    ${SAVE_EMBEDDINGS} \
    ${NORMALIZE_FEATURES} \
    ${STANDARDIZE} \
    ${PCA_DIM}"

# 处理可选参数（避免空值干扰）
if [ -n "${PERPLEXITY_LIST}" ]; then
    CMD+=" ${PERPLEXITY_LIST}"
fi
if [ -n "${COMPARE_METRICS}" ]; then
    CMD+=" ${COMPARE_METRICS}"
fi

# ===================== 执行命令 =====================
echo ""
echo "====================================================="
echo "执行命令:"
echo "====================================================="
echo "${CMD}"
echo "====================================================="
echo ""

# 确认执行
read -p "是否执行上述命令? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "开始执行t-SNE可视化（优化同类聚集）..."
    echo ""
    # 使用bash -c执行，避免eval的参数解析问题
    bash -c "${CMD}"
else
    echo "已取消执行。"
    exit 0
fi

# ===================== 参数调优建议 =====================
echo ""
echo "====================================================="
echo "t-SNE参数调优建议（同类聚集专用）:"
echo "====================================================="
echo "1. 困惑度 (perplexity): 30是最优值（4096样本），勿超过50"
echo "2. 学习率: 300-400区间效果最佳，过高会导致分散"
echo "3. 早期放大因子: 12-15，越高类间区分越明显"
echo "4. 迭代次数: 至少20000，确保t-SNE完全收敛"
echo "5. 特征预处理: PCA降维（50维）+标准化+L2归一必须开启"
echo "====================================================="