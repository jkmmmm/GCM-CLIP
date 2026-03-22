import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import torch
import os
import json
import sys
from datetime import datetime
from typing import Optional, Tuple, Dict, List
import argparse
from argparse import Namespace
from scipy import stats
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib



# 导入项目核心模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")
from open_clip import create_model_and_transforms, get_tokenizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device
from open_clip_train.precision import get_autocast
from open_clip_train.file_utils import pt_load
from open_clip_train.params import parse_args as parse_original_args


def parse_args():
    """解析独立运行时的命令行参数"""
    parser = argparse.ArgumentParser(description='t-SNE Visualization for CLIP/CM-CLIP Features')
    
    # 模型配置
    parser.add_argument('--model', type=str, required=True, help='模型名称，如 "ViT-B-32"')
    parser.add_argument("--force-CMCLIP", default=False, action='store_true', help="Force use of category mining model (separate text-tower).")
    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--base-model', type=str, help='原始CLIP模型名称，用于对比（可选）')
    parser.add_argument('--base-checkpoint', type=str, help='原始CLIP模型checkpoint路径（可选）')
    parser.add_argument('--model-name0', type=str, required=True, help='模型0的显示名称')
    parser.add_argument('--model-name1', type=str, required=True, help='模型1的显示名称')
    
    # 数据配置
    parser.add_argument('--config', type=str, required=True, help='数据集配置文件路径')
    parser.add_argument('--val-data', type=str, required=True, help='验证集CSV文件路径')
    parser.add_argument("--dataset-type", choices=["webdataset", "csv", "synthetic", "auto"], default="auto", help="Which type of dataset to process.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers per GPU.")
    parser.add_argument("--csv-img-key", type=str, default="filepath", help="For csv-like datasets, the name of the key for the image paths.")
    parser.add_argument("--csv-caption-key", type=str, default="title", help="For csv-like datasets, the name of the key for the captions.")
    parser.add_argument("--csv-disease-category", type=str, default=None, help="For csv-like datasets, the name of the key for the disease category.")
    parser.add_argument("--csv-disease-location", type=str, default=None, help="For csv-like datasets, the name of the key for the disease locations.")
    parser.add_argument("--csv-separator", type=str, default="\t", help="For csv-like datasets, which separator to use.")
    parser.add_argument("--imagenet-val", type=str, default=None, help="Path to imagenet val set for conducting zero shot evaluation.")
    parser.add_argument("--imagenet-v2", type=str, default=None, help="Path to imagenet v2 for conducting zero shot evaluation.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override system default cache path for model & tokenizer file downloads.")
    
    # t-SNE参数
    parser.add_argument('--tsne-method', type=str, default='barnes_hut', choices=['barnes_hut', 'exact'],
                       help='t-SNE计算方法: barnes_hut(快速,大数据集)或exact(精确,小数据集)')
    parser.add_argument('--perplexity', type=float, default=30.0, help='t-SNE困惑度参数(通常5-50)')
    parser.add_argument('--early-exaggeration', type=float, default=12.0, help='早期放大因子')
    parser.add_argument('--learning-rate', type=float, default=200.0, help='学习率(通常10-1000)')
    parser.add_argument('--max-iter', type=int, default=1000, help='t-SNE优化迭代次数')
    parser.add_argument('--n-iter-without-progress', type=int, default=300, help='无进展时的最大迭代次数')
    parser.add_argument('--min-grad-norm', type=float, default=1e-7, help='梯度范数最小值阈值')
    parser.add_argument('--metric', type=str, default='euclidean', choices=['euclidean', 'cosine', 'manhattan', 'chebyshev'],
                       help='距离度量方法')
    parser.add_argument('--init', type=str, default='random', choices=['random', 'pca'],
                       help='初始化方法: random或pca')
    parser.add_argument('--angle', type=float, default=0.5, help='Barnes-Hut算法的角度参数(0.2-0.8)')
    parser.add_argument('--random-state', type=int, default=42, help='随机种子')
    parser.add_argument('--feature-type', type=str, default='image', choices=['image', 'text'], 
                      help='特征类型：image或text')
    parser.add_argument('--n-components', type=int, default=2, choices=[2, 3],
                       help='降维后的维度数(2或3)')
    parser.add_argument('--perplexity-list', type=str, default=None,
                       help='多个困惑度参数列表，用逗号分隔，如"5,15,30,50"')
    parser.add_argument('--compare-metrics', action='store_true',
                       help='比较不同距离度量方法的效果')
    
    # 特征预处理参数
    parser.add_argument('--normalize-features', action='store_true',
                       help='对特征进行L2归一化')
    parser.add_argument('--pca-dim', type=int, default=None,
                       help='在t-SNE之前先使用PCA降维到指定维度')
    parser.add_argument('--standardize', action='store_true',
                       help='对特征进行标准化(零均值,单位方差)')
    
    # 可视化参数
    parser.add_argument('--output-dir', type=str, default='./tsne_results', help='可视化结果保存目录')
    parser.add_argument('--class-names', type=str, default='category', 
                      help='配置文件中的类别名称字段，如"category"或"location"')
    parser.add_argument('--plot-style', type=str, default='scatter', 
                      choices=['scatter', 'density', 'hexbin', 'kde', 'alpha_scatter'],
                      help='绘图样式')
    parser.add_argument('--point-size', type=int, default=30, help='散点大小')
    parser.add_argument('--alpha', type=float, default=0.7, help='透明度')
    parser.add_argument('--colormap', type=str, default='tab20', help='颜色映射方案')
    parser.add_argument('--show-legend', action='store_true', help='显示图例')
    parser.add_argument('--legend-loc', type=str, default='best', 
                      choices=['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right',
                              'center left', 'center right', 'lower center', 'upper center', 'center'],
                      help='图例位置')
    parser.add_argument('--dpi', type=int, default=300, help='输出图像DPI')
    parser.add_argument('--save-embeddings', action='store_true', help='保存降维后的嵌入向量')
    parser.add_argument('--plot-cluster-quality', action='store_true', 
                      help='绘制聚类质量评估指标')
    parser.add_argument('--highlight-outliers', action='store_true',
                       help='高亮显示异常值')
    parser.add_argument('--outlier-threshold', type=float, default=3.0,
                       help='异常值检测的阈值(基于Z-score)')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='计算设备：cuda或cpu')
    parser.add_argument('--precision', type=str, default='amp', choices=['amp', 'fp16', 'fp32'],
                      help='计算精度')
    

    return parser.parse_args()


def get_input_dtype(precision):
    """根据精度设置获取输入数据类型"""
    if precision == 'fp16':
        return torch.float16
    elif precision == 'fp32':
        return torch.float32
    else:  # amp or amp_bfloat16
        return torch.float32


def preprocess_features(features: np.ndarray, args) -> np.ndarray:
    """对特征进行预处理"""
    if args.standardize:
        # 标准化：零均值，单位方差
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    
    if args.normalize_features:
        # L2归一化
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / norms
    
    if args.pca_dim is not None and args.pca_dim < features.shape[1]:
        # PCA降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=args.pca_dim, random_state=args.random_state)
        features = pca.fit_transform(features)
        print(f"PCA降维: {features.shape[1]} -> {args.pca_dim}, 保留方差: {np.sum(pca.explained_variance_ratio_):.3f}")
    
    return features


def compute_cluster_metrics(features_2d: np.ndarray, labels: np.ndarray) -> Dict:
    """计算聚类质量指标"""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.neighbors import NearestNeighbors
    
    metrics = {}
    
    try:
        # 轮廓系数 (-1到1，越大越好)
        metrics['silhouette'] = silhouette_score(features_2d, labels)
    except:
        metrics['silhouette'] = None
    
    try:
        # Calinski-Harabasz指数 (越大越好)
        metrics['calinski_harabasz'] = calinski_harabasz_score(features_2d, labels)
    except:
        metrics['calinski_harabasz'] = None
    
    try:
        # Davies-Bouldin指数 (越小越好)
        metrics['davies_bouldin'] = davies_bouldin_score(features_2d, labels)
    except:
        metrics['davies_bouldin'] = None
    
    # 类内平均距离
    unique_labels = np.unique(labels)
    intra_distances = []
    for label in unique_labels:
        cluster_points = features_2d[labels == label]
        if len(cluster_points) > 1:
            centroid = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            intra_distances.append(np.mean(distances))
    
    metrics['avg_intra_distance'] = np.mean(intra_distances) if intra_distances else None
    
    return metrics


def detect_outliers(features: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """检测异常值"""
    # 计算每个维度的Z-score
    z_scores = np.abs(stats.zscore(features, axis=0))
    # 标记任何维度超过阈值的点为异常值
    outlier_mask = np.any(z_scores > threshold, axis=1)
    return outlier_mask


def load_model(args, model_name: str, checkpoint_path: str, device: torch.device):
    """加载模型（兼容原始CLIP和CM-CLIP）"""
    model_kwargs = {}
    if 'siglip' in model_name.lower():
        model_kwargs['init_logit_scale'] = np.log(10)
        model_kwargs['init_logit_bias'] = -10
        
    model, _, preprocess_val = create_model_and_transforms(
        model_name,
        pretrained=checkpoint_path,
        precision=args.precision,
        force_CMCLIP=args.force_CMCLIP,
        device=device,
        jit=False,
        output_dict=True,
        **model_kwargs
    )
    print(f"成功加载模型: {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    return model, preprocess_val


def prepare_data(args, preprocess_val, tokenizer):
    """准备数据集"""
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    args.train_data = None
    
    data = get_data(
        args,
        (preprocess_val, preprocess_val),
        config=config,
        epoch=0,
        tokenizer=tokenizer,
    )
    
    if 'val' not in data:
        raise ValueError("数据加载失败，未找到验证集")
    
    return data['val'].dataloader


def extract_features(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    args,
    device: torch.device,
    feature_type: str = "image"
) -> Tuple[np.ndarray, np.ndarray]:
    """提取特征和标签"""
    all_features = []
    all_labels = []
    
    input_dtype = get_input_dtype(args.precision)
    autocast = get_autocast(args.precision, device_type=device.type)
    
    model.eval()
    with torch.inference_mode():
        for batch in data_loader:
            images, texts, category, location = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            
            with autocast():
                if feature_type == "image":
                    model_out = model(image=images)
                    batch_features = model_out["image_features"].cpu()
                else:
                    model_out = model(text=texts)
                    batch_features = model_out["text_features"].cpu()
            
            if args.class_names == 'category':
                all_labels.append(category.cpu())
            else:
                all_labels.append(location.cpu())
                
            all_features.append(batch_features)
    
    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    # 将 one-hot 编码转换为类别索引
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    
    return features, labels


def get_colormap(colormap_name: str, n_colors: int):
    """获取颜色映射，兼容不同版本的matplotlib"""
    try:
        # 新版本matplotlib
        if hasattr(plt, 'colormaps'):
            cmap = plt.colormaps[colormap_name]
            if n_colors > 0:
                return cmap(np.linspace(0, 1, n_colors))
            return cmap
        else:
            # 旧版本matplotlib
            cmap = plt.cm.get_cmap(colormap_name, n_colors)
            return [cmap(i) for i in range(n_colors)]
    except Exception as e:
        print(f"警告: 无法获取颜色映射 '{colormap_name}', 使用默认值: {e}")
        # 使用seaborn的调色板作为备选
        return sns.color_palette("husl", n_colors)


def visualize_tsne_2d(
    args,
    features_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    class_names: List[str],
    output_path: str,
    perplexity: float = None
):
    """绘制2D t-SNE可视化图"""
    num_models = len(features_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(10 * num_models, 8))
    axes = [axes] if num_models == 1 else axes
    
    # 颜色映射 - 修复兼容性问题
    unique_labels = np.unique(labels)
    colors = get_colormap(args.colormap, len(unique_labels))
    
    # 确保alpha值在有效范围内
    alpha = max(0.0, min(1.0, args.alpha))
    
    # 检测异常值
    outlier_masks = {}
    if args.highlight_outliers:
        for model_name, feat_2d in features_dict.items():
            outlier_masks[model_name] = detect_outliers(feat_2d, args.outlier_threshold)
    
    # 绘制每个模型的特征分布
    for i, (model_name, feat_2d) in enumerate(features_dict.items()):
        ax = axes[i]
        
        # 根据绘图样式选择不同的可视化方法
        if args.plot_style == 'density':
            # 核密度估计
            from scipy.stats import gaussian_kde
            for label_idx, label in enumerate(unique_labels):
                mask = labels == label
                if np.sum(mask) > 1:
                    points = feat_2d[mask]
                    kde = gaussian_kde(points.T)
                    x_min, x_max = points[:, 0].min(), points[:, 0].max()
                    y_min, y_max = points[:, 1].min(), points[:, 1].max()
                    x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
                    z = np.reshape(kde(positions).T, x_grid.shape)
                    ax.contour(x_grid, y_grid, z, levels=5, colors=[colors[label_idx]], alpha=0.5)
                    ax.scatter(points[:, 0], points[:, 1], color=colors[label_idx], 
                              label=class_names[label] if label < len(class_names) else f"Class {label}",
                              alpha=alpha, s=args.point_size)
        
        elif args.plot_style == 'hexbin':
            # 六边形分箱图
            for label_idx, label in enumerate(unique_labels):
                mask = labels == label
                points = feat_2d[mask]
                hexbin = ax.hexbin(points[:, 0], points[:, 1], gridsize=20, 
                                  cmap=plt.cm.Blues if label_idx % 2 == 0 else plt.cm.Reds,
                                  alpha=0.6, edgecolors='none')
        
        elif args.plot_style == 'alpha_scatter':
            # 透明度渐变散点图
            for label_idx, label in enumerate(unique_labels):
                mask = labels == label
                points = feat_2d[mask]
                # 根据点密度调整透明度
                from sklearn.neighbors import KernelDensity
                if len(points) > 10:
                    kde = KernelDensity(bandwidth=0.5)
                    kde.fit(points)
                    densities = np.exp(kde.score_samples(points))
                    # 确保透明度在[0, 1]范围内
                    density_normalized = (densities - densities.min()) / (densities.max() - densities.min() + 1e-10)
                    alphas = 0.1 + 0.9 * density_normalized
                    # 确保所有alpha值都在[0, 1]范围内
                    alphas = np.clip(alphas, 0.0, 1.0)
                else:
                    alphas = np.full(len(points), alpha)
                
                ax.scatter(points[:, 0], points[:, 1], color=colors[label_idx],
                          label=class_names[label] if label < len(class_names) else f"Class {label}",
                          alpha=alphas, s=args.point_size, edgecolor='white', linewidth=0.5)
        
        else:  # 默认散点图
            for label_idx, label in enumerate(unique_labels):
                mask = labels == label
                points = feat_2d[mask]
                
                # 高亮异常值
                if args.highlight_outliers and model_name in outlier_masks:
                    outlier_mask_model = outlier_masks[model_name]
                    outlier_mask = outlier_mask_model[mask]
                    normal_mask = ~outlier_mask
                    
                    # 绘制正常点
                    if np.any(normal_mask):
                        ax.scatter(points[normal_mask, 0], points[normal_mask, 1], 
                                  color=colors[label_idx],
                                  label=class_names[label] if label < len(class_names) else f"Class {label}",
                                  alpha=alpha, s=args.point_size, 
                                  edgecolor='white', linewidth=0.5)
                    
                    # 高亮异常点
                    if np.any(outlier_mask):
                        ax.scatter(points[outlier_mask, 0], points[outlier_mask, 1],
                                  color='red', marker='X', s=args.point_size * 2,
                                  alpha=1.0, edgecolor='black', linewidth=1.5,
                                  label=f"{class_names[label]} (异常值)" if label < len(class_names) else f"Class {label} (异常值)")
                else:
                    ax.scatter(points[:, 0], points[:, 1], color=colors[label_idx],
                              label=class_names[label] if label < len(class_names) else f"Class {label}",
                              alpha=alpha, s=args.point_size, 
                              edgecolor='white', linewidth=0.5)
        
        # 计算并显示聚类质量指标
        if args.plot_cluster_quality:
            metrics = compute_cluster_metrics(feat_2d, labels)
            metrics_text = f"聚类质量指标:\n"
            if metrics['silhouette'] is not None:
                metrics_text += f"轮廓系数: {metrics['silhouette']:.3f}\n"
            if metrics['calinski_harabasz'] is not None:
                metrics_text += f"CH指数: {metrics['calinski_harabasz']:.0f}\n"
            if metrics['avg_intra_distance'] is not None:
                metrics_text += f"类内平均距离: {metrics['avg_intra_distance']:.3f}"
            
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 设置标题和标签
        title = f"{model_name}\n({args.feature_type.capitalize()} Features)"
        if perplexity is not None:
            title += f"\nPerplexity={perplexity}"
        ax.set_title(title, fontsize=14, pad=20)
        
        if args.show_legend and len(unique_labels) <= 20:  # 避免图例过于拥挤
            ax.legend(bbox_to_anchor=(1.05, 1), loc=args.legend_loc, fontsize=9)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
        if i == 0:
            ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    
    plt.tight_layout()
    return fig


def visualize_tsne_3d(
    args,
    features_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    class_names: List[str],
    output_path: str
):
    """绘制3D t-SNE可视化图"""
    num_models = len(features_dict)
    fig = plt.figure(figsize=(8 * num_models, 8))
    
    # 颜色映射 - 修复兼容性问题
    unique_labels = np.unique(labels)
    colors = get_colormap(args.colormap, len(unique_labels))
    
    for i, (model_name, feat_3d) in enumerate(features_dict.items()):
        ax = fig.add_subplot(1, num_models, i+1, projection='3d')
        
        for label_idx, label in enumerate(unique_labels):
            mask = labels == label
            points = feat_3d[mask]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      color=colors[label_idx],
                      label=class_names[label] if label < len(class_names) else f"Class {label}",
                      alpha=args.alpha, s=args.point_size,
                      edgecolor='white', linewidth=0.5)
        
        ax.set_title(f"{model_name}\n(3D {args.feature_type.capitalize()} Features)", fontsize=14, pad=20)
        ax.set_xlabel("t-SNE Dim 1", fontsize=12)
        ax.set_ylabel("t-SNE Dim 2", fontsize=12)
        ax.set_zlabel("t-SNE Dim 3", fontsize=12)
        
        if args.show_legend and len(unique_labels) <= 15:
            ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', fontsize=9)
    
    plt.tight_layout()
    return fig


def compare_perplexity(
    args,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_dir: str
):
    """比较不同困惑度参数的效果"""
    if args.perplexity_list is None:
        perplexities = [5, 15, 30, 50, 100]
    else:
        perplexities = [float(p) for p in args.perplexity_list.split(',')]
    
    n_perplexities = len(perplexities)
    n_cols = min(3, n_perplexities)
    n_rows = (n_perplexities + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    unique_labels = np.unique(labels)
    colors = get_colormap(args.colormap, len(unique_labels))
    
    for idx, perplexity in enumerate(perplexities):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # 执行t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=args.n_iter,
            random_state=args.random_state,
            metric=args.metric,
            init=args.init,
            learning_rate=args.learning_rate,
            early_exaggeration=args.early_exaggeration,
            n_iter_without_progress=args.n_iter_without_progress,
            min_grad_norm=args.min_grad_norm,
            method=args.tsne_method,
            angle=args.angle
        )
        
        feat_2d = tsne.fit_transform(features)
        
        # 绘制
        for label_idx, label in enumerate(unique_labels):
            mask = labels == label
            points = feat_2d[mask]
            ax.scatter(points[:, 0], points[:, 1], color=colors[label_idx],
                      label=class_names[label] if label < len(class_names) else f"Class {label}",
                      alpha=args.alpha, s=args.point_size, edgecolor='white', linewidth=0.5)
        
        # 计算并显示聚类质量
        metrics = compute_cluster_metrics(feat_2d, labels)
        metrics_text = f"Perplexity={perplexity}\n"
        if metrics['silhouette'] is not None:
            metrics_text += f"Silhouette: {metrics['silhouette']:.3f}\n"
        if metrics['calinski_harabasz'] is not None:
            metrics_text += f"CH: {metrics['calinski_harabasz']:.0f}"
        
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f"Perplexity = {perplexity}", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 只在第一个子图显示图例
        if idx == 0 and args.show_legend and len(unique_labels) <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(len(perplexities), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"perplexity_comparison_{args.feature_type}.png")
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"困惑度比较图已保存至: {output_path}")
    plt.show()


def compare_metrics(
    args,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_dir: str
):
    """比较不同距离度量方法的效果"""
    metrics_list = ['euclidean', 'cosine', 'manhattan', 'chebyshev']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    unique_labels = np.unique(labels)
    colors = get_colormap(args.colormap, len(unique_labels))
    
    for idx, metric in enumerate(metrics_list):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # 执行t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            n_iter=args.n_iter,
            random_state=args.random_state,
            metric=metric,
            init=args.init,
            learning_rate=args.learning_rate,
            early_exaggeration=args.early_exaggeration,
            n_iter_without_progress=args.n_iter_without_progress,
            min_grad_norm=args.min_grad_norm,
            method=args.tsne_method,
            angle=args.angle
        )
        
        feat_2d = tsne.fit_transform(features)
        
        # 绘制
        for label_idx, label in enumerate(unique_labels):
            mask = labels == label
            points = feat_2d[mask]
            ax.scatter(points[:, 0], points[:, 1], color=colors[label_idx],
                      label=class_names[label] if label < len(class_names) else f"Class {label}",
                      alpha=args.alpha, s=args.point_size, edgecolor='white', linewidth=0.5)
        
        # 计算并显示聚类质量
        metrics = compute_cluster_metrics(feat_2d, labels)
        metrics_text = f"Metric={metric}\n"
        if metrics['silhouette'] is not None:
            metrics_text += f"Silhouette: {metrics['silhouette']:.3f}\n"
        if metrics['calinski_harabasz'] is not None:
            metrics_text += f"CH: {metrics['calinski_harabasz']:.0f}"
        
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f"Distance Metric: {metric}", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"metric_comparison_{args.feature_type}.png")
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"距离度量比较图已保存至: {output_path}")
    plt.show()


def main():
    # 解析命令行参数
    args = parse_args()
    
    print("=" * 60)
    print(f"t-SNE可视化参数:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # 初始化设备
    device = init_distributed_device(args)
    print(f"使用计算设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 从配置文件的指定字段获取类别名称
    class_dict = config.get(args.class_names, {})
    if not class_dict:
        raise ValueError(f"配置文件中未找到指定的类别字段: {args.class_names}")
    
    # 将字典按键转换为整数排序，然后取值
    sorted_keys = sorted(map(int, class_dict.keys()))
    class_names = [class_dict[str(key)] for key in sorted_keys]
    print(f"类别数量: {len(class_names)}")
    
    # 加载模型
    print("\n加载主模型...")
    main_model, preprocess_val = load_model(
        args, 
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=device
    )
    
    # 加载对比模型（如果提供）
    base_model = None
    if args.base_model and args.base_checkpoint:
        print("\n加载对比模型...")
        base_model, _ = load_model(
            args,
            model_name=args.base_model,
            checkpoint_path=args.base_checkpoint,
            device=device
        )
    
    # 获取tokenizer
    tokenizer = get_tokenizer(args.model)
    
    # 准备数据
    print("\n准备数据集...")
    val_dataloader = prepare_data(args, preprocess_val, tokenizer)
    print(f"验证集样本数量: {val_dataloader.num_samples}")
    
    # 提取特征
    print(f"\n提取{args.feature_type}特征...")
    features_dict = {}
    labels = None
    
    # 提取主模型特征
    main_feat, main_labels = extract_features(
        main_model, val_dataloader, args, device, args.feature_type
    )
    labels = main_labels
    main_feat = preprocess_features(main_feat, args)
    features_dict[args.model_name0] = main_feat
    
    # 提取对比模型特征（如果存在）
    if base_model is not None:
        base_feat, _ = extract_features(
            base_model, val_dataloader, args, device, args.feature_type
        )
        base_feat = preprocess_features(base_feat, args)
        features_dict[args.model_name1] = base_feat
    
    # 保存原始特征
    if args.save_embeddings:
        for model_name, feat in features_dict.items():
            embed_path = os.path.join(args.output_dir, f"{model_name}_features.npy")
            np.save(embed_path, feat)
            print(f"保存特征到: {embed_path}")
    
    # 困惑度比较
    if args.perplexity_list is not None:
        print("\n执行困惑度参数比较...")
        compare_perplexity(args, main_feat, labels, class_names, args.output_dir)
    
    # 距离度量比较
    if args.compare_metrics:
        print("\n执行距离度量方法比较...")
        compare_metrics(args, main_feat, labels, class_names, args.output_dir)
    
    # t-SNE降维
    print(f"\n执行t-SNE降维 (n_components={args.n_components})...")
    
    # 配置t-SNE参数
    tsne_kwargs = {
        'n_components': args.n_components,
        'perplexity': args.perplexity,
        'max_iter': args.max_iter,
        'random_state': args.random_state,
        'metric': args.metric,
        'init': args.init,
        'learning_rate': args.learning_rate,
        'early_exaggeration': args.early_exaggeration,
        'n_iter_without_progress': args.n_iter_without_progress,
        'min_grad_norm': args.min_grad_norm,
        'method': args.tsne_method,
        'angle': args.angle,
        'verbose': 1,
        'n_jobs': -1
    }
    
    # 对每个模型的特征进行降维
    tsne_results = {}
    for model_name, feat in features_dict.items():
        print(f"对{model_name}特征进行t-SNE降维...")
        tsne = TSNE(**tsne_kwargs)
        feat_reduced = tsne.fit_transform(feat)
        tsne_results[model_name] = feat_reduced
        
        # 保存降维后的嵌入
        if args.save_embeddings:
            embed_path = os.path.join(args.output_dir, f"{model_name}_tsne_embeddings.npy")
            np.save(embed_path, feat_reduced)
            print(f"保存t-SNE嵌入到: {embed_path}")
    
    # 可视化
    print("\n生成t-SNE可视化图...")
    
    # 生成输出文件名
    output_filename = f"tsne_{args.feature_type}_{args.class_names}"
    if args.n_components == 3:
        output_filename += "_3d"
    if args.perplexity_list is not None:
        output_filename += f"_perplexity{args.perplexity}"
    if args.compare_metrics:
        output_filename += f"_metric{args.metric}"
    output_filename += f"_{args.model_name0}_{args.model_name1}_{current_time}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # 根据维度选择可视化函数
    if args.n_components == 2:
        fig = visualize_tsne_2d(
            args,
            features_dict=tsne_results,
            labels=labels,
            class_names=class_names,
            output_path=output_path,
            perplexity=args.perplexity
        )
    else:  # 3D
        fig = visualize_tsne_3d(
            args,
            features_dict=tsne_results,
            labels=labels,
            class_names=class_names,
            output_path=output_path
        )
    
    # 保存图像
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"t-SNE可视化结果已保存至: {output_path}")
    plt.show()
    
    # 生成参数报告
    report_path = os.path.join(args.output_dir, f"tsne_report_{current_time}.txt")
    with open(report_path, 'w') as f:
        f.write("t-SNE Visualization Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Feature type: {args.feature_type}\n")
        f.write(f"Class names field: {args.class_names}\n")
        f.write(f"Number of classes: {len(class_names)}\n")
        f.write(f"Number of samples: {len(labels)}\n")
        f.write("\nModel Information:\n")
        for model_name in features_dict.keys():
            f.write(f"  - {model_name}\n")
        f.write("\nt-SNE Parameters:\n")
        for key, value in tsne_kwargs.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nVisualization Parameters:\n")
        f.write(f"  Plot style: {args.plot_style}\n")
        f.write(f"  Point size: {args.point_size}\n")
        f.write(f"  Alpha: {args.alpha}\n")
        f.write(f"  Colormap: {args.colormap}\n")
    
    print(f"参数报告已保存至: {report_path}")
    print("\n所有操作完成!")


if __name__ == "__main__":
    main()