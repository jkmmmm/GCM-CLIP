"""
t-SNE可视化绘图测试脚本
测试不同参数配置下的绘图效果
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
import argparse
from typing import Dict, List
import matplotlib

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入绘图函数
from tSNEanalysis import (
    visualize_tsne_2d,
    visualize_tsne_3d,
    compare_perplexity,
    compare_metrics,
    compute_cluster_metrics,
    detect_outliers,
    get_colormap,
    preprocess_features
)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_test_data(
    n_samples_per_class: int = 100,
    n_classes: int = 10,
    n_features: int = 512,
    random_state: int = 42
) -> tuple:
    """生成测试数据"""
    np.random.seed(random_state)
    
    # 生成聚类明显的特征
    all_features = []
    all_labels = []
    
    for class_idx in range(n_classes):
        # 每个类别有一个中心点
        center = np.random.randn(n_features) * 5
        # 添加类别特定的偏移
        center += class_idx * 3
        
        # 生成围绕中心点的特征
        class_features = np.random.randn(n_samples_per_class, n_features) * 0.5 + center
        class_labels = np.full(n_samples_per_class, class_idx)
        
        all_features.append(class_features)
        all_labels.append(class_labels)
    
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    
    # 添加一些噪声
    features += np.random.randn(*features.shape) * 0.1
    
    return features, labels


def test_2d_visualization():
    """测试2D可视化"""
    print("=" * 60)
    print("测试2D t-SNE可视化")
    print("=" * 60)
    
    # 生成测试数据
    features, labels = generate_test_data(
        n_samples_per_class=50,
        n_classes=8,
        n_features=256
    )
    
    # 模拟t-SNE降维结果
    np.random.seed(42)
    tsne_results = {}
    
    # 模型1的特征（聚类明显）
    for model_name in ["CLIP模型", "CM-CLIP模型"]:
        feat_2d = np.zeros((len(labels), 2))
        
        # 为每个类别生成聚类
        for class_idx in np.unique(labels):
            mask = labels == class_idx
            n_samples = np.sum(mask)
            
            # 设置类别中心
            if model_name == "CLIP模型":
                center = np.array([class_idx * 3, class_idx * 2])
                spread = 1.0
            else:
                # CM-CLIP应该有更好的聚类
                center = np.array([class_idx * 3, class_idx * 2])
                spread = 0.5  # 更紧密的聚类
            
            # 生成围绕中心的点
            feat_2d[mask] = np.random.randn(n_samples, 2) * spread + center
        
        tsne_results[model_name] = feat_2d
    
    # 类别名称
    disease_categories = [
        "急性淋巴细胞白血病",
        "急性髓系白血病", 
        "多发性骨髓瘤",
        "淋巴瘤",
        "骨髓增生异常综合征",
        "骨髓增殖性肿瘤",
        "再生障碍性贫血",
        "溶血性贫血"
    ]
    
    class_names = disease_categories[:len(np.unique(labels))]
    
    # 创建模拟参数
    class MockArgs:
        def __init__(self):
            self.plot_style = 'scatter'
            self.point_size = 30
            self.alpha = 0.7
            self.colormap = 'tab20'
            self.show_legend = True
            self.legend_loc = 'upper right'
            self.feature_type = 'image'
            self.plot_cluster_quality = True
            self.highlight_outliers = True
            self.outlier_threshold = 3.0
            self.class_names = 'category'
            self.n_components = 2
    
    args = MockArgs()
    
    # 创建输出目录
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试不同的绘图样式
    plot_styles = ['scatter', 'alpha_scatter']  # 'density'和'hexbin'需要更多计算
    
    for plot_style in plot_styles:
        print(f"\n测试绘图样式: {plot_style}")
        args.plot_style = plot_style
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"test_2d_{plot_style}_{timestamp}.png")
        
        # 调用绘图函数
        fig = visualize_tsne_2d(
            args,
            features_dict=tsne_results,
            labels=labels,
            class_names=class_names,
            output_path=output_path,
            perplexity=30.0
        )
        
        # 保存图像
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图像已保存: {output_path}")
        plt.show()


def test_3d_visualization():
    """测试3D可视化"""
    print("\n" + "=" * 60)
    print("测试3D t-SNE可视化")
    print("=" * 60)
    
    # 生成测试数据
    features, labels = generate_test_data(
        n_samples_per_class=30,
        n_classes=6,
        n_features=256
    )
    
    # 模拟3D t-SNE降维结果
    np.random.seed(42)
    tsne_results = {}
    
    # 模型1的特征（聚类明显）
    for model_name in ["CLIP模型", "CM-CLIP模型"]:
        feat_3d = np.zeros((len(labels), 3))
        
        # 为每个类别生成聚类
        for class_idx in np.unique(labels):
            mask = labels == class_idx
            n_samples = np.sum(mask)
            
            # 设置类别中心
            if model_name == "CLIP模型":
                center = np.array([class_idx * 4, class_idx * 3, class_idx * 2])
                spread = 1.2
            else:
                # CM-CLIP应该有更好的聚类
                center = np.array([class_idx * 4, class_idx * 3, class_idx * 2])
                spread = 0.8  # 更紧密的聚类
            
            # 生成围绕中心的点
            feat_3d[mask] = np.random.randn(n_samples, 3) * spread + center
        
        tsne_results[model_name] = feat_3d
    
    # 类别名称（疾病位置）
    disease_locations = [
        "骨髓",
        "淋巴结", 
        "脾脏",
        "肝脏",
        "外周血",
        "中枢神经系统"
    ]
    
    class_names = disease_locations[:len(np.unique(labels))]
    
    # 创建模拟参数
    class MockArgs:
        def __init__(self):
            self.plot_style = 'scatter'
            self.point_size = 40
            self.alpha = 0.7
            self.colormap = 'tab20c'
            self.show_legend = True
            self.legend_loc = 'upper left'
            self.feature_type = 'text'  # 测试文本特征
            self.n_components = 3
    
    args = MockArgs()
    
    # 创建输出目录
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"test_3d_{timestamp}.png")
    
    # 调用绘图函数
    fig = visualize_tsne_3d(
        args,
        features_dict=tsne_results,
        labels=labels,
        class_names=class_names,
        output_path=output_path
    )
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"3D图像已保存: {output_path}")
    plt.show()


def test_perplexity_comparison():
    """测试困惑度参数比较"""
    print("\n" + "=" * 60)
    print("测试困惑度参数比较")
    print("=" * 60)
    
    # 生成测试数据
    features, labels = generate_test_data(
        n_samples_per_class=40,
        n_classes=5,
        n_features=128
    )
    
    # 类别名称
    class_names = ["类别A", "类别B", "类别C", "类别D", "类别E"]
    
    # 创建模拟参数
    class MockArgs:
        def __init__(self):
            self.perplexity_list = "5,15,30,50,100"
            self.n_iter = 500
            self.random_state = 42
            self.metric = 'euclidean'
            self.init = 'random'
            self.learning_rate = 200.0
            self.early_exaggeration = 12.0
            self.n_iter_without_progress = 300
            self.min_grad_norm = 1e-7
            self.tsne_method = 'barnes_hut'
            self.angle = 0.5
            self.point_size = 25
            self.alpha = 0.6
            self.colormap = 'Set2'
            self.show_legend = True
            self.feature_type = 'image'
    
    args = MockArgs()
    
    # 创建输出目录
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 调用困惑度比较函数
    compare_perplexity(
        args,
        features=features,
        labels=labels,
        class_names=class_names,
        output_dir=output_dir
    )


def test_metrics_comparison():
    """测试距离度量比较"""
    print("\n" + "=" * 60)
    print("测试距离度量方法比较")
    print("=" * 60)
    
    # 生成测试数据
    features, labels = generate_test_data(
        n_samples_per_class=30,
        n_classes=4,
        n_features=64
    )
    
    # 对特征进行预处理（归一化）
    features = preprocess_features(features, MockArgs())
    
    # 类别名称
    class_names = ["类型I", "类型II", "类型III", "类型IV"]
    
    # 创建模拟参数
    class MockArgs:
        def __init__(self):
            self.compare_metrics = True
            self.perplexity = 20.0
            self.n_iter = 300
            self.random_state = 42
            self.metric = 'cosine'
            self.init = 'pca'
            self.learning_rate = 200.0
            self.early_exaggeration = 12.0
            self.n_iter_without_progress = 300
            self.min_grad_norm = 1e-7
            self.tsne_method = 'barnes_hut'
            self.angle = 0.5
            self.point_size = 30
            self.alpha = 0.7
            self.colormap = 'tab10'
            self.show_legend = True
            self.feature_type = 'image'
            self.normalize_features = True
            self.standardize = False
    
    args = MockArgs()
    
    # 创建输出目录
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 调用距离度量比较函数
    compare_metrics(
        args,
        features=features,
        labels=labels,
        class_names=class_names,
        output_dir=output_dir
    )


def test_cluster_metrics():
    """测试聚类质量指标计算"""
    print("\n" + "=" * 60)
    print("测试聚类质量指标计算")
    print("=" * 60)
    
    # 生成不同质量的聚类数据
    np.random.seed(42)
    
    # 良好聚类
    good_clusters = []
    good_labels = []
    for i in range(3):
        center = np.array([i*5, i*5])
        points = np.random.randn(50, 2) * 0.5 + center
        good_clusters.append(points)
        good_labels.extend([i] * 50)
    good_features = np.vstack(good_clusters)
    good_labels = np.array(good_labels)
    
    # 较差聚类（重叠）
    poor_clusters = []
    poor_labels = []
    for i in range(3):
        center = np.array([i*2, i*2])  # 中心更接近
        points = np.random.randn(50, 2) * 1.5 + center  # 更大的方差
        poor_clusters.append(points)
        poor_labels.extend([i] * 50)
    poor_features = np.vstack(poor_clusters)
    poor_labels = np.array(poor_labels)
    
    # 计算聚类质量指标
    print("良好聚类的指标:")
    good_metrics = compute_cluster_metrics(good_features, good_labels)
    for key, value in good_metrics.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")
    
    print("\n较差聚类的指标:")
    poor_metrics = compute_cluster_metrics(poor_features, poor_labels)
    for key, value in poor_metrics.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制良好聚类
    axes[0].scatter(good_features[:, 0], good_features[:, 1], 
                   c=good_labels, cmap='tab10', s=20, alpha=0.7)
    axes[0].set_title(f"良好聚类\n轮廓系数: {good_metrics['silhouette']:.3f}", fontsize=12)
    
    # 绘制较差聚类
    axes[1].scatter(poor_features[:, 0], poor_features[:, 1], 
                   c=poor_labels, cmap='tab10', s=20, alpha=0.7)
    axes[1].set_title(f"较差聚类\n轮廓系数: {poor_metrics['silhouette']:.3f}", fontsize=12)
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cluster_metrics_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n聚类质量对比图已保存: {output_path}")
    plt.show()


def test_outlier_detection():
    """测试异常值检测"""
    print("\n" + "=" * 60)
    print("测试异常值检测")
    print("=" * 60)
    
    # 生成包含异常值的数据
    np.random.seed(42)
    
    # 正常数据
    normal_data = np.random.randn(200, 2) * 1.0
    labels = np.zeros(200)
    
    # 添加异常值
    outliers = np.array([
        [8, 8],    # 远离中心的点
        [-6, -6],  # 另一个方向的异常点
        [10, -5],  # 对角线上的异常点
        [-8, 7],   # 混合异常点
        [0, 12]    # 垂直方向异常点
    ])
    
    # 合并数据
    all_data = np.vstack([normal_data, outliers])
    labels = np.concatenate([labels, np.ones(5)])  # 异常值标签为1
    
    # 检测异常值
    outlier_mask = detect_outliers(all_data, threshold=3.0)
    print(f"检测到异常值数量: {np.sum(outlier_mask)}")
    print(f"实际异常值数量: {len(outliers)}")
    
    # 可视化
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制正常点
    normal_mask = ~outlier_mask
    ax.scatter(all_data[normal_mask, 0], all_data[normal_mask, 1],
              color='blue', alpha=0.6, s=30, label='正常点')
    
    # 绘制检测到的异常值
    ax.scatter(all_data[outlier_mask, 0], all_data[outlier_mask, 1],
              color='red', marker='X', s=100, label='检测到的异常值')
    
    # 绘制实际异常值位置（用黑色圆圈标记）
    ax.scatter(outliers[:, 0], outliers[:, 1],
              facecolors='none', edgecolors='black', 
              s=150, linewidth=2, label='实际异常值位置')
    
    ax.set_title("异常值检测示例", fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "outlier_detection_test.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"异常值检测图已保存: {output_path}")
    plt.show()


def test_colormap():
    """测试颜色映射"""
    print("\n" + "=" * 60)
    print("测试颜色映射")
    print("=" * 60)
    
    # 测试不同的颜色映射
    colormaps = ['tab20', 'tab20c', 'Set2', 'Set3', 'Paired', 'viridis', 'plasma']
    n_colors = 10
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, cmap_name in enumerate(colormaps):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        try:
            colors = get_colormap(cmap_name, n_colors)
            
            # 显示颜色条
            for i, color in enumerate(colors):
                ax.barh(i, 1, color=color, edgecolor='black')
            
            ax.set_title(cmap_name, fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, n_colors - 0.5)
            ax.set_yticks(range(n_colors))
            ax.set_yticklabels([f"类别{i}" for i in range(n_colors)])
            ax.grid(False)
            ax.set_xticks([])
            
        except Exception as e:
            ax.text(0.5, 0.5, f"错误:\n{e}", 
                   ha='center', va='center', fontsize=10)
            ax.set_title(f"{cmap_name} (失败)")
    
    # 隐藏多余的子图
    for idx in range(len(colormaps), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("不同颜色映射方案测试", fontsize=16)
    plt.tight_layout()
    
    # 保存图像
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "colormap_test.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"颜色映射测试图已保存: {output_path}")
    plt.show()


def test_all():
    """运行所有测试"""
    print("开始运行t-SNE可视化测试套件")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行测试
    test_functions = [
        test_2d_visualization,
        test_3d_visualization,
        test_perplexity_comparison,
        test_metrics_comparison,
        test_cluster_metrics,
        test_outlier_detection,
        test_colormap
    ]
    
    for i, test_func in enumerate(test_functions, 1):
        print(f"\n[{i}/{len(test_functions)}] 运行测试: {test_func.__name__}")
        try:
            test_func()
            print(f"✓ 测试通过: {test_func.__name__}")
        except Exception as e:
            print(f"✗ 测试失败: {test_func.__name__}")
            print(f"  错误信息: {str(e)}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print(f"测试结果保存在: {os.path.abspath(output_dir)}")


def interactive_test():
    """交互式测试菜单"""
    print("t-SNE可视化测试菜单")
    print("=" * 40)
    print("1. 测试2D可视化")
    print("2. 测试3D可视化")
    print("3. 测试困惑度比较")
    print("4. 测试距离度量比较")
    print("5. 测试聚类质量指标")
    print("6. 测试异常值检测")
    print("7. 测试颜色映射")
    print("8. 运行所有测试")
    print("0. 退出")
    print("=" * 40)
    
    while True:
        try:
            choice = input("请选择测试项目 (0-8): ").strip()
            
            if choice == '0':
                print("退出测试程序")
                break
            elif choice == '1':
                test_2d_visualization()
            elif choice == '2':
                test_3d_visualization()
            elif choice == '3':
                test_perplexity_comparison()
            elif choice == '4':
                test_metrics_comparison()
            elif choice == '5':
                test_cluster_metrics()
            elif choice == '6':
                test_outlier_detection()
            elif choice == '7':
                test_colormap()
            elif choice == '8':
                test_all()
                break
            else:
                print("无效选择，请重新输入")
        
        except KeyboardInterrupt:
            print("\n用户中断")
            break
        except Exception as e:
            print(f"测试出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='t-SNE可视化测试工具')
    parser.add_argument('--test-all', action='store_true', help='运行所有测试')
    parser.add_argument('--interactive', action='store_true', help='交互式测试菜单')
    parser.add_argument('--output-dir', type=str, default='./test_results', help='测试结果输出目录')
    
    args = parser.parse_args()
    
    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.interactive:
        interactive_test()
    elif args.test_all:
        test_all()
    else:
        # 默认运行核心测试
        print("运行核心测试...")
        test_2d_visualization()
        test_3d_visualization()
        test_perplexity_comparison()
        print("\n测试完成！")