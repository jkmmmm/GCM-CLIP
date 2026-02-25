import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import torch
import os
import json
import sys
from datetime import datetime
from typing import Optional, Tuple
import argparse
from argparse import Namespace
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# 导入项目核心模块
from open_clip import create_model_and_transforms, get_tokenizer
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
    parser.add_argument('--model-name0', type=str, required=True, help='在CLIP模型上的修改')
    parser.add_argument('--model-name1', type=str, required=True, help='在CLIP模型上的修改')
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
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity参数')
    parser.add_argument('--n-iter', type=int, default=2000, help='t-SNE迭代次数')
    parser.add_argument('--random-state', type=int, default=42, help='随机种子')
    parser.add_argument('--feature-type', type=str, default='image', choices=['image', 'text'], 
                      help='特征类型：image或text')
    
    # 输出配置
    parser.add_argument('--output-dir', type=str, default='./tsne_results', help='可视化结果保存目录')
    parser.add_argument('--class-names', type=str, default='category', 
                      help='配置文件中的类别名称字段，如"category"或"location"')
    
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
    
def load_model(args, model_name: str, checkpoint_path: str, device: torch.device):
    """加载模型（兼容原始CLIP和CM-CLIP）"""
    # 创建模型（复用主函数的模型初始化逻辑）
    model_kwargs = {}
    if 'siglip' in model_name.lower():
        model_kwargs['init_logit_scale'] = np.log(10)
        model_kwargs['init_logit_bias'] = -10
        
    model, _, preprocess_val = create_model_and_transforms(
        model_name,
        pretrained=checkpoint_path,  # 直接加载预训练权重
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
    """准备数据集（复用主函数的数据加载逻辑）"""
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    args.train_data = None
    
    # 获取数据加载器
    data = get_data(
        args,
        (preprocess_val, preprocess_val),  # 验证集使用相同的预处理
        config=config,
        epoch=0,
        tokenizer=tokenizer,
    )
    
    if 'val' not in data:
        raise ValueError("数据加载失败，未找到验证集")
    
    return data['val'].dataloader


# def extract_features(
#     model: torch.nn.Module,
#     data_loader: torch.utils.data.DataLoader,
#     args,
#     device: torch.device,
#     feature_type: str = "image"
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """提取特征和标签"""
#     all_features = []
#     all_labels = []
    
#     input_dtype = get_input_dtype(args.precision)
#     autocast = get_autocast(args.precision, device_type=device.type)
    
#     model.eval()
#     with torch.inference_mode():
#         for batch in data_loader:
#             images, texts, category, location = batch
#             images = images.to(device=device, dtype=input_dtype, non_blocking=True)
#             texts = texts.to(device=device, non_blocking=True)
            
#             with autocast():
#                 if feature_type == "image":
#                     model_out = model(image=images)
#                     batch_features = model_out["image_features"].cpu()
#                 else:
#                     model_out = model(text=texts)
#                     batch_features = model_out["text_features"].cpu()
            
#             # 根据参数选择使用category还是location作为标签
#             if args.class_names == 'category':
#                 all_labels.append(category.cpu())
#             else:  # location
#                 all_labels.append(location.cpu())
                
#             all_features.append(batch_features)
    
#     features = torch.cat(all_features, dim=0).numpy()
#     labels = torch.cat(all_labels, dim=0).numpy()
    
#     return features, labels
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
            
            # 根据参数选择使用category还是location作为标签
            if args.class_names == 'category':
                all_labels.append(category.cpu())
            else:  # location
                all_labels.append(location.cpu())
                
            all_features.append(batch_features)
    
    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    # 将 one-hot 编码转换为类别索引
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    
    return features, labels

def visualize_tsne(
    args,
    features_dict: dict,
    labels: np.ndarray,
    class_names: list,
    output_path: str
):
    """绘制t-SNE可视化图"""
    # 创建画布
    num_models = len(features_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(10 * num_models, 8))
    axes = [axes] if num_models == 1 else axes
    
    # 颜色映射
    unique_labels = np.unique(labels)
    colors = sns.color_palette("hsv", len(unique_labels))
    
    # 绘制每个模型的特征分布
    for i, (model_name, feat_2d) in enumerate(features_dict.items()):
        ax = axes[i]
        for label_idx, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                feat_2d[mask, 0],
                feat_2d[mask, 1],
                color=colors[label_idx],
                label=class_names[label] if label < len(class_names) else f"Class {label}",
                alpha=0.7,
                s=30,
                edgecolor="white",
                linewidth=0.5
            )
        
        # 设置标题和标签
        ax.set_title(
            f"{model_name}\n({args.feature_type.capitalize()} Features)",
            fontsize=14,
            pad=20
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
        if i == 0:
            ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"t-SNE可视化结果已保存至: {output_path}")
    plt.show()


def main():
    # 解析命令行参数
    args = parse_args()
    print("=" * 60)
    print(f"t-SNE可视化参数: {args}")
    print("=" * 60)
    
    # 初始化设备
    device = init_distributed_device(args)
    print(f"使用计算设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir, 
        f"tsne_{args.feature_type}_{args.class_names}_{args.model_name0}_{args.model_name1}.png"
    )
    
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
    features_dict[args.model_name0] = main_feat
    
    # 提取对比模型特征（如果存在）
    if base_model is not None:
        base_feat, _ = extract_features(
            base_model, val_dataloader, args, device, args.feature_type
        )
        features_dict[args.model_name1] = base_feat
    
    # t-SNE降维
    print("\n执行t-SNE降维...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        max_iter=args.n_iter,
        random_state=args.random_state,
        verbose=1,
        n_jobs=-1
    )
    
    # 对每个模型的特征进行降维
    for model_name in features_dict:
        print(f"对{model_name}特征进行t-SNE降维...")
        features_dict[model_name] = tsne.fit_transform(features_dict[model_name])
    
    # 可视化
    print("\n生成t-SNE可视化图...")
    visualize_tsne(
        args,
        features_dict=features_dict,
        labels=labels,
        class_names=class_names,
        output_path=output_path
    )
    
    print("\n所有操作完成!")


if __name__ == "__main__":
    main()