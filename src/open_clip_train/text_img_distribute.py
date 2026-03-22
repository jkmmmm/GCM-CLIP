import torch
import torch.nn.functional as F
import sys
sys.path.append("/root/MedCLIP-SAMv2-main/biomedclip_finetuning/open_clip/src")
import open_clip
import numpy as np
import random
import os
import json
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import time  # 添加time模块用于重试机制

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryEfficientMedicalSimilarityCalculator:
    def __init__(self, model, preprocess, tokenizer, device, config=None):
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.model = self.model.to(self.device)
        self.model.eval()
        self.h5_path = None  # 初始化h5_path属性
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, model_name, config=None):
        """从检查点创建相似度计算器"""
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=None,
            force_custom_text=True
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logging.info(f'Loading pretrained {model_name} weights ({checkpoint_path}).')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=False)
        tokenizer = open_clip.get_tokenizer(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        return cls(model, preprocess, tokenizer, device, config)
    
    def extract_features_to_disk(self, dataloader, h5_path, max_samples=None):
        """将特征提取到磁盘文件，避免内存溢出"""
        self.model.eval()
        
        # 先设置h5_path属性
        self.h5_path = h5_path
        
        # 创建HDF5文件
        with h5py.File(h5_path, 'w') as h5_file:  # 使用h5_path而不是self.h5_path
            # 预分配数据集
            image_features_dset = h5_file.create_dataset(
                'image_features', (0, 512), maxshape=(None, 512), dtype='float32'
            )
            text_features_dset = h5_file.create_dataset(
                'text_features', (0, 512), maxshape=(None, 512), dtype='float32'
            )
            health_labels_dset = h5_file.create_dataset(
                'health_labels', (0,), maxshape=(None,), dtype='int32'
            )
            location_labels_dset = h5_file.create_dataset(
                'location_labels', (0,), maxshape=(None,), dtype='int32'
            )
            
            # 存储元数据
            categories = []
            locations = []
            captions = []
            image_paths = []
            
            total_samples = 0
            batch_count = 0
            
            print("开始提取特征到磁盘...")
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader)):
                    if max_samples and total_samples >= max_samples:
                        break
                        
                    if len(batch) >= 4:
                        images = batch[0]
                        texts = batch[1]
                        health_labels = batch[2]
                        location_labels = batch[3]
                        
                        # 获取元数据
                        if len(batch) > 4:
                            batch_categories = batch[4] if hasattr(batch[4], '__len__') else [batch[4]] * len(images)
                            batch_locations = batch[5] if hasattr(batch[5], '__len__') else [batch[5]] * len(images)
                            batch_captions = batch[6] if hasattr(batch[6], '__len__') else [batch[6]] * len(images)
                            batch_image_paths = batch[7] if len(batch) > 7 else [f"batch_{batch_idx}_idx_{i}" for i in range(len(images))]
                        else:
                            batch_categories = [f"category_batch_{batch_idx}"] * len(images)
                            batch_locations = [f"location_batch_{batch_idx}"] * len(images)
                            batch_captions = [f"caption_batch_{batch_idx}"] * len(images)
                            batch_image_paths = [f"batch_{batch_idx}_idx_{i}" for i in range(len(images))]
                    else:
                        continue
                    
                    images = images.to(self.device)
                    texts = texts.to(self.device)
                    
                    # 提取特征
                    image_features = self.model.encode_image(images)
                    text_features = self.model.encode_text(texts)
                    
                    # 处理不同形状的特征
                    if len(image_features.shape) == 3:
                        image_features = image_features.mean(dim=1)
                    
                    if len(text_features.shape) == 3:
                        text_features = text_features.mean(dim=1)
                        print(f"应用平均池化后文本特征形状: {text_features.shape}")
                    
                    # 归一化特征
                    image_features = F.normalize(image_features, p=2, dim=1)
                    text_features = F.normalize(text_features, p=2, dim=1)
                    
                    # 确保特征维度正确
                    if image_features.shape[1] != 512:
                        print(f"警告: 图像特征维度不是512，而是{image_features.shape[1]}，进行投影")
                        if not hasattr(self, 'image_proj'):
                            self.image_proj = torch.nn.Linear(image_features.shape[1], 512).to(self.device)
                        image_features = self.image_proj(image_features)
                        image_features = F.normalize(image_features, p=2, dim=1)
                    
                    if text_features.shape[1] != 512:
                        print(f"警告: 文本特征维度不是512，而是{text_features.shape[1]}，进行投影")
                        if not hasattr(self, 'text_proj'):
                            self.text_proj = torch.nn.Linear(text_features.shape[1], 512).to(self.device)
                        text_features = self.text_proj(text_features)
                        text_features = F.normalize(text_features, p=2, dim=1)
                    
                    # 转换标签
                    health_indices = torch.argmax(health_labels, dim=1).cpu().numpy()
                    location_indices = torch.argmax(location_labels, dim=1).cpu().numpy()
                    
                    batch_size = len(images)
                    
                    # 扩展数据集并写入
                    new_size = total_samples + batch_size
                    
                    image_features_dset.resize((new_size, 512))
                    text_features_dset.resize((new_size, 512))
                    health_labels_dset.resize((new_size,))
                    location_labels_dset.resize((new_size,))
                    
                    # 确保特征形状正确
                    image_features_np = image_features.cpu().numpy().astype('float32')
                    text_features_np = text_features.cpu().numpy().astype('float32')
                    
                    image_features_dset[total_samples:new_size] = image_features_np
                    text_features_dset[total_samples:new_size] = text_features_np
                    health_labels_dset[total_samples:new_size] = health_indices.astype('int32')
                    location_labels_dset[total_samples:new_size] = location_indices.astype('int32')
                    
                    # 存储元数据
                    categories.extend(batch_categories)
                    locations.extend(batch_locations)
                    captions.extend(batch_captions)
                    image_paths.extend(batch_image_paths)
                    
                    total_samples = new_size
                    batch_count += 1
                    
                    # 定期清理GPU缓存
                    if batch_count % 100 == 0:
                        torch.cuda.empty_cache()
                    
                    if batch_count % 500 == 0:
                        print(f"已处理 {batch_count} 个批次，{total_samples} 个样本")
            
            # 修改这里：使用可变长度字符串存储，支持Unicode
            # 方法1：使用可变长度字符串
            h5_file.create_dataset('categories', data=categories, dtype=h5py.special_dtype(vlen=str))
            h5_file.create_dataset('locations', data=locations, dtype=h5py.special_dtype(vlen=str))
            h5_file.create_dataset('captions', data=captions, dtype=h5py.special_dtype(vlen=str))
            h5_file.create_dataset('image_paths', data=image_paths, dtype=h5py.special_dtype(vlen=str))
            
        print(f"特征提取完成，共 {total_samples} 个样本，保存到: {h5_path}")
        self.total_samples = total_samples
        
        # 强制关闭HDF5文件句柄
        import gc
        gc.collect()
        time.sleep(1)  # 等待文件完全关闭
        
        return self
    
    def calculate_similarity_from_disk(self, condition_config, sample_fraction=0.1):
        """从磁盘文件计算相似度，使用采样避免内存问题"""
        if not hasattr(self, 'h5_path') or self.h5_path is None:
            raise ValueError("请先调用 extract_features_to_disk 方法")
        
        modality = condition_config.get('modality', 'image-text')
        health_condition = condition_config.get('health_condition', 'any')
        location_condition = condition_config.get('location_condition', 'any')
        sample_size = condition_config.get('sample_size', 1000)
        
        print(f"\n计算条件: {condition_config}")
        
        # 从磁盘加载数据
        with h5py.File(self.h5_path, 'r') as h5_file:
            n_samples = self.total_samples
            
            # 使用采样来减少内存使用
            if sample_fraction < 1.0:
                sample_indices = np.random.choice(
                    n_samples, 
                    size=int(n_samples * sample_fraction), 
                    replace=False
                )
                print(f"使用 {sample_fraction*100}% 的样本进行采样: {len(sample_indices)} 个样本")
            else:
                sample_indices = np.arange(n_samples)
            
            # 生成符合条件的样本对
            pairs = self._generate_sample_pairs_from_indices(
                h5_file, sample_indices, health_condition, location_condition, sample_size
            )
        
        if not pairs:
            print("没有找到符合条件的样本对")
            return None
        
        similarities = []
        pair_details = []
        
        # 分批计算相似度，避免内存溢出
        batch_size = 10000
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            batch_similarities = self._calculate_batch_similarity(batch_pairs, modality)
            similarities.extend(batch_similarities)
            
            # 为前几个样本对保存详细信息
            if i == 0:
                for j, (idx_i, idx_j) in enumerate(batch_pairs[:10]):
                    with h5py.File(self.h5_path, 'r') as h5_file:
                        pair_details.append({
                            'sample_i': idx_i,
                            'sample_j': idx_j,
                            'similarity': batch_similarities[j],
                            'health_i': int(h5_file['health_labels'][idx_i]),
                            'health_j': int(h5_file['health_labels'][idx_j]),
                            'location_i': int(h5_file['location_labels'][idx_i]),
                            'location_j': int(h5_file['location_labels'][idx_j])
                        })
            
            # 清理内存
            if (i // batch_size) % 10 == 0:
                torch.cuda.empty_cache()
        
        results = {
            'similarities': similarities,
            'pair_details': pair_details,
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'median_similarity': float(np.median(similarities)),
            'q1_similarity': float(np.percentile(similarities, 25)),
            'q3_similarity': float(np.percentile(similarities, 75)),
            'condition_config': condition_config,
            'num_pairs': len(pairs)
        }
        
        self._print_results(results)
        return results
    
    def calculate_category_wise_similarity(self, modality='image-text', sample_size_per_category=200, sample_fraction=0.2):
        """遍历每个死因类别和部位类别，计算相同类和不同类的相似度"""
        if not hasattr(self, 'h5_path') or self.h5_path is None:
            raise ValueError("请先调用 extract_features_to_disk 方法")
        
        print("\n开始遍历所有类别计算相似度...")
        
        # 从HDF5文件中获取类别信息
        with h5py.File(self.h5_path, 'r') as h5_file:
            health_labels = h5_file['health_labels'][:]
            location_labels = h5_file['location_labels'][:]
            
            # 获取所有唯一的类别
            unique_health_categories = np.unique(health_labels)
            unique_location_categories = np.unique(location_labels)
            
            print(f"发现 {len(unique_health_categories)} 个死因类别")
            print(f"发现 {len(unique_location_categories)} 个部位类别")
        
        all_results = {}
        
        # 1. 遍历每个死因类别
        print("\n=== 计算死因类别相似度 ===")
        for health_category in tqdm(unique_health_categories):
            health_name = self._get_health_category_name(health_category)
            
            # 相同死因类别的相似度
            same_health_result = self._calculate_category_specific_similarity(
                modality=modality,
                target_health_category=int(health_category),  # 转换为Python int
                target_location_category=None,
                health_condition='same',
                location_condition='any',
                sample_size=sample_size_per_category,
                sample_fraction=sample_fraction
            )
            
            if same_health_result:
                all_results[f"health_{health_category}_same"] = same_health_result
            
            # 不同死因类别的相似度
            different_health_result = self._calculate_category_specific_similarity(
                modality=modality,
                target_health_category=int(health_category),  # 转换为Python int
                target_location_category=None,
                health_condition='different',
                location_condition='any',
                sample_size=sample_size_per_category,
                sample_fraction=sample_fraction
            )
            
            if different_health_result:
                all_results[f"health_{health_category}_different"] = different_health_result
        
        # 2. 遍历每个部位类别
        print("\n=== 计算部位类别相似度 ===")
        for location_category in tqdm(unique_location_categories):
            location_name = self._get_location_category_name(location_category)
            
            # 相同部位类别的相似度
            same_location_result = self._calculate_category_specific_similarity(
                modality=modality,
                target_health_category=None,
                target_location_category=int(location_category),  # 转换为Python int
                health_condition='any',
                location_condition='same',
                sample_size=sample_size_per_category,
                sample_fraction=sample_fraction
            )
            
            if same_location_result:
                all_results[f"location_{location_category}_same"] = same_location_result
            
            # 不同部位类别的相似度
            different_location_result = self._calculate_category_specific_similarity(
                modality=modality,
                target_health_category=None,
                target_location_category=int(location_category),  # 转换为Python int
                health_condition='any',
                location_condition='different',
                sample_size=sample_size_per_category,
                sample_fraction=sample_fraction
            )
            
            if different_location_result:
                all_results[f"location_{location_category}_different"] = different_location_result
        
        return all_results

    def _calculate_category_specific_similarity(self, modality, target_health_category, target_location_category, 
                                            health_condition, location_condition, sample_size, sample_fraction):
        """计算特定类别的相似度，如果样本不足则使用所有可用样本"""
        with h5py.File(self.h5_path, 'r') as h5_file:
            n_samples = self.total_samples
            
            # 使用采样来减少内存使用
            if sample_fraction < 1.0:
                sample_indices = np.random.choice(
                    n_samples, 
                    size=int(n_samples * sample_fraction), 
                    replace=False
                )
            else:
                sample_indices = np.arange(n_samples)
            
            # 生成符合条件的样本对
            pairs = self._generate_category_specific_pairs(
                h5_file, sample_indices, target_health_category, target_location_category,
                health_condition, location_condition, sample_size
            )
        
        if not pairs:
            print(f"警告: 没有找到符合条件的样本对 - 健康类别: {target_health_category}, 部位类别: {target_location_category}, 健康条件: {health_condition}, 部位条件: {location_condition}")
            return None
        
        # 如果生成的样本对数量少于要求的sample_size，使用警告信息
        if len(pairs) < sample_size:
            print(f"注意: 只生成了 {len(pairs)} 个样本对 (要求 {sample_size}) - 健康类别: {target_health_category}, 部位类别: {target_location_category}, 健康条件: {health_condition}, 部位条件: {location_condition}")
        
        similarities = []
        
        # 分批计算相似度，避免内存溢出
        batch_size = 1000
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            batch_similarities = self._calculate_batch_similarity(batch_pairs, modality)
            similarities.extend(batch_similarities)
            
            # 清理内存
            if (i // batch_size) % 10 == 0:
                torch.cuda.empty_cache()
        
        # 构建结果
        condition_config = {
            'modality': modality,
            'health_condition': health_condition,
            'location_condition': location_condition,
            'target_health_category': target_health_category,
            'target_location_category': target_location_category
        }
        
        results = {
            'similarities': similarities,
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'median_similarity': float(np.median(similarities)),
            'q1_similarity': float(np.percentile(similarities, 25)),
            'q3_similarity': float(np.percentile(similarities, 75)),
            'condition_config': condition_config,
            'num_pairs': len(pairs)
        }
        
        return results
    
    def _generate_category_specific_pairs(self, h5_file, indices, target_health_category, target_location_category,
                                        health_condition, location_condition, sample_size):
        """生成特定类别的样本对，如果样本不足则使用所有可用样本"""
        pairs = []
        max_attempts = sample_size * 100  # 增加最大尝试次数
        
        health_labels = h5_file['health_labels'][:]
        location_labels = h5_file['location_labels'][:]
        
        # 过滤出符合条件的索引
        filtered_indices = indices.copy()
        
        # 如果指定了目标健康类别，只考虑该类别
        if target_health_category is not None:
            if health_condition == 'same':
                # 对于相同类别，两个样本都必须是目标类别
                filtered_indices = [idx for idx in filtered_indices if health_labels[idx] == target_health_category]
            elif health_condition == 'different':
                # 对于不同类别，第一个样本是目标类别，第二个样本不是
                filtered_indices_i = [idx for idx in filtered_indices if health_labels[idx] == target_health_category]
                filtered_indices_j = [idx for idx in filtered_indices if health_labels[idx] != target_health_category]
        
        # 如果指定了目标部位类别，只考虑该类别
        if target_location_category is not None:
            if location_condition == 'same':
                # 对于相同类别，两个样本都必须是目标类别
                filtered_indices = [idx for idx in filtered_indices if location_labels[idx] == target_location_category]
            elif location_condition == 'different':
                # 对于不同类别，第一个样本是目标类别，第二个样本不是
                filtered_indices_i = [idx for idx in filtered_indices if location_labels[idx] == target_location_category]
                filtered_indices_j = [idx for idx in filtered_indices if location_labels[idx] != target_location_category]
        
        # 检查是否有足够的样本
        if target_health_category is not None and health_condition == 'different':
            if len(filtered_indices_i) == 0 or len(filtered_indices_j) == 0:
                return []
        elif target_location_category is not None and location_condition == 'different':
            if len(filtered_indices_i) == 0 or len(filtered_indices_j) == 0:
                return []
        else:
            if len(filtered_indices) < 2:
                return []
        
        attempts = 0
        while len(pairs) < sample_size and attempts < max_attempts:
            attempts += 1
            
            if target_health_category is not None and health_condition == 'different':
                # 不同健康类别的情况
                if len(filtered_indices_i) == 0 or len(filtered_indices_j) == 0:
                    break
                i_idx = np.random.choice(filtered_indices_i)
                j_idx = np.random.choice(filtered_indices_j)
            elif target_location_category is not None and location_condition == 'different':
                # 不同部位类别的情况
                if len(filtered_indices_i) == 0 or len(filtered_indices_j) == 0:
                    break
                i_idx = np.random.choice(filtered_indices_i)
                j_idx = np.random.choice(filtered_indices_j)
            else:
                # 相同类别或任意类别的情况
                if len(filtered_indices) < 2:
                    break
                i_idx, j_idx = np.random.choice(filtered_indices, 2, replace=False)
            
            if i_idx == j_idx:
                continue
            
            # 检查其他条件
            if health_condition == 'same' and health_labels[i_idx] != health_labels[j_idx]:
                continue
            elif health_condition == 'different' and health_labels[i_idx] == health_labels[j_idx]:
                continue
            
            if location_condition == 'same' and location_labels[i_idx] != location_labels[j_idx]:
                continue
            elif location_condition == 'different' and location_labels[i_idx] == location_labels[j_idx]:
                continue
            
            pairs.append((int(i_idx), int(j_idx)))
        
        # 如果样本对数量不足，尝试使用重复采样
        if len(pairs) < sample_size and len(pairs) > 0:
            print(f"警告: 类别 {target_health_category if target_health_category is not None else target_location_category} 样本不足，仅生成 {len(pairs)} 个样本对")
        
        return pairs
    
    def _get_health_category_name(self, category_id):
        """获取死因类别名称"""
        if self.config and 'category' in self.config:
            for key, value in self.config['category'].items():
                if int(key) - 1 == category_id:
                    return value
        return f"health_{category_id}"
    
    def _get_location_category_name(self, category_id):
        """获取部位类别名称"""
        if self.config and 'location' in self.config:
            for key, value in self.config['location'].items():
                if int(key) - 1 == category_id:
                    return value
        return f"location_{category_id}"
    
    def _generate_sample_pairs_from_indices(self, h5_file, indices, health_condition, location_condition, sample_size):
        """从索引生成符合条件的样本对"""
        pairs = []
        max_attempts = sample_size * 20
        
        health_labels = h5_file['health_labels'][:]
        location_labels = h5_file['location_labels'][:]
        
        attempts = 0
        while len(pairs) < sample_size and attempts < max_attempts:
            attempts += 1
            i_idx, j_idx = np.random.choice(indices, 2, replace=False)
            
            if i_idx == j_idx:
                continue
            
            # 检查健康条件
            if health_condition == 'same' and health_labels[i_idx] != health_labels[j_idx]:
                continue
            elif health_condition == 'different' and health_labels[i_idx] == health_labels[j_idx]:
                continue
            
            # 检查部位条件 - 修复变量名错误
            if location_condition == 'same' and location_labels[i_idx] != location_labels[j_idx]:
                continue
            elif location_condition == 'different' and location_labels[i_idx] == location_labels[j_idx]:
                continue
            
            pairs.append((int(i_idx), int(j_idx)))
        
        return pairs
    
    def _calculate_batch_similarity(self, pairs, modality):
        """批量计算相似度"""
        similarities = []
        
        with h5py.File(self.h5_path, 'r') as h5_file:
            image_features = h5_file['image_features']
            text_features = h5_file['text_features']
            
            for i_idx, j_idx in pairs:
                if modality == 'image-text':
                    img_feat = torch.tensor(image_features[i_idx]).to(self.device)
                    txt_feat = torch.tensor(text_features[j_idx]).to(self.device)
                    sim = (img_feat @ txt_feat.T).item()
                elif modality == 'image-image':
                    img_feat_i = torch.tensor(image_features[i_idx]).to(self.device)
                    img_feat_j = torch.tensor(image_features[j_idx]).to(self.device)
                    sim = (img_feat_i @ img_feat_j.T).item()
                elif modality == 'text-text':
                    txt_feat_i = torch.tensor(text_features[i_idx]).to(self.device)
                    txt_feat_j = torch.tensor(text_features[j_idx]).to(self.device)
                    sim = (txt_feat_i @ txt_feat_j.T).item()
                else:
                    sim = 0.0
                
                similarities.append(sim)
        
        return similarities
    
    def _print_results(self, results):
        """打印结果"""
        config = results['condition_config']
        print(f"\n{'='*60}")
        print(f"相似度计算结果")
        print(f"{'='*60}")
        print(f"模态: {config['modality']}")
        print(f"健康条件: {config['health_condition']}")
        print(f"部位条件: {config['location_condition']}")
        print(f"样本对数量: {results['num_pairs']}")
        print(f"平均相似度: {results['mean_similarity']:.4f}")
        print(f"相似度标准差: {results['std_similarity']:.4f}")
        print(f"最小相似度: {results['min_similarity']:.4f}")
        print(f"最大相似度: {results['max_similarity']:.4f}")
        print(f"中位数相似度: {results['median_similarity']:.4f}")
        print(f"Q1 (25%分位数): {results['q1_similarity']:.4f}")
        print(f"Q3 (75%分位数): {results['q3_similarity']:.4f}")

# 数据集类保持不变
class MedicalCsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, config, sep=",", tokenizer=None):
        logging.debug(f'Loading medical csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.category = df["category"].tolist()
        self.location = df["location"].tolist()
        self.health_label_to_idx = {v: int(k)-1 for k, v in config['category'].items()}
        self.location_label_to_idx = {v: int(k)-1 for k, v in config['location'].items()}
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading medical data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        try:
            images = self.transforms(Image.open(str(self.images[idx])))
        except Exception as e:
            print(f"警告：样本 {idx} 的图像加载失败，使用随机图像替代: {str(e)}")
            images = torch.randn(3, 224, 224)
            
        if self.tokenize:
            texts = self.tokenize([str(self.captions[idx])])[0]
        else:
            texts = str(self.captions[idx])
            
        category = str(self.category[idx])
        location = str(self.location[idx])
        
        # 健康状况标签
        health_name = self.category[idx] 
        health_label = torch.zeros(len(self.health_label_to_idx), dtype=torch.float)
        if health_name in self.health_label_to_idx:
            health_label[self.health_label_to_idx[health_name]] = 1.0
                
        # 身体部位标签
        location_name = self.location[idx]
        location_label = torch.zeros(len(self.location_label_to_idx), dtype=torch.float)
        if location_name in self.location_label_to_idx:
            location_label[self.location_label_to_idx[location_name]] = 1.0
        
        return images, texts, health_label, location_label, category, location, self.captions[idx], self.images[idx]

def save_raw_data_to_csv(all_results, calculator, save_dir):
    """保存原始数据到CSV文件"""
    raw_data = []
    
    for cond_name, result in all_results.items():
        config = result['condition_config']
        
        # 获取类别信息
        health_category_name = None
        location_category_name = None
        comparison_type = None
        
        if cond_name.startswith('health_'):
            category_id = cond_name.split('_')[1]
            health_category_name = calculator._get_health_category_name(int(category_id))
            comparison_type = 'same' if cond_name.endswith('_same') else 'different'
        elif cond_name.startswith('location_'):
            category_id = cond_name.split('_')[1]
            location_category_name = calculator._get_location_category_name(int(category_id))
            comparison_type = 'same' if cond_name.endswith('_same') else 'different'
        
        # 为每个相似度值创建一行
        for similarity in result['similarities']:
            raw_data.append({
                'condition_name': cond_name,
                'health_category': health_category_name,
                'location_category': location_category_name,
                'comparison_type': comparison_type,
                'modality': config['modality'],
                'similarity_value': similarity,
                'mean_similarity': result['mean_similarity'],
                'std_similarity': result['std_similarity'],
                'num_pairs': result['num_pairs']
            })
    
    # 创建DataFrame并保存
    if raw_data:
        raw_df = pd.DataFrame(raw_data)
        csv_path = os.path.join(save_dir, "raw_similarity_data.csv")
        raw_df.to_csv(csv_path, index=False)
        print(f"原始相似度数据已保存到: {csv_path}")
        
        # 同时保存汇总数据
        save_summary_data(all_results, calculator, save_dir)

def save_summary_data(all_results, calculator, save_dir):
    """保存汇总数据到CSV"""
    summary_data = []
    
    for cond_name, result in all_results.items():
        config = result['condition_config']
        
        # 获取类别信息
        health_category_name = None
        location_category_name = None
        comparison_type = None
        
        if cond_name.startswith('health_'):
            category_id = cond_name.split('_')[1]
            health_category_name = calculator._get_health_category_name(int(category_id))
            comparison_type = 'same' if cond_name.endswith('_same') else 'different'
        elif cond_name.startswith('location_'):
            category_id = cond_name.split('_')[1]
            location_category_name = calculator._get_location_category_name(int(category_id))
            comparison_type = 'same' if cond_name.endswith('_same') else 'different'
        
        summary_data.append({
            'condition_name': cond_name,
            'health_category': health_category_name,
            'location_category': location_category_name,
            'comparison_type': comparison_type,
            'modality': config['modality'],
            'mean_similarity': result['mean_similarity'],
            'std_similarity': result['std_similarity'],
            'min_similarity': result['min_similarity'],
            'max_similarity': result['max_similarity'],
            'median_similarity': result['median_similarity'],
            'q1_similarity': result['q1_similarity'],
            'q3_similarity': result['q3_similarity'],
            'num_pairs': result['num_pairs']
        })
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(save_dir, "similarity_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"相似度汇总数据已保存到: {csv_path}")

# 新增函数：绘制相似度-概率密度图 - 分成多张图片
# 修改可视化函数，调整图例位置避免遮挡
def plot_similarity_density_comparison_separate(all_results, calculator, save_dir="./similarity_results"):
    """绘制相似度概率密度比较图，每个类别单独一张图，视觉美化配色，优化图例位置"""
    os.makedirs(save_dir, exist_ok=True)

    # 使用指定的颜色方案
    disease_same_color = (66/255, 145/255, 178/255)    # RGB(66,145,178)
    disease_diff_color = (170/255, 78/255, 126/255)    # RGB(170,78,126)
    
    location_same_color = (241/255, 153/255, 26/255)   # RGB(241,153,26)
    location_diff_color = (203/255, 80/255, 51/255)    # RGB(203,80,51)
    
    # 重叠区域样式
    overlap_hatch = '////'  # 斜线填充
    overlap_alpha = 0.3     # 透明度
    
    bg_color = "white"

    # 分离死因类别和部位类别的结果
    health_same_results = {}
    health_different_results = {}
    location_same_results = {}
    location_different_results = {}

    for key, result in all_results.items():
        if key.startswith('health_') and key.endswith('_same'):
            category_id = key.split('_')[1]
            health_same_results[category_id] = result
        elif key.startswith('health_') and key.endswith('_different'):
            category_id = key.split('_')[1]
            health_different_results[category_id] = result
        elif key.startswith('location_') and key.endswith('_same'):
            category_id = key.split('_')[1]
            location_same_results[category_id] = result
        elif key.startswith('location_') and key.endswith('_different'):
            category_id = key.split('_')[1]
            location_different_results[category_id] = result

    # 保存原始数据到CSV
    save_raw_data_to_csv(all_results, calculator, save_dir)

    def _get_values_or_approx(res):
        if res is None:
            return None
        if 'similarities' in res and res['similarities']:
            vals = np.asarray(res['similarities'], dtype=float)
            if len(vals) > 0:
                # clip 到 [0,1] 以防异常值
                vals = np.clip(vals, 0.0, 1.0)
                return vals
        mean = float(res.get('mean_similarity', 0.0))
        std = float(res.get('std_similarity', 1e-6))
        n = int(res.get('num_pairs', 100))
        if std <= 0:
            std = 1e-6
        samples = np.random.normal(loc=mean, scale=std, size=max(50, min(n, 1000)))
        return np.clip(samples, 0.0, 1.0)

    def _compute_pdf(values, x_grid=None):
        if values is None or len(values) == 0:
            return None, None
        values = np.asarray(values, dtype=float)
        values = np.clip(values, 0.0, 1.0)
        if len(values) >= 2:
            try:
                kde = stats.gaussian_kde(values)
                if x_grid is None:
                    xmin, xmax = values.min(), values.max()
                    span = max(0.001, (xmax - xmin) * 0.2)
                    low = max(0.0, xmin - span)
                    high = min(1.0, xmax + span)
                    if low >= high:
                        low, high = 0.0, 1.0
                    x_grid = np.linspace(low, high, 2000)
                else:
                    x_grid = np.clip(x_grid, 0.0, 1.0)
                pdf = kde(x_grid)
                return x_grid, pdf
            except Exception:
                pass
        mu = float(np.mean(values))
        sigma = float(np.std(values))
        if sigma <= 0:
            sigma = 1e-6
        if x_grid is None:
            low = max(0.0, mu - 4*sigma)
            high = min(1.0, mu + 4*sigma)
            if low >= high:
                low, high = 0.0, 1.0
            x_grid = np.linspace(low, high, 2000)
        else:
            x_grid = np.clip(x_grid, 0.0, 1.0)
        pdf = stats.norm.pdf(x_grid, loc=mu, scale=sigma)
        return x_grid, pdf

    # 绘制每个死因类别密度图 - 优化图例位置
    if health_same_results or health_different_results:
        health_density_dir = os.path.join(save_dir, "health_density_plots_beautified")
        os.makedirs(health_density_dir, exist_ok=True)

        all_ids = sorted(set(list(health_same_results.keys()) + list(health_different_results.keys())), key=lambda x: int(x))
        for category_id in all_ids:
            same_res = health_same_results.get(category_id)
            diff_res = health_different_results.get(category_id)
            same_vals = _get_values_or_approx(same_res) if same_res is not None else None
            diff_vals = _get_values_or_approx(diff_res) if diff_res is not None else None

            if same_vals is None and diff_vals is None:
                continue

            # x 网格统一在 [0,1]
            x_grid = np.linspace(0.0, 1.0, 2000)
            x1, pdf1 = _compute_pdf(same_vals, x_grid=x_grid) if same_vals is not None else (x_grid, np.zeros_like(x_grid))
            x2, pdf2 = _compute_pdf(diff_vals, x_grid=x_grid) if diff_vals is not None else (x_grid, np.zeros_like(x_grid))

            overlap = np.trapz(np.minimum(pdf1, pdf2), x_grid)

            # 创建瀑布图样式的密度图
            plt.figure(figsize=(14, 8), facecolor=bg_color)  # 增加图形尺寸
            
            # 设置样式
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # 绘制密度曲线 - 使用瀑布图样式
            if same_vals is not None:
                plt.plot(x_grid, pdf1, label=f"Same Disease (n={same_res.get('num_pairs', len(same_vals))})",
                         color=disease_same_color, linewidth=3.0, alpha=0.9)
                # 填充曲线下方区域
                plt.fill_between(x_grid, pdf1, alpha=0.4, color=disease_same_color)
            
            if diff_vals is not None:
                plt.plot(x_grid, pdf2, label=f"Different Disease (n={diff_res.get('num_pairs', len(diff_vals))})",
                         color=disease_diff_color, linewidth=3.0, alpha=0.9, linestyle='--')
                # 填充曲线下方区域
                plt.fill_between(x_grid, pdf2, alpha=0.3, color=disease_diff_color)

            # 重叠区域用斜线填充
            mask = np.minimum(pdf1, pdf2) > 0
            plt.fill_between(x_grid, np.minimum(pdf1, pdf2), where=mask, 
                            facecolor='gray', alpha=overlap_alpha, hatch=overlap_hatch, 
                            edgecolor='black', linewidth=0.5)

            # 美化图表
            plt.xlabel('Similarity Value', fontsize=14, fontweight='bold')
            plt.xlim(0.0, 1.0)
            plt.ylabel('Probability Density', fontsize=14, fontweight='bold')
            category_name = calculator._get_health_category_name(int(category_id))
            plt.title(f'Disease Similarity Density — {category_name}\nOverlap: {overlap:.3f}', 
                     fontsize=16, fontweight='bold', pad=20)
            
            # 优化图例位置 - 放在图表外部右上角
            legend = plt.legend(frameon=True, fontsize=12, loc='upper right',
                              fancybox=True, shadow=True, framealpha=0.9,
                              bbox_to_anchor=(0.98, 0.98))  # 调整位置到右上角
            legend.get_frame().set_facecolor('white')
            
            # 美化坐标轴
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_linewidth(1.5)
            plt.gca().spines['bottom'].set_linewidth(1.5)

            # 调整布局，为图例留出空间
            plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # 调整rect参数为图例留空间

            save_path = os.path.join(health_density_dir, f"health_{category_id}_{category_name}_density_beautified.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=bg_color, 
                       edgecolor='none', transparent=False)
            plt.close()

        print(f"美化版死因类别相似度密度图已保存到: {health_density_dir}")

    # 绘制每个部位类别密度图 - 优化图例位置
    if location_same_results or location_different_results:
        location_density_dir = os.path.join(save_dir, "location_density_plots_beautified")
        os.makedirs(location_density_dir, exist_ok=True)

        all_ids = sorted(set(list(location_same_results.keys()) + list(location_different_results.keys())), key=lambda x: int(x))
        for category_id in all_ids:
            same_res = location_same_results.get(category_id)
            diff_res = location_different_results.get(category_id)
            same_vals = _get_values_or_approx(same_res) if same_res is not None else None
            diff_vals = _get_values_or_approx(diff_res) if diff_res is not None else None

            if same_vals is None and diff_vals is None:
                continue

            x_grid = np.linspace(0.0, 1.0, 2000)
            x1, pdf1 = _compute_pdf(same_vals, x_grid=x_grid) if same_vals is not None else (x_grid, np.zeros_like(x_grid))
            x2, pdf2 = _compute_pdf(diff_vals, x_grid=x_grid) if diff_vals is not None else (x_grid, np.zeros_like(x_grid))

            overlap = np.trapz(np.minimum(pdf1, pdf2), x_grid)

            # 创建瀑布图样式的密度图
            plt.figure(figsize=(14, 8), facecolor=bg_color)  # 增加图形尺寸
            
            # 设置样式
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # 绘制密度曲线
            if same_vals is not None:
                plt.plot(x_grid, pdf1, label=f"Same Location (n={same_res.get('num_pairs', len(same_vals))})",
                         color=location_same_color, linewidth=3.0, alpha=0.9)
                # 填充曲线下方区域
                plt.fill_between(x_grid, pdf1, alpha=0.4, color=location_same_color)
            
            if diff_vals is not None:
                plt.plot(x_grid, pdf2, label=f"Different Location (n={diff_res.get('num_pairs', len(diff_vals))})",
                         color=location_diff_color, linewidth=3.0, alpha=0.9, linestyle='--')
                # 填充曲线下方区域
                plt.fill_between(x_grid, pdf2, alpha=0.3, color=location_diff_color)

            # 重叠区域用斜线填充
            mask = np.minimum(pdf1, pdf2) > 0
            plt.fill_between(x_grid, np.minimum(pdf1, pdf2), where=mask, 
                            facecolor='gray', alpha=overlap_alpha, hatch=overlap_hatch, 
                            edgecolor='black', linewidth=0.5)

            # 美化图表
            location_name = calculator._get_location_category_name(int(category_id))
            plt.xlabel('Similarity Value', fontsize=14, fontweight='bold')
            plt.xlim(0.0, 1.0)
            plt.ylabel('Probability Density', fontsize=14, fontweight='bold')
            plt.title(f'Location Similarity Density — {location_name}\nOverlap: {overlap:.3f}', 
                     fontsize=16, fontweight='bold', pad=20)
            
            # 优化图例位置 - 放在图表外部右上角
            legend = plt.legend(frameon=True, fontsize=12, loc='upper right',
                              fancybox=True, shadow=True, framealpha=0.9,
                              bbox_to_anchor=(0.98, 0.98))  # 调整位置到右上角
            legend.get_frame().set_facecolor('white')
            
            # 美化坐标轴
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_linewidth(1.5)
            plt.gca().spines['bottom'].set_linewidth(1.5)

            # 调整布局，为图例留出空间
            plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # 调整rect参数为图例留空间

            save_path = os.path.join(location_density_dir, f"location_{category_id}_{location_name}_density_beautified.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=bg_color, 
                       edgecolor='none', transparent=False)
            plt.close()

        print(f"美化版部位类别相似度密度图已保存到: {location_density_dir}")

    # 3. 创建总体比较图 - 也优化图例位置
    create_overall_comparison_plots(health_same_results, health_different_results,
                                  location_same_results, location_different_results,
                                  calculator, save_dir)

def create_overall_comparison_plots(health_same_results, health_different_results,
                                  location_same_results, location_different_results,
                                  calculator, save_dir):
    """创建总体比较图并计算/标注重叠区域，视觉美化配色，优化图例位置"""
    # 使用指定的颜色方案
    disease_same_color = (66/255, 145/255, 178/255)
    disease_diff_color = (170/255, 78/255, 126/255)
    location_same_color = (241/255, 153/255, 26/255)
    location_diff_color = (203/255, 80/255, 51/255)
    
    overlap_hatch = '////'
    overlap_alpha = 0.3
    bg_color = "white"

    def _collect_all_values(results_dict):
        vals = []
        weighted_pdf_components = []
        for res in (results_dict or {}).values():
            if res is None:
                continue
            if 'similarities' in res and res['similarities']:
                vals.extend([float(v) for v in res['similarities']])
            else:
                mean = float(res.get('mean_similarity', 0.0))
                std = float(res.get('std_similarity', 1e-6))
                n = int(res.get('num_pairs', 0) or 0)
                if n > 0:
                    weighted_pdf_components.append((mean, max(std, 1e-6), n))
        vals = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)
        return vals, weighted_pdf_components

    def _mixture_pdf_from_components(x, components):
        if not components:
            return np.zeros_like(x)
        total_w = sum([c[2] for c in components]) or 1
        pdf = np.zeros_like(x, dtype=float)
        for mu, sd, w in components:
            pdf += (w * stats.norm.pdf(x, loc=mu, scale=sd))
        return pdf / total_w

    os.makedirs(save_dir, exist_ok=True)

    # overall disease comparison - 美化版
    same_vals, same_components = _collect_all_values(health_same_results)
    diff_vals, diff_components = _collect_all_values(health_different_results)
    if same_vals.size == 0 and not same_components:
        same_vals = None
    if diff_vals.size == 0 and not diff_components:
        diff_vals = None

    if same_vals is not None or diff_vals is not None:
        x = np.linspace(0.0, 1.0, 3000)

        if same_vals is not None and len(same_vals) > 1:
            kde_s = stats.gaussian_kde(same_vals)
            pdf_s = kde_s(x)
        else:
            pdf_s = _mixture_pdf_from_components(x, same_components) if same_components else np.zeros_like(x)

        if diff_vals is not None and len(diff_vals) > 1:
            kde_d = stats.gaussian_kde(diff_vals)
            pdf_d = kde_d(x)
        else:
            pdf_d = _mixture_pdf_from_components(x, diff_components) if diff_components else np.zeros_like(x)

        overlap = np.trapz(np.minimum(pdf_s, pdf_d), x)

        plt.figure(figsize=(14, 8), facecolor=bg_color)  # 增加图形尺寸
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 绘制密度曲线
        if same_vals is not None:
            plt.plot(x, pdf_s, label="Overall Same Disease", color=disease_same_color, linewidth=3.0, alpha=0.9)
            plt.fill_between(x, pdf_s, alpha=0.4, color=disease_same_color)
        
        if diff_vals is not None:
            plt.plot(x, pdf_d, label="Overall Different Disease", color=disease_diff_color, linewidth=3.0, alpha=0.9, linestyle='--')
            plt.fill_between(x, pdf_d, alpha=0.3, color=disease_diff_color)
        
        # 重叠区域用斜线填充
        mask = np.minimum(pdf_s, pdf_d) > 0
        plt.fill_between(x, np.minimum(pdf_s, pdf_d), where=mask, 
                        facecolor='gray', alpha=overlap_alpha, hatch=overlap_hatch, 
                        edgecolor='black', linewidth=0.5)

        plt.xlabel('Similarity Value', fontsize=14, fontweight='bold')
        plt.xlim(0.0, 1.0)
        plt.ylabel('Probability Density', fontsize=14, fontweight='bold')
        plt.title(f'Overall Disease Category Similarity Density Comparison\nOverlap: {overlap:.3f}', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # 优化图例位置
        legend = plt.legend(frameon=True, fontsize=12, loc='upper right',
                          fancybox=True, shadow=True, framealpha=0.9,
                          bbox_to_anchor=(0.98, 0.98))
        legend.get_frame().set_facecolor('white')
        
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)

        # 调整布局
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])

        save_path = os.path.join(save_dir, "overall_disease_similarity_density_beautified.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
        plt.close()
        print(f"美化版死因类别总体相似度密度比较图已保存: {save_path}  overlap={overlap:.4f}")

    # overall location comparison - 美化版
    same_vals, same_components = _collect_all_values(location_same_results)
    diff_vals, diff_components = _collect_all_values(location_different_results)
    if same_vals.size == 0 and not same_components:
        same_vals = None
    if diff_vals.size == 0 and not diff_components:
        diff_vals = None

    if same_vals is not None or diff_vals is not None:
        x = np.linspace(0.0, 1.0, 3000)

        if same_vals is not None and len(same_vals) > 1:
            kde_s = stats.gaussian_kde(same_vals)
            pdf_s = kde_s(x)
        else:
            pdf_s = _mixture_pdf_from_components(x, same_components) if same_components else np.zeros_like(x)

        if diff_vals is not None and len(diff_vals) > 1:
            kde_d = stats.gaussian_kde(diff_vals)
            pdf_d = kde_d(x)
        else:
            pdf_d = _mixture_pdf_from_components(x, diff_components) if diff_components else np.zeros_like(x)

        overlap = np.trapz(np.minimum(pdf_s, pdf_d), x)

        plt.figure(figsize=(14, 8), facecolor=bg_color)  # 增加图形尺寸
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 绘制密度曲线
        if same_vals is not None:
            plt.plot(x, pdf_s, label="Overall Same Location", color=location_same_color, linewidth=3.0, alpha=0.9)
            plt.fill_between(x, pdf_s, alpha=0.4, color=location_same_color)
        
        if diff_vals is not None:
            plt.plot(x, pdf_d, label="Overall Different Location", color=location_diff_color, linewidth=3.0, alpha=0.9, linestyle='--')
            plt.fill_between(x, pdf_d, alpha=0.3, color=location_diff_color)
        
        # 重叠区域用斜线填充
        mask = np.minimum(pdf_s, pdf_d) > 0
        plt.fill_between(x, np.minimum(pdf_s, pdf_d), where=mask, 
                        facecolor='gray', alpha=overlap_alpha, hatch=overlap_hatch, 
                        edgecolor='black', linewidth=0.5)

        plt.xlabel('Similarity Value', fontsize=14, fontweight='bold')
        plt.xlim(0.0, 1.0)
        plt.ylabel('Probability Density', fontsize=14, fontweight='bold')
        plt.title(f'Overall Location Category Similarity Density Comparison\nOverlap: {overlap:.3f}', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # 优化图例位置
        legend = plt.legend(frameon=True, fontsize=12, loc='upper right',
                          fancybox=True, shadow=True, framealpha=0.9,
                          bbox_to_anchor=(0.98, 0.98))
        legend.get_frame().set_facecolor('white')
        
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)

        # 调整布局
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])

        save_path = os.path.join(save_dir, "overall_location_similarity_density_beautified.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
        plt.close()
        print(f"美化版部位类别总体相似度密度比较图已保存: {save_path}  overlap={overlap:.4f}")

    return

# 修改可视化函数，确保所有类别都被绘制
def visualize_similarity_comparison_separate(all_results, calculator, save_dir="./similarity_results"):
    """将相似度比较图分成location和category两部分，用不同颜色标记different和same，并添加误差棒"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 分离category和location的结果
    category_results = {}
    location_results = {}
    
    for key, result in all_results.items():
        if key.startswith('health_'):
            category_results[key] = result
        elif key.startswith('location_'):
            location_results[key] = result
    
    # 定义颜色
    same_color = 'skyblue'
    different_color = 'lightcoral'
    
    # 1. Category (Health) 比较图
    if category_results:
        plt.figure(figsize=(max(14, len(category_results)//3), 8))
        
        # 按类别分组，每个类别有same和different两个结果
        category_groups = {}
        for key, result in category_results.items():
            config = result['condition_config']
            category_id = config['target_health_category']
            category_name = calculator._get_health_category_name(category_id)
            
            if category_name not in category_groups:
                category_groups[category_name] = {'same': None, 'different': None}
            
            if key.endswith('_same'):
                category_groups[category_name]['same'] = result
            elif key.endswith('_different'):
                category_groups[category_name]['different'] = result
        
        # 准备绘图数据 - 现在包括所有类别，即使只有一种结果
        category_labels = []
        same_means = []
        different_means = []
        same_stds = []
        different_stds = []
        same_available = []  # 标记哪些类别有same结果
        different_available = []  # 标记哪些类别有different结果
        
        for category_name, results in category_groups.items():
            category_labels.append(category_name)
            
            if results['same']:
                same_means.append(results['same']['mean_similarity'])
                same_stds.append(results['same']['std_similarity'])
                same_available.append(True)
            else:
                same_means.append(0)  # 占位值，不会显示
                same_stds.append(0)
                same_available.append(False)
            
            if results['different']:
                different_means.append(results['different']['mean_similarity'])
                different_stds.append(results['different']['std_similarity'])
                different_available.append(True)
            else:
                different_means.append(0)  # 占位值，不会显示
                different_stds.append(0)
                different_available.append(False)
        
        # 创建分组柱状图
        x_pos = np.arange(len(category_labels))
        bar_width = 0.35
        
        # 绘制相同类别的柱子 - 只绘制有数据的
        same_bars = []
        for i, (mean, std, available) in enumerate(zip(same_means, same_stds, same_available)):
            if available:
                bar = plt.bar(x_pos[i] - bar_width/2, mean, bar_width, 
                             yerr=std, color=same_color, alpha=0.7, 
                             capsize=5, error_kw={'elinewidth': 1, 'markeredgewidth': 1})[0]
                same_bars.append(bar)
                # 在柱子上添加数值
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 绘制不同类别的柱子 - 只绘制有数据的
        different_bars = []
        for i, (mean, std, available) in enumerate(zip(different_means, different_stds, different_available)):
            if available:
                bar = plt.bar(x_pos[i] + bar_width/2, mean, bar_width,
                             yerr=std, color=different_color, alpha=0.7,
                             capsize=5, error_kw={'elinewidth': 1, 'markeredgewidth': 1})[0]
                different_bars.append(bar)
                # 在柱子上添加数值
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.ylabel('Mean Similarity Value')
        plt.title(f'Disease Category Similarity Comparison (Total: {len(category_labels)} categories)')
        plt.xticks(x_pos, category_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 创建图例
        legend_elements = []
        if same_bars:
            legend_elements.append(plt.Rectangle((0,0),1,1, fc=same_color, alpha=0.7, label='Same Disease'))
        if different_bars:
            legend_elements.append(plt.Rectangle((0,0),1,1, fc=different_color, alpha=0.7, label='Different Disease'))
        
        if legend_elements:
            plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, "disease_category_similarity_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"疾病类别相似度比较图已保存: {save_path}")
        print(f"总共绘制了 {len(category_labels)} 个疾病类别")
    
    # 2. Location 比较图
    if location_results:
        plt.figure(figsize=(max(14, len(location_results)//3), 8))
        
        # 按部位分组，每个部位有same和different两个结果
        location_groups = {}
        for key, result in location_results.items():
            config = result['condition_config']
            location_id = config['target_location_category']
            location_name = calculator._get_location_category_name(location_id)
            
            if location_name not in location_groups:
                location_groups[location_name] = {'same': None, 'different': None}
            
            if key.endswith('_same'):
                location_groups[location_name]['same'] = result
            elif key.endswith('_different'):
                location_groups[location_name]['different'] = result
        
        # 准备绘图数据 - 现在包括所有部位，即使只有一种结果
        location_labels = []
        same_means = []
        different_means = []
        same_stds = []
        different_stds = []
        same_available = []  # 标记哪些部位有same结果
        different_available = []  # 标记哪些部位有different结果
        
        for location_name, results in location_groups.items():
            location_labels.append(location_name)
            
            if results['same']:
                same_means.append(results['same']['mean_similarity'])
                same_stds.append(results['same']['std_similarity'])
                same_available.append(True)
            else:
                same_means.append(0)  # 占位值，不会显示
                same_stds.append(0)
                same_available.append(False)
            
            if results['different']:
                different_means.append(results['different']['mean_similarity'])
                different_stds.append(results['different']['std_similarity'])
                different_available.append(True)
            else:
                different_means.append(0)  # 占位值，不会显示
                different_stds.append(0)
                different_available.append(False)
        
        # 创建分组柱状图
        x_pos = np.arange(len(location_labels))
        bar_width = 0.35
        
        # 绘制相同部位的柱子 - 只绘制有数据的
        same_bars = []
        for i, (mean, std, available) in enumerate(zip(same_means, same_stds, same_available)):
            if available:
                bar = plt.bar(x_pos[i] - bar_width/2, mean, bar_width, 
                             yerr=std, color=same_color, alpha=0.7, 
                             capsize=5, error_kw={'elinewidth': 1, 'markeredgewidth': 1})[0]
                same_bars.append(bar)
                # 在柱子上添加数值
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 绘制不同部位的柱子 - 只绘制有数据的
        different_bars = []
        for i, (mean, std, available) in enumerate(zip(different_means, different_stds, different_available)):
            if available:
                bar = plt.bar(x_pos[i] + bar_width/2, mean, bar_width,
                             yerr=std, color=different_color, alpha=0.7,
                             capsize=5, error_kw={'elinewidth': 1, 'markeredgewidth': 1})[0]
                different_bars.append(bar)
                # 在柱子上添加数值
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.ylabel('Mean Similarity Value')
        plt.title(f'Location Category Similarity Comparison (Total: {len(location_labels)} locations)')
        plt.xticks(x_pos, location_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 创建图例
        legend_elements = []
        if same_bars:
            legend_elements.append(plt.Rectangle((0,0),1,1, fc=same_color, alpha=0.7, label='Same Location'))
        if different_bars:
            legend_elements.append(plt.Rectangle((0,0),1,1, fc=different_color, alpha=0.7, label='Different Location'))
        
        if legend_elements:
            plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, "location_category_similarity_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"部位类别相似度比较图已保存: {save_path}")
        print(f"总共绘制了 {len(location_labels)} 个部位类别")
    
    # 保存详细结果到CSV
    results_data = []
    for cond_name, result in all_results.items():
        config = result['condition_config']
        
        # 添加类别名称信息
        health_category_name = None
        location_category_name = None
        
        if 'target_health_category' in config and config['target_health_category'] is not None:
            health_category_name = calculator._get_health_category_name(config['target_health_category'])
        
        if 'target_location_category' in config and config['target_location_category'] is not None:
            location_category_name = calculator._get_location_category_name(config['target_location_category'])
        
        results_data.append({
            'condition': cond_name,
            'modality': config['modality'],
            'health_condition': config['health_condition'],
            'location_condition': config['location_condition'],
            'health_category_name': health_category_name,
            'location_category_name': location_category_name,
            'mean_similarity': result['mean_similarity'],
            'std_similarity': result['std_similarity'],
            'min_similarity': result['min_similarity'],
            'max_similarity': result['max_similarity'],
            'median_similarity': result['median_similarity'],
            'q1_similarity': result['q1_similarity'],
            'q3_similarity': result['q3_similarity'],
            'num_pairs': result['num_pairs']
        })
    
    results_df = pd.DataFrame(results_data)
    csv_path = os.path.join(save_dir, "similarity_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"详细结果已保存到: {csv_path}")

def check_category_sample_counts(calculator, max_retries=5, retry_delay=1):
    """检查每个类别的样本数量，添加重试机制"""
    if not hasattr(calculator, 'h5_path') or calculator.h5_path is None:
        print("请先提取特征到磁盘")
        return
    
    for attempt in range(max_retries):
        try:
            with h5py.File(calculator.h5_path, 'r') as h5_file:
                health_labels = h5_file['health_labels'][:]
                location_labels = h5_file['location_labels'][:]
                
                # 统计健康类别样本数量
                unique_health, health_counts = np.unique(health_labels, return_counts=True)
                print("\n=== 健康类别样本统计 ===")
                for category_id, count in zip(unique_health, health_counts):
                    category_name = calculator._get_health_category_name(category_id)
                    print(f"{category_name} (ID: {category_id}): {count} 个样本")
                
                # 统计部位类别样本数量
                unique_location, location_counts = np.unique(location_labels, return_counts=True)
                print("\n=== 部位类别样本统计 ===")
                for category_id, count in zip(unique_location, location_counts):
                    category_name = calculator._get_location_category_name(category_id)
                    print(f"{category_name} (ID: {category_id}): {count} 个样本")
            
            # 如果成功打开文件，跳出循环
            break
            
        except (BlockingIOError, OSError) as e:
            print(f"尝试 {attempt + 1}/{max_retries} 打开HDF5文件失败: {e}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print("达到最大重试次数，跳过样本统计")
                return

# 完整的使用示例 - 内存优化版本
def main_memory_efficient():
    """内存优化的主函数"""
    # 加载配置
    JSON_CONFIG_PATH = "/root/autodl-tmp/data/forensic_CT_statistics.json"
    with open(JSON_CONFIG_PATH, 'r', encoding='gbk') as f:
        medical_config = json.load(f)
    
    print(f"加载到 {len(medical_config.get('category', {}))} 个诊断类别")
    print(f"加载到 {len(medical_config.get('location', {}))} 个身体部位")
    
    # 创建数据预处理
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建医学数据集
    print("Loading dataset...")
    medical_dataset = MedicalCsvDataset(
        input_filename="/root/autodl-tmp/data/forensic_CT_train.csv",
        transforms=preprocess,
        img_key="image_path",
        caption_key="original_text",
        config=medical_config
    )
    print(f"Dataset loaded successfully! Total samples: {len(medical_dataset)}")
    
    # 创建数据加载器，使用更小的batch_size
    dataloader = DataLoader(
        medical_dataset, 
        batch_size=1024,  # 减少batch_size以避免内存问题
        shuffle=False,
        num_workers=2
    )
    
    # 创建内存优化的相似度计算器
    print("\nLoading model from checkpoint...")
    try:
        calculator = MemoryEfficientMedicalSimilarityCalculator.from_checkpoint(
            checkpoint_path="/root/autodl-fs/clip_log/2025_11_26-23_26_51-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/checkpoints/best_model.pt",
            model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            config=medical_config
        )
        medical_dataset.tokenize = calculator.tokenizer
        print("Model loaded successfully!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 设置保存路径
    SAVE_ROOT = "/root/MedCLIP-SAMv2-main/similarity_analysis_results"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = os.path.join(SAVE_ROOT, f"similarity_analysis_{timestamp}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. 提取特征到磁盘（使用部分数据进行测试）
    h5_path = os.path.join(SAVE_DIR, "features.h5")
    print("\n开始提取特征到磁盘...")
    
    # 限制样本数量进行测试，避免内存问题
    max_samples = 1000000  # 进一步减少样本数量
    calculator.extract_features_to_disk(dataloader, h5_path, max_samples=max_samples)
    
    print("\n检查类别样本数量...")
    check_category_sample_counts(calculator)
    
    # 2. 遍历所有死因类别和部位类别计算相似度
    print("\n开始遍历所有类别计算相似度...")
    all_results = calculator.calculate_category_wise_similarity(
        modality='text-text',
        sample_size_per_category=128,  # 样本对设置
        sample_fraction=0.8  # 增加采样比例到80%
    )
    
    # 3. 可视化结果
    if all_results:
        print("\n生成可视化结果...")
        
        # 生成分开的柱状图比较
        visualize_similarity_comparison_separate(all_results, calculator, save_dir=SAVE_DIR)
        
        # 生成新的概率密度图（分成多张图片）
        plot_similarity_density_comparison_separate(all_results, calculator, save_dir=SAVE_DIR)
        
        # 保存详细结果
        detailed_results_path = os.path.join(SAVE_DIR, "detailed_similarity_results.json")
        serializable_results = {}
        for cond_name, result in all_results.items():
            # 确保所有值都是可序列化的
            serializable_result = {
                'mean_similarity': float(result['mean_similarity']),
                'std_similarity': float(result['std_similarity']),
                'min_similarity': float(result['min_similarity']),
                'max_similarity': float(result['max_similarity']),
                'median_similarity': float(result['median_similarity']),
                'q1_similarity': float(result['q1_similarity']),
                'q3_similarity': float(result['q3_similarity']),
                'num_pairs': result['num_pairs'],
                'condition_config': result['condition_config']
            }
            
            # 添加类别名称信息
            config = result['condition_config']
            if 'target_health_category' in config and config['target_health_category'] is not None:
                serializable_result['health_category_name'] = calculator._get_health_category_name(
                    config['target_health_category']
                )
            if 'target_location_category' in config and config['target_location_category'] is not None:
                serializable_result['location_category_name'] = calculator._get_location_category_name(
                    config['target_location_category']
                )
            
            serializable_results[cond_name] = serializable_result
        
        with open(detailed_results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n分析完成！所有结果已保存到: {SAVE_DIR}")
        print(f"- 特征文件: {h5_path}")
        print(f"- 可视化结果: {SAVE_DIR}")
        print(f"- 统计报告: {os.path.join(SAVE_DIR, 'similarity_results.csv')}")
        print(f"- 详细结果: {detailed_results_path}")
        print(f"- 原始数据: {os.path.join(SAVE_DIR, 'raw_similarity_data.csv')}")
        print(f"- 汇总数据: {os.path.join(SAVE_DIR, 'similarity_summary.csv')}")
        
        # 打印汇总统计
        print("\n=== 汇总统计 ===")
        same_results = [r for k, r in all_results.items() if k.endswith('_same')]
        different_results = [r for k, r in all_results.items() if k.endswith('_different')]
        
        if same_results:
            same_means = [r['mean_similarity'] for r in same_results]
            print(f"相同类别平均相似度: {np.mean(same_means):.4f} ± {np.std(same_means):.4f}")
        
        if different_results:
            different_means = [r['mean_similarity'] for r in different_results]
            print(f"不同类别平均相似度: {np.mean(different_means):.4f} ± {np.std(different_means):.4f}")
    
    return all_results

if __name__ == "__main__":
    print("开始运行内存优化版本...")
    results = main_memory_efficient()