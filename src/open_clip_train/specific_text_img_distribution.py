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
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpecifiedHealthCategorySimilarityCalculator:
    """指定死因类别相似度计算器"""
    
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
        with h5py.File(h5_path, 'w') as h5_file:
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
                    
                    # 归一化特征
                    image_features = F.normalize(image_features, p=2, dim=1)
                    text_features = F.normalize(text_features, p=2, dim=1)
                    
                    # 确保特征维度正确
                    if image_features.shape[1] != 512:
                        if not hasattr(self, 'image_proj'):
                            self.image_proj = torch.nn.Linear(image_features.shape[1], 512).to(self.device)
                        image_features = self.image_proj(image_features)
                        image_features = F.normalize(image_features, p=2, dim=1)
                    
                    if text_features.shape[1] != 512:
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
            
            # 保存元数据
            h5_file.create_dataset('categories', data=categories, dtype=h5py.special_dtype(vlen=str))
            h5_file.create_dataset('locations', data=locations, dtype=h5py.special_dtype(vlen=str))
            h5_file.create_dataset('captions', data=captions, dtype=h5py.special_dtype(vlen=str))
            h5_file.create_dataset('image_paths', data=image_paths, dtype=h5py.special_dtype(vlen=str))
            
        print(f"特征提取完成，共 {total_samples} 个样本，保存到: {h5_path}")
        self.total_samples = total_samples
        
        # 强制关闭HDF5文件句柄
        import gc
        gc.collect()
        time.sleep(1)
        
        return self
    
    def calculate_specified_health_categories_similarity(self, 
                                                      specified_health_categories,
                                                      modality='image-image',
                                                      sample_size_per_category=200,
                                                      sample_fraction=0.8,
                                                      location_condition='any'):
        """计算指定死因类别的相似度
        
        参数:
            specified_health_categories: 指定的死因类别列表，可以是ID、名称或"category_id"格式
            modality: 模态类型 ('image-image', 'image-text', 'text-text')
            sample_size_per_category: 每个类别的样本对数量
            sample_fraction: 采样比例
            location_condition: 部位条件 ('any', 'same', 'different')
        
        返回:
            指定死因类别的相似度结果字典
        """
        if not hasattr(self, 'h5_path') or self.h5_path is None:
            raise ValueError("请先调用 extract_features_to_disk 方法")
        
        print("\n开始计算指定死因类别相似度...")
        
        # 获取所有唯一的死因类别
        with h5py.File(self.h5_path, 'r') as h5_file:
            health_labels = h5_file['health_labels'][:]
            unique_health_categories = np.unique(health_labels)
            print(f"发现 {len(unique_health_categories)} 个死因类别")
        
        # 将指定的类别转换为ID
        target_category_ids = []
        for category_spec in specified_health_categories:
            category_id = self._convert_health_category_to_id(category_spec)
            if category_id is not None and category_id in unique_health_categories:
                target_category_ids.append(category_id)
            else:
                print(f"警告: 未找到死因类别 '{category_spec}'，跳过")
        
        if not target_category_ids:
            print("错误: 没有找到任何有效的死因类别")
            return {}
        
        print(f"将计算 {len(target_category_ids)} 个指定的死因类别:")
        for category_id in target_category_ids:
            category_name = self._get_health_category_name(category_id)
            print(f"  - {category_name} (ID: {category_id})")
        
        all_results = {}
        
        # 计算每个指定死因类别的相似度
        print("\n=== 计算死因类别相似度 ===")
        for health_category in tqdm(target_category_ids):
            health_name = self._get_health_category_name(health_category)
            print(f"\n处理死因类别: {health_name} (ID: {health_category})")
            
            # 相同死因类别的相似度
            same_health_result = self._calculate_category_specific_similarity(
                modality=modality,
                target_health_category=int(health_category),
                target_location_category=None,
                health_condition='same',
                location_condition=location_condition,
                sample_size=sample_size_per_category,
                sample_fraction=sample_fraction
            )
            
            if same_health_result:
                all_results[f"health_{health_category}_same"] = same_health_result
            
            # 不同死因类别的相似度
            different_health_result = self._calculate_category_specific_similarity(
                modality=modality,
                target_health_category=int(health_category),
                target_location_category=None,
                health_condition='different',
                location_condition=location_condition,
                sample_size=sample_size_per_category,
                sample_fraction=sample_fraction
            )
            
            if different_health_result:
                all_results[f"health_{health_category}_different"] = different_health_result
        
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
            print(f"警告: 没有找到符合条件的样本对 - 健康类别: {target_health_category}, 健康条件: {health_condition}, 部位条件: {location_condition}")
            return None
        
        # 如果生成的样本对数量少于要求的sample_size，使用警告信息
        if len(pairs) < sample_size:
            print(f"注意: 只生成了 {len(pairs)} 个样本对 (要求 {sample_size}) - 健康类别: {target_health_category}, 健康条件: {health_condition}, 部位条件: {location_condition}")
        
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
        max_attempts = sample_size * 100
        
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
    
    def _convert_health_category_to_id(self, category_spec):
        """将死因类别规范（可能是名称或ID）转换为类别ID"""
        # 如果已经是整数，直接返回
        if isinstance(category_spec, int):
            return category_spec
        
        # 如果是字符串，尝试通过配置查找
        if isinstance(category_spec, str):
            # 查找死因类别
            if self.config and 'category' in self.config:
                for key, value in self.config['category'].items():
                    if value == category_spec:
                        # 配置中的key是从1开始的，需要减1得到ID
                        return int(key) - 1
            
            # 检查是否为"health_X"格式
            if category_spec.startswith('health_'):
                try:
                    return int(category_spec.split('_')[1])
                except (IndexError, ValueError):
                    pass
            
            # 检查是否为"category_X"格式
            if category_spec.startswith('category_'):
                try:
                    return int(category_spec.split('_')[1])
                except (IndexError, ValueError):
                    pass
        
        # 尝试直接转换为整数
        try:
            return int(category_spec)
        except (ValueError, TypeError):
            return None
    
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
    
    def check_category_sample_counts(self, max_retries=5, retry_delay=1):
        """检查每个死因类别的样本数量，添加重试机制"""
        if not hasattr(self, 'h5_path') or self.h5_path is None:
            print("请先提取特征到磁盘")
            return {}
        
        for attempt in range(max_retries):
            try:
                with h5py.File(self.h5_path, 'r') as h5_file:
                    health_labels = h5_file['health_labels'][:]
                    
                    # 统计健康类别样本数量
                    unique_health, health_counts = np.unique(health_labels, return_counts=True)
                    print("\n=== 死因类别样本统计 ===")
                    category_counts = {}
                    for category_id, count in zip(unique_health, health_counts):
                        category_name = self._get_health_category_name(category_id)
                        category_counts[category_id] = {
                            'name': category_name,
                            'count': int(count)
                        }
                        print(f"{category_name} (ID: {category_id}): {count} 个样本")
                
                return category_counts
                
            except (BlockingIOError, OSError) as e:
                print(f"尝试 {attempt + 1}/{max_retries} 打开HDF5文件失败: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    print("达到最大重试次数，跳过样本统计")
                    return {}
    
    def list_all_health_categories(self):
        """列出所有可用的死因类别"""
        category_counts = self.check_category_sample_counts()
        if not category_counts:
            return []
        
        categories_info = []
        for category_id, info in category_counts.items():
            categories_info.append({
                'id': int(category_id),
                'name': info['name'],
                'count': info['count']
            })
        
        # 按样本数量降序排列
        categories_info.sort(key=lambda x: x['count'], reverse=True)
        
        print("\n=== 所有死因类别列表（按样本数降序） ===")
        for i, info in enumerate(categories_info):
            print(f"{i+1:3d}. {info['name']:20s} (ID: {info['id']:3d}): {info['count']:5d} 个样本")
        
        return categories_info

# 数据集类
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

# 可视化函数
def visualize_specified_health_categories(all_results, calculator, save_dir="./specified_health_results"):
    """可视化指定死因类别的相似度结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取死因类别结果
    category_results = {}
    for key, result in all_results.items():
        if key.startswith('health_'):
            category_results[key] = result
    
    if not category_results:
        print("没有找到死因类别结果")
        return
    
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
    
    # 准备绘图数据
    category_labels = []
    same_means = []
    different_means = []
    same_stds = []
    different_stds = []
    
    for category_name, results in category_groups.items():
        if results['same'] and results['different']:
            category_labels.append(category_name)
            same_means.append(results['same']['mean_similarity'])
            different_means.append(results['different']['mean_similarity'])
            same_stds.append(results['same']['std_similarity'])
            different_stds.append(results['different']['std_similarity'])
    
    if not category_labels:
        print("没有足够的数据进行可视化")
        return
    
    # 创建分组柱状图
    plt.figure(figsize=(max(10, len(category_labels)), 8))
    
    x_pos = np.arange(len(category_labels))
    bar_width = 0.35
    
    # 绘制相同类别的柱子
    same_bars = plt.bar(x_pos - bar_width/2, same_means, bar_width, 
                        yerr=same_stds, color='skyblue', alpha=0.7, 
                        capsize=5, error_kw={'elinewidth': 1, 'markeredgewidth': 1},
                        label='Same Disease')
    
    # 绘制不同类别的柱子
    different_bars = plt.bar(x_pos + bar_width/2, different_means, bar_width,
                            yerr=different_stds, color='lightcoral', alpha=0.7,
                            capsize=5, error_kw={'elinewidth': 1, 'markeredgewidth': 1},
                            label='Different Disease')
    
    # 在柱子上添加数值
    for bar in same_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in different_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.ylabel('Mean Similarity Value')
    plt.title(f'Specified Disease Categories Similarity Comparison (Total: {len(category_labels)} categories)')
    plt.xticks(x_pos, category_labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, "specified_disease_categories_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"指定死因类别相似度比较图已保存: {save_path}")
    
    # 保存详细结果到CSV
    save_detailed_results(all_results, calculator, save_dir)

def save_detailed_results(all_results, calculator, save_dir):
    """保存详细结果到CSV"""
    results_data = []
    
    for cond_name, result in all_results.items():
        config = result['condition_config']
        
        # 添加类别名称信息
        health_category_name = None
        if 'target_health_category' in config and config['target_health_category'] is not None:
            health_category_name = calculator._get_health_category_name(config['target_health_category'])
        
        results_data.append({
            'condition': cond_name,
            'modality': config['modality'],
            'health_condition': config['health_condition'],
            'location_condition': config['location_condition'],
            'health_category_name': health_category_name,
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
    csv_path = os.path.join(save_dir, "specified_categories_similarity_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"详细结果已保存到: {csv_path}")

# 主函数
def main_specified_health_categories():
    """计算指定死因类别相似度的主函数"""
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
    
    # 创建数据加载器
    dataloader = DataLoader(
        medical_dataset, 
        batch_size=1024,
        shuffle=False,
        num_workers=2
    )
    
    # 创建相似度计算器
    print("\nLoading model from checkpoint...")
    try:
        calculator = SpecifiedHealthCategorySimilarityCalculator.from_checkpoint(
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
    SAVE_DIR = os.path.join(SAVE_ROOT, f"specified_health_categories_{timestamp}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. 提取特征到磁盘
    h5_path = os.path.join(SAVE_DIR, "features.h5")
    print("\n开始提取特征到磁盘...")
    
    # 限制样本数量
    max_samples = 1000000
    calculator.extract_features_to_disk(dataloader, h5_path, max_samples=max_samples)
    
    # 2. 列出所有可用的死因类别
    print("\n列出所有可用的死因类别...")
    all_categories = calculator.list_all_health_categories()
    
    # 3. 指定要计算的死因类别
    # 这里有三种方式指定类别：
    # 方式1: 直接指定类别名称列表
    specified_categories = [
        "Healthy",           # 类别名称
        # "health_10",       # 方式2: 使用health_ID格式
        # 15,                # 方式3: 直接使用类别ID
    ]
    
    # 方式4: 选择样本数量最多的前N个类别
    # top_n = 10
    # specified_categories = [cat['name'] for cat in all_categories[:top_n]]
    
    print(f"\n指定的死因类别: {specified_categories}")
    
    # 4. 计算指定死因类别的相似度
    print(f"\n开始计算指定死因类别相似度...")
    all_results = calculator.calculate_specified_health_categories_similarity(
        specified_health_categories=specified_categories,
        modality='image-image',  # 可以改为 'image-text' 或 'text-text'
        sample_size_per_category=200,
        sample_fraction=0.8,
        location_condition='any'  # 可以改为 'same' 或 'different'
    )
    
    # 5. 可视化结果
    if all_results:
        print(f"\n生成可视化结果...")
        print(f"共计算了 {len(all_results)} 个条件")
        
        # 生成柱状图比较
        visualize_specified_health_categories(all_results, calculator, save_dir=SAVE_DIR)
        
        # 保存详细结果到JSON
        detailed_results_path = os.path.join(SAVE_DIR, "detailed_similarity_results.json")
        serializable_results = {}
        for cond_name, result in all_results.items():
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
            
            config = result['condition_config']
            if 'target_health_category' in config and config['target_health_category'] is not None:
                serializable_result['health_category_name'] = calculator._get_health_category_name(
                    config['target_health_category']
                )
            
            serializable_results[cond_name] = serializable_result
        
        with open(detailed_results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n分析完成！所有结果已保存到: {SAVE_DIR}")
        
        # 打印汇总统计
        print("\n=== 汇总统计 ===")
        same_results = [r for k, r in all_results.items() if k.endswith('_same')]
        different_results = [r for k, r in all_results.items() if k.endswith('_different')]
        
        if same_results:
            same_means = [r['mean_similarity'] for r in same_results]
            print(f"相同死因类别平均相似度: {np.mean(same_means):.4f} ± {np.std(same_means):.4f}")
            print(f"相同类别样本对总数: {sum([r['num_pairs'] for r in same_results])}")
        
        if different_results:
            different_means = [r['mean_similarity'] for r in different_results]
            print(f"不同死因类别平均相似度: {np.mean(different_means):.4f} ± {np.std(different_means):.4f}")
            print(f"不同类别样本对总数: {sum([r['num_pairs'] for r in different_results])}")
    
    return all_results

# 交互式主程序
if __name__ == "__main__":
    print("=" * 60)
    print("指定死因类别相似度计算器")
    print("=" * 60)
    
    # 运行主函数
    results = main_specified_health_categories()
    
    if results:
        print("\n" + "=" * 60)
        print("计算完成！")
        print("=" * 60)
    else:
        print("\n计算失败，请检查错误信息。")