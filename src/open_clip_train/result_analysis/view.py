import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
from PIL import Image
import pandas as pd
import logging
from torch.utils.data import Dataset
import random
import open_clip

class MedicalCLIPVisualizer:
    def __init__(self, model, preprocess, tokenizer, config=None):
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.config = config
        self.model.eval()
        
    @classmethod
    def from_checkpoint(cls, checkpoint_path, model_name, config=None):
        """从检查点创建可视化器"""
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=None
        )
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 适配可能的键名变化
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
        tokenizer = open_clip.get_tokenizer(model_name)
        return cls(model, preprocess, tokenizer, config)
    
    def get_similarity(self, image_tensor, text_tensor):
        """计算图像和文本的相似度"""
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor.unsqueeze(0))
            text_features = self.model.encode_text(text_tensor.unsqueeze(0))
            
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            similarity = (image_features @ text_features.T).squeeze()
            
        return similarity.item()
    
    def generate_saliency_map(self, image_tensor, text_tensor):
        """生成显著图"""
        image_tensor = image_tensor.unsqueeze(0).requires_grad_()
        text_tensor = text_tensor.unsqueeze(0).requires_grad_()
        
        image_features = self.model.encode_image(image_tensor)
        text_features = self.model.encode_text(text_tensor)
        
        similarity = F.cosine_similarity(image_features, text_features)
        similarity.backward()
        
        saliency = image_tensor.grad.data.abs()
        saliency = saliency.squeeze().mean(dim=0)
        
        return saliency.detach().cpu().numpy()
    
    def visualize_medical_sample(self, dataset, idx, method='saliency'):
        """可视化医学影像样本"""
        image_tensor, text_tensor, health_label, location_label = dataset[idx]
        
        # 获取样本信息
        text = dataset.captions[idx]
        category = dataset.category[idx]
        location = dataset.location[idx]
        
        # 计算相似度
        similarity = self.get_similarity(image_tensor, text_tensor)
        
        # 生成热力图
        heatmap = self.generate_saliency_map(image_tensor, text_tensor)
        
        # 可视化
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 反预处理图像
        image_np = self.denormalize(image_tensor).cpu().numpy().transpose(1, 2, 0)
        image_np = np.clip(image_np, 0, 1)
        
        # 调整热力图大小
        heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
        
        # 原图
        axes[0].imshow(image_np, cmap='gray')
        axes[0].set_title(f'Original CT Image\nLocation: {location}')
        axes[0].axis('off')
        
        # 热力图
        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title(f'Saliency Map\nSimilarity: {similarity:.3f}')
        axes[1].axis('off')
        
        # 叠加图
        axes[2].imshow(image_np, cmap='gray')
        axes[2].imshow(heatmap_resized, cmap='jet', alpha=0.5)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        # 文本信息
        axes[3].axis('off')
        info_text = f"Diagnosis: {category}\n\nLocation: {location}\n\nSimilarity: {similarity:.3f}\n\nDescription: {text[:100]}..."
        axes[3].text(0.1, 0.9, info_text, transform=axes[3].transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(f'Medical Image Analysis - {category}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return similarity, heatmap
    
    def compare_diagnosis_categories(self, dataset, diagnosis_categories, num_samples=2):
        """比较不同诊断类别的注意力模式"""
        fig, axes = plt.subplots(len(diagnosis_categories), num_samples*2, 
                                figsize=(6*num_samples*2, 5*len(diagnosis_categories)))
        
        if len(diagnosis_categories) == 1:
            axes = axes.reshape(1, -1)
        
        for i, diagnosis in enumerate(diagnosis_categories):
            # 获取该诊断类别的样本
            diagnosis_indices = [idx for idx in range(len(dataset)) 
                               if dataset.category[idx] == diagnosis][:num_samples]
            
            for j, idx in enumerate(diagnosis_indices):
                image_tensor, text_tensor, health_label, location_label = dataset[idx]
                location = dataset.location[idx]
                
                # 生成热力图
                heatmap = self.generate_saliency_map(image_tensor, text_tensor)
                
                # 反预处理图像
                image_np = self.denormalize(image_tensor).cpu().numpy().transpose(1, 2, 0)
                image_np = np.clip(image_np, 0, 1)
                
                # 调整热力图大小
                heatmap_resized = cv.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
                heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
                
                # 显示原图和热力图
                col_idx = j * 2
                axes[i, col_idx].imshow(image_np, cmap='gray')
                axes[i, col_idx].set_title(f'{diagnosis}\nLocation: {location}')
                axes[i, col_idx].axis('off')
                
                axes[i, col_idx+1].imshow(image_np, cmap='gray')
                axes[i, col_idx+1].imshow(heatmap_resized, cmap='jet', alpha=0.5)
                axes[i, col_idx+1].set_title(f'{diagnosis}\nAttention Map')
                axes[i, col_idx+1].axis('off')
        
        plt.suptitle('Comparison of Attention Patterns by Diagnosis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def analyze_body_locations(self, dataset, body_locations, num_samples=3):
        """分析不同身体部位的注意力模式"""
        fig, axes = plt.subplots(len(body_locations), num_samples*2, 
                                figsize=(6*num_samples*2, 5*len(body_locations)))
        
        if len(body_locations) == 1:
            axes = axes.reshape(1, -1)
        
        for i, location in enumerate(body_locations):
            # 获取该部位的样本
            location_indices = [idx for idx in range(len(dataset)) 
                              if dataset.location[idx] == location][:num_samples]
            
            for j, idx in enumerate(location_indices):
                image_tensor, text_tensor, health_label, location_label = dataset[idx]
                diagnosis = dataset.category[idx]
                
                # 生成热力图
                heatmap = self.generate_saliency_map(image_tensor, text_tensor)
                
                # 反预处理图像
                image_np = self.denormalize(image_tensor).cpu().numpy().transpose(1, 2, 0)
                image_np = np.clip(image_np, 0, 1)
                
                # 调整热力图大小
                heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
                heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
                
                # 显示
                col_idx = j * 2
                axes[i, col_idx].imshow(image_np, cmap='gray')
                axes[i, col_idx].set_title(f'Location: {location}\nDiagnosis: {diagnosis}')
                axes[i, col_idx].axis('off')
                
                axes[i, col_idx+1].imshow(image_np, cmap='gray')
                axes[i, col_idx+1].imshow(heatmap_resized, cmap='jet', alpha=0.5)
                axes[i, col_idx+1].set_title(f'Attention Focus')
                axes[i, col_idx+1].axis('off')
        
        plt.suptitle('Attention Analysis by Body Location', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def generate_diagnosis_report(self, dataset, indices):
        """生成诊断报告可视化"""
        n_samples = len(indices)
        fig, axes = plt.subplots(n_samples, 3, figsize=(18, 6*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            image_tensor, text_tensor, health_label, location_label = dataset[idx]
            text = dataset.captions[idx]
            diagnosis = dataset.category[idx]
            location = dataset.location[idx]
            
            similarity = self.get_similarity(image_tensor, text_tensor)
            heatmap = self.generate_saliency_map(image_tensor, text_tensor)
            
            # 反预处理图像
            image_np = self.denormalize(image_tensor).cpu().numpy().transpose(1, 2, 0)
            image_np = np.clip(image_np, 0, 1)
            
            # 调整热力图大小
            heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
            heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
            
            # CT原图
            axes[i, 0].imshow(image_np, cmap='gray')
            axes[i, 0].set_title(f'CT Scan - Sample {idx}', fontsize=12)
            axes[i, 0].axis('off')
            
            # 注意力叠加
            axes[i, 1].imshow(image_np, cmap='gray')
            axes[i, 1].imshow(heatmap_resized, cmap='jet', alpha=0.6)
            axes[i, 1].set_title(f'Attention Map\nSimilarity: {similarity:.3f}', fontsize=12)
            axes[i, 1].axis('off')
            
            # 诊断报告
            axes[i, 2].axis('off')
            report_text = f"""
            MEDICAL REPORT
            
            Diagnosis: {diagnosis}
            Location: {location}
            Confidence Score: {similarity:.3f}
            
            Clinical Findings:
            {text}
            
            Attention Analysis:
            Model focuses on relevant anatomical regions
            corresponding to the diagnosed condition.
            """
            axes[i, 2].text(0.05, 0.95, report_text, transform=axes[i, 2].transAxes, 
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Medical Diagnosis Attention Analysis Report', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def denormalize(self, tensor):
        """反归一化图像张量"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return tensor * std + mean

# 适配您的医学数据加载方式
class MedicalCsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, disease_category, disease_location, config, sep="\t", tokenizer=None):
        logging.debug(f'Loading medical csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        self.disease_category = df[disease_category].tolist()
        self.disease_location = df[disease_location].tolist()
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
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
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
        
        return images, texts, health_label, location_label

# 使用示例
if __name__ == "__main__":
    # 配置您的医学类别和位置
    medical_config = {
        'category': {
            "1":"Healthy", "2":"Hemopericardium", "3":"Putrefactive gas bubbles",
            "4":"Periaortic hematoma", "5":"Intimal tear", "6":"Gastric content reflux",
            "7":"Blood sedimentation", "8":"Hypostatic pulmonary edema", "9":"Visceral autolysis",
            "10":"Skin burns", "11":"Tissue necrosis", "12":"Airway burns",
            "13":"Pulmonary edema", "14":"Laryngeal edema", "15":"Cerebral edema",
            "16":"Airway foreign body", "17":"Coronary artery occlusion", "18":"Intracranial hemorrhage",
            "19":"Soft tissue hematoma", "20":"Fractures", "21":"Internal organ injury",
            "22":"Brain herniation", "23":"Hyperdense lesion in brain parenchyma", "24":"Ballistic hemorrhage",
            "25":"Visceral perforation", "26":"Sinus fluid", "27":"Airway and digestive tract fluid",
            "28":"Internal bleeding", "29":"Hemopneumothorax", "30":"Hyoid bone fracture",
            "31":"Neck hematoma", "32":"High cervical fracture-dislocation", "33":"Skull fracture",
            "34":"Aortic rupture", "35":"Fracture along the wound track", "36":"Incised wounds",
            "37":"Pericardial effusion", "38":"Aortic dissection", "39":"Coronary artery calcification",
            "40":"Pulmonary embolism"
        },
        'location': {
            "1":"Lower Leg", "2":"Chest", "3":"Abdomen", "4":"Unknown",
            "5":"Upper Leg", "6":"Head", "7":"Neck", "8":"Pelvis"
        }
    }
    
    # 创建数据加载器
    from torchvision import transforms
    
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    # 创建医学数据集
    medical_dataset = MedicalCsvDataset(
        input_filename="your_medical_dataset.csv",
        transforms=preprocess,
        img_key="image_path",
        caption_key="caption",
        disease_category="disease_category",
        disease_location="disease_location",
        config=medical_config,
        tokenizer=tokenizer
    )
    
    # 创建医学可视化器
    visualizer = MedicalCLIPVisualizer.from_checkpoint(
        checkpoint_path="/root/autodl-fs/clip_log/2025_10_17-00_47_54-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_200-j_4-p_amp/checkpoints/epoch_latest.pt",
        model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        config=medical_config
    )
    
    # 1. 可视化单个医学样本
    print("Visualizing single medical sample...")
    visualizer.visualize_medical_sample(medical_dataset, idx=0)
    
    # 2. 比较不同诊断类别
    print("Comparing diagnosis categories...")
    selected_diagnoses = ["Intracranial hemorrhage", "Pulmonary embolism", "Fractures"]
    visualizer.compare_diagnosis_categories(medical_dataset, selected_diagnoses, num_samples=10)
    
    # 3. 分析不同身体部位
    print("Analyzing body locations...")
    selected_locations = ["Head", "Chest", "Abdomen"]
    visualizer.analyze_body_locations(medical_dataset, selected_locations, num_samples=10)
    
    # 4. 生成诊断报告
    print("Generating diagnosis reports...")
    visualizer.generate_diagnosis_report(medical_dataset, indices=[0, 1, 2])