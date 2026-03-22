# GCM-CLIP: 面向虚拟尸检的百万级多模态数据集与表征学习框架
[![Paper](https://img.shields.io/badge/Paper-ForVA%20%26%20GCM--CLIP-blue)](https://github.com/jkmmmm/GCM-CLIP)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0+-red)](https://pytorch.org/)

本仓库是论文 **ForVA and GCM-CLIP: A Million-Scale Multimodal Dataset and Representation Learning Framework for Virtual Autopsy** 的官方开源实现。

针对法医学虚拟尸检领域**专业多模态数据稀缺、细粒度语义对齐困难、模型易受死后伪影干扰发生灾难性捷径学习**的核心瓶颈，我们构建了**ForVA**——全球首个百万级标准化虚拟尸检多模态数据集，并提出了**GCM-CLIP**语义增强对比学习框架，实现了法医CT影像与病理报告的精准跨模态对齐，为虚拟尸检的智能化分析提供了标准化数据基础设施与可复现的建模范式。

---

## 项目核心亮点
1.  **百万级专业虚拟尸检数据集ForVA**
    - 包含**1,257,349对**标准化法医CT影像-病理报告图文对，覆盖9类核心死因、40种法医病变类型、8大解剖区域
    - 配套203,478对细粒度专家标注数据集，用于模型微调与评估，涵盖98种实例级病变亚型
    - 严格遵循法医学影像采集与重建规范，通过「大模型生成-自动化逻辑校验-法医专家人工审核」的闭环流程构建，消除语义幻觉，保障司法证据可靠性

2.  **语义增强的GCM-CLIP建模框架**
    - 核心创新**通用类别挖掘（General-Category Mining, GCM）机制**，作为高精度「语义滤波器」，自适应解耦层级化法医报告的细粒度病理特征
    - 包含**动态语义解耦（DSD）**与**自适应平衡聚类（ABC）**双模块，从报告中挖掘隐式细粒度病理属性，与显式语义锚点联合构建高纯度监督信号
    - 引入**梯度正交化策略（GOS）**，解决多任务监督的梯度冲突问题，显著提升训练稳定性与收敛效率

3.  **SOTA性能与强泛化能力**
    - 在ForVA基准测试中，零样本分类相对SOTA基线提升25%，跨模态检索绝对提升6-10%
    - 极端低数据场景下，仅用1%标注数据的线性探测性能，超过BiomedCLIP基线100%全监督训练的结果
    - 在MIMIC-CXR、ROCOv2、PMC-OA三大外部临床数据集上实现零样本跨域泛化，无需微调即可适配常规医学影像分析场景

4.  **法医临床落地价值**
    - 盲法对照试验证实，GCM-CLIP辅助诊断可使初级法医诊断准确率从42%提升至72%，超过无AI辅助的资深法医专家水平（69%）
    - 可作为无偏「第二阅片者」，识别因认知锚定效应被人工遗漏的隐匿性微病变，降低法医误诊风险，保障司法裁判的客观性

---

## ForVA 数据集介绍
ForVA（Forensic Virtual Autopsy）是目前全球规模最大、标注最规范的虚拟尸检多模态数据集，由中国法医学会鉴定中心提供标准化虚拟解剖影像，经7名资深法医专家全程质控构建。

### 数据集构成
| 数据集子集 | 规模 | 用途 | 标注粒度 |
|------------|------|------|----------|
| 预训练集 | 1,257,349 图文对 | 模型预训练 | 层级化病理报告（Discovery+Impression）、9类死因标签 |
| 微调评估集 | 203,478 图文对 | 线性探测、下游任务微调 | 8个解剖区域、40类病变、98种实例级亚型 |
| 域内测试集 | 4,096 图文对 | 零样本分类、跨模态检索评估 | 专家金标准标注，40类病变均衡分布 |
| 外部零样本测试集 | MIMIC-CXR(1024)、ROCOv2(4096)、PMC-OA(4096) | 跨域泛化能力评估 | 临床胸部CT影像-报告配对 |
| 下游任务数据集 | 目标检测(9230样本)、语义分割(900样本) | 密集预测任务验证 | 6类核心法医病变像素级/框级标注 |

### 数据采集与标注规范
- **影像采集**：严格遵循标准化CT扫描与重建协议，覆盖头颈部、胸腹部、盆部与下肢三大核心区域，详细参数见补充材料Table S1
- **标注流程**：采用「InternVL生成候选报告→DeepSeek-R1逻辑校验→法医专家人工审核」的三级闭环标注流程，同时保留阳性发现与阴性对照，缓解分布偏差
- **标签体系**：建立法医学层级化分类体系，包含「死因大类→解剖区域→病变类型→实例级亚型」四级语义结构，适配模型的显式-隐式多任务监督

### 数据集获取
本数据集仅用于学术研究，需签署数据使用协议。如需申请，请联系通讯作者 **Jing Cai (caijing@zjjcxy.cn)** 提交申请，审核通过后将提供数据访问权限。

---

## GCM-CLIP 核心框架
GCM-CLIP基于对比语言-图像预训练的双编码器架构，针对法医领域的层级化语义特征进行了专项优化，核心结构如下：

1.  **双编码器骨干**
    - 文本编码器：采用PubMedBERT，针对长文本法医报告优化，最大上下文窗口设置为256 tokens，完整保留病理描述细节
    - 图像编码器：采用高分辨率Vision Transformer (ViT)，引入Patch Dropout策略，提升高分辨率CT影像的处理效率与病变位置鲁棒性

2.  **GCM核心机制**
    - **动态语义解耦（DSD）模块**：通过动态PCA对文本特征进行协方差矩阵分解与特征正交化，量化每个语义维度的显著性权重，从冗余模板噪声中过滤出核心病理信号
    - **自适应平衡聚类（ABC）模块**：基于聚类畸变曲线的二阶导数自动确定最优聚类数，协同解耦特征与显式语义锚点，挖掘隐式病理语义簇，通过均衡约束避免隐式属性退化为显式标签的子类
    - **梯度正交化策略（GOS）**：将隐式任务梯度投影到显式主任务的正交子空间，消除梯度冲突分量，保留互补信息，提升多任务训练的稳定性

3.  **多任务监督对比学习**
    构建融合显式监督损失、隐式监督损失与全局跨模态对比损失的联合优化目标，实现显式解剖/病变标签与隐式病理属性的协同对齐，迫使模型聚焦于法医学核心病变特征，抑制死后伪影等无关噪声。

---

## 核心实验结果
### 零样本分类性能（ForVA基准测试集）
| 模型 | Dis_Loc@1 | Dis_Loc@5 | Dis@1 | Dis@5 | Loc@1 | Loc@5 |
|------|-----------|-----------|-------|-------|-------|-------|
| PubMedCLIP | 0.0175 | 0.1552 | 0.0856 | 0.2421 | 0.2614 | 0.4960 |
| BiomedCLIP | 0.0993 | 0.2519 | 0.2028 | 0.3981 | 0.5546 | 0.9187 |
| GLoRIA | 0.0769 | 0.2356 | 0.1530 | 0.2844 | 0.4328 | 0.5827 |
| MGCA | 0.0759 | 0.2788 | 0.1530 | 0.3505 | 0.3413 | 0.5507 |
| medsiglip | 0.0478 | 0.2458 | 0.1696 | 0.3442 | 0.2417 | 0.4006 |
| **GCM-CLIP (Ours)** | **0.1220** | **0.2858** | **0.2536** | **0.4169** | **0.6530** | **0.9502** |

### 跨模态检索性能
| 模型 | 图像到文本 | | | 文本到图像 | | |
|------|------------|------------|------------|------------|------------|------------|
| | R@1 | R@5 | R@10 | R@1 | R@5 | R@10 |
| PubMedCLIP | 0.0752 | 0.2763 | 0.4345 | 0.0905 | 0.2995 | 0.4440 |
| BiomedCLIP | 0.2431 | 0.5459 | 0.6823 | 0.2500 | 0.5407 | 0.6796 |
| medsiglip | 0.2731 | 0.5937 | 0.7487 | 0.2675 | 0.6032 | 0.7448 |
| **GCM-CLIP (Ours)** | **0.3230** | **0.6425** | **0.7675** | **0.3134** | **0.6413** | **0.7702** |

### 低数据场景线性探测性能（实例级未见病变任务）
| 模型 | 1% 标注数据 | 10% 标注数据 | 100% 标注数据 |
|------|--------------|---------------|----------------|
| PubMedCLIP | 0.5716 | 0.5826 | 0.5875 |
| BiomedCLIP | 0.4528 | 0.4770 | 0.4860 |
| MGCA | 0.5281 | 0.5575 | 0.6027 |
| medsiglip | 0.5271 | 0.5971 | 0.5980 |
| **GCM-CLIP (Ours)** | **0.5429** | **0.5849** | **0.6215** |

### 下游目标检测性能（冻结编码器）
| 病变类别 | BiomedCLIP | GCM-CLIP (Ours) | 相对提升 |
|----------|------------|------------------|----------|
| Sinus Fluid | 0.682 | 0.793 | +16.3% |
| Pulm Edema | 0.715 | 0.826 | +15.5% |
| HPTX | 0.604 | 0.758 | +25.5% |
| Gastric Reflux | 0.657 | 0.781 | +18.9% |
| Hyoid Fx | 0.000 | 0.785 | - |
| Airway/Digest Fluid | 0.628 | 0.715 | +13.9% |
| **Macro Avg** | **0.670** | **0.776** | **+15.8%** |

---

## 环境安装
### 1. 快速安装（推理/下游微调）
适用于仅使用预训练模型进行推理、下游任务微调的场景：
```bash
# 基础安装（仅推理）
pip install open_clip_torch

# 完整安装（含训练、评估全量依赖）
pip install 'open_clip_torch[training]'
```

### 2. 源码安装（二次开发/全流程训练）
适用于需要修改模型源码、复现论文预训练实验的场景：
```bash
# 克隆仓库到本地
git clone https://github.com/jkmmmm/GCM-CLIP.git
cd GCM-CLIP

# 创建并激活虚拟环境
python3 -m venv .env
# Linux/macOS
source .env/bin/activate
# Windows
# .env\Scripts\activate

# 升级pip
pip install -U pip

# 安装基础依赖
make install

# 安装PyTorch（根据你的CUDA版本选择对应命令，参考https://pytorch.org/get-started/locally/）
# 示例：CUDA 12.1 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装训练全量依赖
make install-training

# 安装测试依赖（可选）
make install-test
```

---

## 快速开始
### 零样本法医病变分类
```python
import torch
from PIL import Image
import open_clip

# 加载GCM-CLIP预训练模型、预处理与分词器
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name='ViT-B-32',
    pretrained='path/to/your/gcm_clip_checkpoint.pt'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# 加载并预处理虚拟尸检CT影像
image = preprocess(Image.open("virtual_autopsy_ct.png")).unsqueeze(0)

# 定义法医病变分类文本（适配ForVA数据集40类病变体系）
disease_classes = [
    "Healthy", "Intracranial hemorrhage", "Pulmonary edema",
    "Aortic dissection", "Fractures", "Soft tissue hematoma",
    "Hemopneumothorax", "Gastric content reflux", "Hyoid bone fracture"
]
text_prompts = [f"A forensic virtual autopsy CT image showing {cls}" for cls in disease_classes]
text = tokenizer(text_prompts)

# 推理计算
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # 特征归一化
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    # 计算分类概率
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# 输出分类结果
print("病变分类概率:")
for cls, prob in zip(disease_classes, text_probs[0]):
    print(f"{cls}: {prob:.4f}")
```

### 跨模态图文检索
```python
import torch
from PIL import Image
import open_clip
import numpy as np

# 加载模型
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name='ViT-L-14',
    pretrained='path/to/your/gcm_clip_checkpoint.pt'
)
tokenizer = open_clip.get_tokenizer('ViT-L-14')
model.eval()

# 构建法医报告检索库
report_database = [
    {"text": "Chest CT shows aortic dissection with intimal flap in the ascending aorta", "label": "Aortic dissection"},
    {"text": "Head CT reveals intracranial hemorrhage with mass effect and midline shift", "label": "Intracranial hemorrhage"},
    {"text": "Chest CT demonstrates diffuse pulmonary edema with ground-glass opacities in bilateral lungs", "label": "Pulmonary edema"},
]

# 预计算文本特征库
text_features_list = []
with torch.no_grad(), torch.cuda.amp.autocast():
    for item in report_database:
        text = tokenizer([item["text"]])
        feat = model.encode_text(text)
        feat /= feat.norm(dim=-1, keepdim=True)
        text_features_list.append(feat)
text_features_database = torch.cat(text_features_list, dim=0)

# 加载查询CT影像
query_image = preprocess(Image.open("query_ct.png")).unsqueeze(0)
with torch.no_grad(), torch.cuda.amp.autocast():
    image_feature = model.encode_image(query_image)
    image_feature /= image_feature.norm(dim=-1, keepdim=True)

# 计算相似度，检索Top-1匹配报告
similarity = image_feature @ text_features_database.T
top_idx = torch.argmax(similarity, dim=1).item()
print(f"最匹配的病理报告: {report_database[top_idx]['text']}")
print(f"匹配病变类型: {report_database[top_idx]['label']}")
print(f"相似度得分: {similarity[0, top_idx]:.4f}")
```

---

## 训练指南
### 1. 预训练GCM-CLIP
#### 单卡预训练
```bash
python -m open_clip_train.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/path/to/forva_pretrain.csv"  \
    --val-data="/path/to/forva_val.csv"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --csv-location-key location \
    --csv-disease-key disease \
    --imagenet-val=/path/to/forva_zero_shot/ \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=300 \
    --workers=8 \
    --model ViT-B-32 \
    --precision amp \
    --gcm-enabled \
    --dsd-components 32 \
    --abc-cluster-range 10 100 \
    --gos-enabled
```

#### 多卡分布式预训练
```bash
cd src
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --train-data '/path/to/forva_pretrain_shards/{0000..2175}.tar' \
    --train-num-samples 1257349 \
    --dataset-type webdataset \
    --batch-size 256 \
    --precision amp \
    --workers 4 \
    --model ViT-L-14 \
    --gcm-enabled \
    --gos-enabled \
    --accum-freq 8 \
    --log-every-n-steps 100 \
    --save-frequency 5
```

### 2. 下游任务微调
#### 线性探测
```bash
python -m linear_probe.main \
    --train-data="/path/to/forva_linear_train.csv" \
    --val-data="/path/to/forva_linear_val.csv" \
    --model ViT-B-32 \
    --pretrained="/path/to/gcm_clip_checkpoint.pt" \
    --batch-size 256 \
    --lr 1e-3 \
    --epochs 50 \
    --target disease \
    --freeze-image-encoder
```

#### 目标检测微调
```bash
python -m downstream.detection.train \
    --data-root /path/to/forva_detection_dataset/ \
    --model ViT-B-32 \
    --pretrained="/path/to/gcm_clip_checkpoint.pt" \
    --batch-size 16 \
    --lr 1e-4 \
    --epochs 30 \
    --freeze-backbone
```

---

## 测试与评估
### 零样本分类评估
```bash
python -m open_clip_train.main \
    --imagenet-val /path/to/forva_test_set/ \
    --model ViT-B-32 \
    --pretrained /path/to/gcm_clip_checkpoint.pt \
    --eval-only
```

### 跨模态检索评估
```bash
python -m evaluation.retrieval_eval \
    --test-data /path/to/forva_retrieval_test.csv \
    --model ViT-B-32 \
    --pretrained /path/to/gcm_clip_checkpoint.pt \
    --batch-size 64
```

### 全量单元测试
```bash
# 运行所有测试用例
make test

# 运行指定模块测试
python -m pytest -x -s -v tests -k "gcm"
```

---

## 伦理声明
本研究经中国法医学会鉴定中心研究伦理委员会审查批准，所有流程均符合中国法律法规、国际伦理指南与伦理委员会要求。所有研究数据均已进行去标识化处理，所有受试者（或其法定代理人）均已签署书面知情同意书，同意相关数据与研究结果的学术发表。

本模型仅用于法医学学术研究与辅助诊断场景，不可单独作为司法裁判的唯一依据，所有法医诊断结论必须由执业法医结合临床与现场信息综合判定。

---

## 引用
如果本项目与数据集对您的研究有帮助，请引用我们的论文：
```bibtex
@article{mao2026forva,
  title={ForVA and GCM-CLIP: A Million-Scale Multimodal Dataset and Representation Learning Framework for Virtual Autopsy},
  author={Mao, Jikai and Du, Nanze and Tu, Lyu and Li, Hao and Shen, Yi and Shen, Liang and Guo, Junjun and Cai, Jing},
  journal={},
  year={2026}
}

@software{ilharco_gabriel_2021_5143773,
  author        = {Ilharco, Gabriel and Wortsman, Mitchell and Wightman, Ross and Gordon, Cade and Carlini, Nicholas and Taori, Rohan and Dave, Achal and Shankar, Vaishaal and Namkoong, Hongseok and Miller, John and Hajishirzi, Hannaneh and Farhadi, Ali and Schmidt, Ludwig},
  title         = {OpenCLIP},
  month         = jul,
  year          = 2021,
  publisher     = {Zenodo},
  version       = {0.1},
  doi           = {10.5281/zenodo.5143773},
  url           = {https://doi.org/10.5281/zenodo.5143773}
}
```

---

## 致谢
本研究受国家重点研发计划（十四五）项目（2023YFC3303901）资助。感谢中国法医学会鉴定中心提供的专业数据支持，感谢所有法医专家对数据集标注与模型验证的专业指导，感谢OpenCLIP开源项目为本研究提供的基础框架支持。

---

## 许可证
本项目基于 [MIT LICENSE](LICENSE) 开源，仅可用于学术研究场景，严禁商用。