""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from functools import partial

from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer,\
    text_global_pool
from .utils import to_2tuple


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    timm_model_name: Optional[str] = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = False
    proj_type: str = 'linear'  # control final text projection, 'none' forces no projection
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_type=text_cfg.proj_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None
        self.visual.lock(unlocked_groups=0, freeze_bn_stats=True)  # lock vision tower
        #self.text.lock(unlocked_layers=3, freeze_layer_norm=True)  # lock text tower
        # self.visual.apply_lora()  # apply LoRA to vision tower
        # self.text.apply_lora()    # apply LoRA to text tower

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()

class ImplicitCategoryMiner(nn.Module):
    def __init__(self, input_dim: int, num_components: int = 32):
        """
        隐式类别挖掘模块 - 使用PCA进行特征重建
        
        Args:
            input_dim: 输入特征维度
            num_components: PCA主成分数量
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_components = num_components
        
        # 注册PCA参数缓冲区（不需要梯度）
        self.register_buffer('eigenvectors', torch.zeros(num_components, input_dim))
        self.register_buffer('mean_vector', torch.zeros(input_dim))
        
    def compute_pca(self, features: torch.Tensor):
        """
        计算PCA参数（特征向量和均值）
        
        Args:
            features: 输入特征张量 [batch_size, input_dim]
        
        Returns:
            top_eigenvectors: 顶部特征向量 [num_components, input_dim]
            mean_vector: 均值向量 [input_dim]
        """
        # 1. 中心化数据
        mean_vector = torch.mean(features, dim=0)
        centered_data = features - mean_vector
        
        # 2. 计算协方差矩阵
        cov_matrix = torch.matmul(centered_data.T, centered_data) / (centered_data.size(0) - 1)
        
        # 确保对称性
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        # 3. 使用SVD分解
        U, S, Vh = torch.linalg.svd(cov_matrix.double())
        top_eigenvectors = U[:, :self.num_components].T
        
        # 转换回float32
        return top_eigenvectors.float().to(features.device), mean_vector.to(features.device)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        基于PCA的特征重建
        
        Args:
            features: 输入特征张量 [batch_size, input_dim]
        
        Returns:
            重建后的特征张量 [batch_size, input_dim]
        """
        # 中心化数据
        centered_data = features - self.mean_vector
        
        # 投影到主成分空间
        projected = torch.matmul(centered_data, self.eigenvectors.T)
        
        # 重建特征
        reconstructed = torch.matmul(projected, self.eigenvectors) + self.mean_vector
        # 混合原始特征和重建特征，减少突变
        # alpha = 0.5  # 保留70%原始特征
        # reconstructed = alpha * reconstructed + (1 - alpha) * features
        
        return reconstructed
    
#监督对比学习
class CMCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = 10.0,
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        
        self.teacher_text = copy.deepcopy(self.text)
        self.teacher_image = copy.deepcopy(self.visual)
        
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

        # 隐式类别挖掘
        # 隐式类别挖掘超参数
        # 初始化数值，后续训练不起作用
        self.category_1=8
        self.category_2=16
        with torch.no_grad():
            self.implicit_miner = ImplicitCategoryMiner(
                input_dim=embed_dim,
                num_components=8
            )
        
        # 注册聚类中心缓冲区
        self.register_buffer('implicit_centers_0', torch.zeros(self.category_1, embed_dim, device="cuda"))
        self.register_buffer('implicit_centers_1', torch.zeros(self.category_2, embed_dim,device="cuda"))

        self.current_epoch = -1
        
        #freeze tracher model parameter
        for param in self.teacher_text.parameters():
            param.requires_grad = False        
        for param in self.teacher_image.parameters():
            param.requires_grad = False
            
    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features
    
    @torch.no_grad()
    def teacher_encode_image(self, image, normalize: bool = False):
        with torch.no_grad():
            features = self.teacher_image(image)
            return F.normalize(features, dim=-1) if normalize else features
    
    @torch.no_grad()
    def teacher_encode_text(self, text, normalize: bool = False):
        with torch.no_grad():
            features = self.teacher_text(text)
            return F.normalize(features, dim=-1) if normalize else features
    
    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):

        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        
        teacher_text_features = self.teacher_encode_text(text, normalize=True) if text is not None else None
        teacher_image_features = self.teacher_encode_image(image, normalize=True) if image is not None else None
        
        if (text is not None and image is not None) and self.current_epoch>-1:
            # 隐式类别获取
            with torch.no_grad():
                reconstructed_feature = self.implicit_miner(text_features.detach())

                sim_matrix_0 = torch.mm(
                    F.normalize(reconstructed_feature, p=2, dim=-1),
                    F.normalize(self.implicit_centers_0, p=2, dim=-1).T
                )
                cluster_probs_0 = F.softmax(sim_matrix_0 / 0.1, dim=-1)
                implicit_category_0 = cluster_probs_0.detach()
                sim_matrix_1 = torch.mm(
                    F.normalize(reconstructed_feature, p=2, dim=-1),
                    F.normalize(self.implicit_centers_1, p=2, dim=-1).T
                )
                cluster_probs_1 = F.softmax(sim_matrix_1 / 0.1, dim=-1)
                implicit_category_1 = cluster_probs_1.detach()
                # implicit_categories = [implicit_category_0, implicit_category_1]
        else:
            implicit_category_0 = None
            implicit_category_1 = None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "teacher_text_features": teacher_text_features,
                "teacher_image_features": teacher_image_features,
                "implicit_category_0": implicit_category_0,
                "implicit_category_1": implicit_category_1,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()
    
    def kmeans_plusplus_init(self, features, k):
        """
        K-means++初始化聚类中心
        """
        indices = [torch.randint(0, len(features), (1,)).item()]
        
        for _ in range(1, k):
            dist = torch.cdist(features, features[indices])
            min_dist = dist.min(dim=1)[0]**2
            min_dist = torch.clamp(min_dist, min=1e-10)
            prob = min_dist / (torch.sum(min_dist) + 1e-8)
            indices.append(torch.multinomial(prob, 1).item())
        
        return indices
            
    @torch.no_grad()
    def dynamic_kmeans(self, features, explicit_labels, min_samples_per_cluster=400, max_iters=10000, stability_threshold=0.01):
        """
        动态K-means聚类，自动确定最佳聚类数量并确保类别均衡
        
        Args:
            features: 输入特征 [B, D]
            explicit_labels: 显式类别标签 [B, C] (one-hot编码)
            max_k: 最大聚类数量
            min_samples_per_cluster: 每个聚类的最小样本数
            max_iters: 最大迭代次数
            stability_threshold: 稳定性阈值
            
        Returns:
            cluster_probs: 聚类概率分布 [B, k]
            centers: 聚类中心 [k, D]
            optimal_k: 确定的优化聚类数量
        """
        # 归一化特征
        features = F.normalize(features, p=2, dim=1)
        batch_size, embed_dim = features.shape
        # 计算每个显式类别的样本数量
        class_counts = explicit_labels.sum(dim=0)
        # 确定初始聚类数量
        # 确保每个显式类别至少有min_samples_per_cluster个样本在每个聚类中
        min_k = max(1, int(torch.max(class_counts) / min_samples_per_cluster))

        # 使用肘部法则确定最佳k值
        optimal_k = self.find_optimal_k_elbow(features, min_k)
        # 使用最佳k值运行K-means
        cluster_probs, centers = self.run_kmeans(features, explicit_labels, optimal_k, max_iters, stability_threshold)
        
        return cluster_probs, centers, optimal_k
    
    @torch.no_grad()
    def find_optimal_k_elbow(self, features, min_k, num_trials=3):
        """
        使用肘部法则确定最佳聚类数量
        """
        distortions = []
        k_values = list(range(min_k, min_k + 32,2))  # 只尝试少量k值
        
        for k in k_values:
            # 多次运行取平均以减少随机性
            total_distortion = 0
            for _ in range(num_trials):
                indices = self.kmeans_plusplus_init(features, k)
                centers = features[indices].clone()
                
                # 运行少量迭代计算失真度
                for _ in range(10):
                    sim_matrix = torch.mm(features, centers.t())
                    assignments = sim_matrix.argmax(dim=1)
                    
                    # 更新聚类中心
                    new_centers = centers.clone()
                    for j in range(k):
                        mask = (assignments == j)
                        if mask.any():
                            new_centers[j] = features[mask].mean(dim=0)
                        else:
                            # 处理空簇
                            dist_to_centers = 1 - torch.mm(features, centers.t())
                            farthest_idx = dist_to_centers.sum(dim=1).argmax()
                            new_centers[j] = features[farthest_idx]
                    
                    centers = new_centers
                
                # 计算失真度（到最近聚类中心的距离之和）
                sim_matrix = torch.mm(features, centers.t())
                max_sim = sim_matrix.max(dim=1)[0]
                distortion = (1 - max_sim).sum().item()
                total_distortion += distortion
            
            distortions.append(total_distortion / num_trials)
        
        # 计算失真度的二阶差分找到肘点
        if len(distortions) > 2:
            first_deriv = np.diff(distortions)
            second_deriv = np.diff(first_deriv)
            elbow_point = np.argmax(second_deriv) + 1  # 加1因为差分减少了一个元素
            optimal_k = k_values[elbow_point]
        else:
            optimal_k = k_values[0]
        
        return optimal_k 
    
    
    @torch.no_grad()
    def run_kmeans(self, features, explicit_labels, k, max_iters, stability_threshold):
        """
        运行K-means聚类
        """
        # K-means++初始化
        indices = self.kmeans_plusplus_init(features, k)
        centers = features[indices].clone()
        
        for _ in range(max_iters):
            # 计算余弦相似度
            sim_matrix = torch.mm(features, centers.t())
            assignments = sim_matrix.argmax(dim=1)
            
            # 更新聚类中心
            new_centers = centers.clone()
            cluster_changed = False
            
            for j in range(k):
                mask = (assignments == j)
                
                if not mask.any():
                    # 处理空簇
                    dist_to_centers = 1 - torch.mm(features, centers.t())
                    farthest_idx = dist_to_centers.sum(dim=1).argmax()
                    new_centers[j] = features[farthest_idx]
                    cluster_changed = True
                    continue
                
                # 检查类别分布均衡性
                cluster_labels = explicit_labels[mask]
                class_distribution = cluster_labels.sum(dim=0) / mask.sum()
                
                # 如果某个类别占比过高，调整聚类中心
                max_class_ratio = torch.max(class_distribution)
                if max_class_ratio > 0.5:  # 如果某个类别占比超过70%
                    # 找到占比过高的类别
                    dominant_class = torch.argmax(class_distribution)
                    
                    # 从非主导类别中选择一个样本作为新中心
                    non_dominant_mask = (explicit_labels[:, dominant_class] == 0)
                    if non_dominant_mask.any():
                        non_dominant_features = features[non_dominant_mask]
                        # 选择距离当前中心最远的点
                        dist = torch.cdist(non_dominant_features, centers[j:j+1])
                        new_center_idx = dist.argmax()
                        new_centers[j] = non_dominant_features[new_center_idx]
                        cluster_changed = True
                
                # 标准聚类中心更新
                cluster_mean = features[mask].mean(dim=0)
                new_centers[j] = F.normalize(cluster_mean, p=2, dim=0)
            
            # 检查收敛
            center_similarity = F.cosine_similarity(new_centers, centers, dim=1)
            center_shift = (1 - center_similarity).mean()
            centers = new_centers
            if center_shift < stability_threshold:
                break
        
        # 计算最终相似度
        final_sim_matrix = torch.mm(features, centers.t())
        cluster_probs = F.softmax(final_sim_matrix / 0.1, dim=-1)
        
        return cluster_probs, centers
    def update_implicit_components(self, args, epoch, text, explicit_labels):
        """
        每个epoch更新PCA和聚类中心
        """
        if epoch == self.current_epoch:
            return
        
        self.current_epoch = epoch
        logging.info(f"正在更新隐式组件 (epoch {epoch})")
        # 分批计算特征
        features_list = []
        batch_size = 128  # 使用固定批次大小
        # 收集所有特征和显式标签  
        with torch.no_grad():
            for i in range(args.accum_freq):
                batch_texts = text[i]
                batch_features = self.encode_text(batch_texts, normalize=False)
                features_list.append(batch_features)

        features = torch.cat(features_list, dim=0)
        # 更新PCA
        eigenvectors, mean_vector = self.implicit_miner.compute_pca(features)
        self.implicit_miner.eigenvectors = eigenvectors.to(self.implicit_centers_0.device)
        self.implicit_miner.mean_vector = mean_vector.to(self.implicit_centers_0.device)
        
        # 计算重建特征
        reconstructed = self.implicit_miner(features)
        # 更新聚类中心（带类别平衡约束）
        cluster_probs_0, centers_0, optimal_k_0 = self.dynamic_kmeans(
            reconstructed, 
            explicit_labels, 
            min_samples_per_cluster=200
            )
        cluster_probs_1, centers_1, optimal_k_1 = self.dynamic_kmeans(
            reconstructed, 
            explicit_labels,  
            min_samples_per_cluster=100
            )
        logging.info(
            f"category_1 number:{optimal_k_0}"
            f"category_2 number:{optimal_k_1}"
        )
        self.category_1 = optimal_k_0  # 动态更新隐式类别数量
        self.category_2 = optimal_k_1
        device = centers_0.device
        # 调整缓冲区大小以匹配新的类别数量
        if self.implicit_centers_0.shape[0] != optimal_k_0:
            self.register_buffer('implicit_centers_0', torch.zeros(optimal_k_0, self.implicit_centers_0.shape[1],device=device))
        
        if self.implicit_centers_1.shape[0] != optimal_k_1:
            self.register_buffer('implicit_centers_1', torch.zeros(optimal_k_1, self.implicit_centers_1.shape[1],device=device))
            
        # 存储聚类中心
        if epoch == 0:
            self.implicit_centers_0 = centers_0.clone().detach()
            self.implicit_centers_1 = centers_1.clone().detach()
        else:
            # 平滑更新聚类中心
            alpha = 0.996
            self.implicit_centers_0 = alpha * self.implicit_centers_0 + (1 - alpha) * centers_0
            self.implicit_centers_1 = alpha * self.implicit_centers_1 + (1 - alpha) * centers_1
    
    def update_teacher(self, alpha=0.996):
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_text.parameters(), self.text.parameters()):
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
            for teacher_param, student_param in zip(self.teacher_image.parameters(), self.visual.parameters()):
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
    
def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)
    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def resize_text_pos_embed(state_dict, model, interpolation: str = 'linear', antialias: bool = False):
    old_pos_embed = state_dict.get('positional_embedding', None)
    if old_pos_embed is None:
        return
    # FIXME add support for text cls_token
    model_pos_embed = getattr(model, 'positional_embedding', None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, 'positional_embedding', None)

    old_num_pos = old_pos_embed.shape[0]
    old_width = old_pos_embed.shape[1]
    num_pos = model_pos_embed.shape[0]
    width = model_pos_embed.shape[1]
    assert old_width == width, 'text pos_embed width changed!'
    if old_num_pos == num_pos:
        return

    logging.info('Resizing text position embedding num_pos from %s to %s', old_num_pos, num_pos)
    old_pos_embed = old_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
    old_pos_embed = F.interpolate(
        old_pos_embed,
        size=num_pos,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    old_pos_embed = old_pos_embed.permute(0, 2, 1)[0]
    new_pos_embed = old_pos_embed

    state_dict['positional_embedding'] = new_pos_embed


def get_model_preprocess_cfg(model):
    module = getattr(model, 'visual', model)
    preprocess_cfg = getattr(module, 'preprocess_cfg', {})
    if not preprocess_cfg:
        # use separate legacy attributes if preprocess_cfg dict not found
        size = getattr(module, 'image_size')
        if size is not None:
            preprocess_cfg['size'] = size
        mean = getattr(module, 'image_mean', None)
        if mean is not None:
            preprocess_cfg['mean'] = mean
        std = getattr(module, 'image_std', None)
        if std is not None:
            preprocess_cfg['std'] = std
    return preprocess_cfg


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]):
    module = getattr(model, 'visual', model)
    module.image_mean = preprocess_cfg['mean']  # legacy attribute, keeping for bwd compat
    module.image_std = preprocess_cfg['std']  # legacy attribute, keeping for bwd compat
    module.preprocess_cfg = copy.deepcopy(preprocess_cfg)  # new attr, package all pp cfg as dict


def get_model_tokenize_cfg(model):
    module = getattr(model, 'text', model)
    cfg = {}
    context_length = getattr(module, 'context_length', None)
    if context_length is not None:
        cfg['context_length'] = context_length
    vocab_size = getattr(module, 'vocab_size', None)
    if vocab_size is not None:
        cfg['vocab_size'] = vocab_size
    return cfg

# 线性分类
class cls_CMCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = 10.0,
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.logit_bias = None
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.lock_image_tower(unlocked_groups=0, freeze_bn_stats=False)
        self.feature_norm = nn.LayerNorm(embed_dim)
        # 分类头：改为小型 MLP (Linear -> LayerNorm -> GELU -> Dropout -> Linear)
        cls_num = 40
        mlp_hidden = embed_dim * 4
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden, bias=False),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, cls_num, bias=True),
        )

        
    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = torch.tensor(0.0)
        normalized_features = self.feature_norm(image_features)
        cls_logits = self.cls_head(normalized_features)
        
        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "cls_logits": cls_logits,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, cls_logits, self.logit_bias
        return image_features, text_features, cls_logits
