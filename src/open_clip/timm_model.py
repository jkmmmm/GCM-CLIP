""" timm model adapter

Wraps timm (https://github.com/rwightman/pytorch-image-models) models for use as a vision tower in CLIP model.
"""
import logging
from collections import OrderedDict

import torch
import torch.nn as nn

try:
    import timm
    try:
        # new timm imports >= 0.8.1
        from timm.layers import RotAttentionPool2d
        from timm.layers import AttentionPool2d as AbsAttentionPool2d
        from timm.layers import Mlp, to_2tuple
    except ImportError as e:
        # fallback, try old timm imports < 0.8.1
        from timm.models.layers.attention_pool2d import RotAttentionPool2d
        from timm.models.layers.attention_pool2d import AttentionPool2d as AbsAttentionPool2d
        from timm.models.layers import Mlp, to_2tuple
except ImportError:
    timm = None

from .utils import freeze_batch_norm_2d
from peft import LoraConfig, get_peft_model

class TimmModel(nn.Module):
    """ timm model adapter
    """

    def __init__(
            self,
            model_name,
            embed_dim,
            image_size=224,
            pool='avg',
            proj='linear',
            proj_bias=False,
            drop=0.,
            drop_path=None,
            patch_drop=None,
            pretrained=False,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("Please `pip install timm` to use timm models.")
        self.image_size = to_2tuple(image_size)

        # setup kwargs that may not be common across all models
        timm_kwargs = {}
        if drop_path is not None:
            timm_kwargs['drop_path_rate'] = drop_path
        if patch_drop is not None:
            timm_kwargs['patch_drop_rate'] = patch_drop

        custom_pool = pool in ('abs_attn', 'rot_attn')
        if proj:
            assert proj in ("linear", "mlp", "none")
        extra_proj = proj in ("linear", "mlp")
        if not extra_proj and not custom_pool:
            # use network classifier head as projection if no proj specified and no custom pooling used
            # if projection is explicitly set to "none" will be pass through from network trunk
            proj_dim = 0 if proj == 'none' else embed_dim
            self.trunk = timm.create_model(
                model_name,
                num_classes=proj_dim,
                global_pool=pool,
                pretrained=pretrained,
                **timm_kwargs,
            )
            prev_chs = embed_dim
        else:
            self.trunk = timm.create_model(
                model_name,
                pretrained=pretrained,
                **timm_kwargs,
            )
            feat_size = self.trunk.default_cfg.get('pool_size', None)
            feature_ndim = 1 if not feat_size else 2
            if custom_pool:
                assert feature_ndim == 2
                # if attn pooling used, remove both classifier and default pool
                self.trunk.reset_classifier(0, global_pool='')
            else:
                # reset global pool if pool config set, otherwise leave as network default
                reset_kwargs = dict(global_pool=pool) if pool else {}
                self.trunk.reset_classifier(0, **reset_kwargs)
            prev_chs = self.trunk.num_features

        head_layers = OrderedDict()

        # Add custom pooling to head
        if pool == 'abs_attn':
            head_layers['pool'] = AbsAttentionPool2d(prev_chs, feat_size=feat_size, out_features=embed_dim)
            prev_chs = embed_dim
        elif pool == 'rot_attn':
            head_layers['pool'] = RotAttentionPool2d(prev_chs, out_features=embed_dim)
            prev_chs = embed_dim

        # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
        if proj == 'linear':
            head_layers['drop'] = nn.Dropout(drop)
            head_layers['proj'] = nn.Linear(prev_chs, embed_dim, bias=proj_bias)
        elif proj == 'mlp':
            head_layers['mlp'] = Mlp(prev_chs, 2 * embed_dim, embed_dim, drop=(drop, 0), bias=(True, proj_bias))

        self.head = nn.Sequential(head_layers)
        # lora参数
        self.use_lora = True
        lora_r = 16
        lora_alpha = 32
        lora_dropout = 0.1
        lora_target_modules = None
        lora_bias = "none"
        self.lora_config = {
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'lora_target_modules': lora_target_modules,
            'lora_bias': lora_bias
        }
        self.output_tokens = False

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """ lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            # lock full model
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            # NOTE: partial freeze requires latest timm (master) branch and is subject to change
            try:
                # FIXME import here until API stable and in an official release
                from timm.models.helpers import group_parameters, group_modules
            except ImportError:
                raise RuntimeError(
                    'Please install latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`')
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception as e:
            logging.warning('grad checkpointing not supported for this timm image tower, continuing without...')

    def forward(self, x, output_tokens=None):
        output_tokens = self.output_tokens if output_tokens is None else output_tokens
        
        # 获取真正的底层模型，无论是否被PEFT包装
        trunk_model = self.trunk.get_base_model() if hasattr(self.trunk, 'get_base_model') else self.trunk

        # 直接调用特征提取器
        if hasattr(trunk_model, 'forward_features'):
            # 对于支持 forward_features 的模型 (大多数timm ViT)
            features = trunk_model.forward_features(x)
        else:
            # 对于不支持的模型，回退到标准 forward
            features = trunk_model(x)

        if output_tokens:
            # 当需要输出 tokens 时，我们假设 features 包含了 patch tokens
            # 对于 ViT, forward_features 的输出通常是 (B, num_patches + 1, C)
            # 我们需要分离 [CLS] token 和 patch tokens
            cls_token = features[:, 0]
            patch_tokens = features[:, 1:]
            # 返回 (池化/CLS特征, patch tokens) 的元组，以匹配 visual.py 的期望
            return self.head(cls_token), patch_tokens

        # 如果不需要输出 tokens，则正常通过 head 进行处理
        
        x = self.head(features[:, 0])

        return x

    def apply_lora(self, r=16, alpha=32, dropout=0.1, target_modules=None):
        if not self.use_lora:
            return
        
        if self.use_lora:
            if self.lora_config["lora_target_modules"] is None:
                self.lora_config["lora_target_modules"] = ["qkv", "proj", "fc1", "fc2"]
            
            lora_config = LoraConfig(
                r=self.lora_config["lora_r"],
                lora_alpha=self.lora_config["lora_alpha"],
                target_modules=self.lora_config["lora_target_modules"],
                lora_dropout=self.lora_config["lora_dropout"],
                bias=self.lora_config["lora_bias"],
                task_type="FEATURE_EXTRACTION",
            )
            
            self.trunk = get_peft_model(self.trunk, lora_config)
            self.is_peft_model = True
            
            # 修改基础模型的 forward 方法以接受视觉输入
            base_model = self.trunk.get_base_model()
            original_forward = base_model.forward
            
            # 定义新的 forward 方法
            def new_forward(x=None, pixel_values=None, input_ids=None, **kwargs):
                if x is not None:
                    return original_forward(x)
                elif pixel_values is not None:
                    return original_forward(pixel_values)
                elif input_ids is not None:
                    # 将 input_ids 转换为视觉模型期望的格式
                    return original_forward(input_ids)
                else:
                    raise ValueError("No valid input provided")
            
            base_model.forward = new_forward
            
            for param in self.head.parameters():
                param.requires_grad = True