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

    ls_init_value: Optional[float] = None
    patch_dropout: float = 0.
    attentional_pool: bool = False
    attn_pooler_queries: int = 256
    attn_pooler_heads: int = 8
    no_ln_pre: bool = False
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None


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
    ls_init_value: Optional[float] = None
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False
    final_ln_after_pool: bool = False
    pool_type: str = 'argmax'
    proj_bias: bool = False
    proj_type: str = 'linear'
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'


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

    act_layer = QuickGELU if quick_gelu else nn.GELU
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
        output_tokens=vision_cfg.output_tokens,
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

        x = self.token_embedding(text).to(cast_dtype)

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
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
        self.visual.lock(unlocked_groups=0, freeze_bn_stats=True)

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
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


class DynamicSemanticDecoupling(nn.Module):
    def __init__(self, input_dim: int, num_components: int = 32, ema_beta: float = 0.9):
        super().__init__()
        self.input_dim = input_dim
        self.num_components = num_components
        self.ema_beta = ema_beta

        self.register_buffer('attribute_vectors', torch.zeros(num_components, input_dim))
        self.register_buffer('reconstruction_operator', torch.zeros(num_components, input_dim))
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('decoupling_initialized', torch.zeros(1))

    def _eigen_decompose(self, features: torch.Tensor):
        mean_vector = torch.mean(features, dim=0)
        centered_data = features - mean_vector
        cov_matrix = torch.matmul(centered_data.T, centered_data) / max(centered_data.size(0) - 1, 1)
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        U, S, Vh = torch.linalg.svd(cov_matrix.double())
        W = U[:, :self.num_components].T.float()
        return W, W.clone(), mean_vector.float()

    @torch.no_grad()
    def compute_decoupling(self, features: torch.Tensor):
        return self._eigen_decompose(features)

    @torch.no_grad()
    def update_decoupling(self, W: torch.Tensor, Winv: torch.Tensor, mean: torch.Tensor):
        if self.decoupling_initialized.item() > 0.5:
            beta = self.ema_beta
            W = beta * self.attribute_vectors + (1 - beta) * W
            W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
            Winv = beta * self.reconstruction_operator + (1 - beta) * Winv
            mean = beta * self.feature_mean + (1 - beta) * mean
        self.attribute_vectors = W.to(self.attribute_vectors.device)
        self.reconstruction_operator = Winv.to(self.reconstruction_operator.device)
        self.feature_mean = mean.to(self.feature_mean.device)
        self.decoupling_initialized.fill_(1.0)

    @torch.no_grad()
    def sanitize_after_load(self):
        if float(self.decoupling_initialized.item()) > 0.5 and float(self.attribute_vectors.abs().sum()) == 0.0:
            self.decoupling_initialized.zero_()
            logging.warning(
                'DynamicSemanticDecoupling: decoupling_initialized=1 but attribute vectors are all zero'
                '(skipped by shape filtering at load, common when --dsd-components mismatches the checkpoint); '
                'reset to uninitialized, will be recomputed by the next update_implicit_supervision')

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        centered_data = features - self.feature_mean
        projected = torch.matmul(centered_data, self.attribute_vectors.T)
        reconstructed = torch.matmul(projected, self.reconstruction_operator) + self.feature_mean
        return reconstructed


class GCMCLIP(nn.Module):
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
            implicit_start_epoch: int = 0,
            dsd_components: int = 32,
            dsd_ema_beta: float = 0.9,
            dsd_norm_space: bool = False,
            logit_scale_max: float = 100.0,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.logit_scale_max = logit_scale_max
        self._log_logit_scale_max = float(np.log(logit_scale_max))
        self.dsd_norm_space = dsd_norm_space

        from .loss import Projection_head
        self.image_location_proj = Projection_head(embed_dim, output_dim=embed_dim // 4)
        self.text_location_proj = Projection_head(embed_dim, output_dim=embed_dim // 4)
        self.image_health_proj = Projection_head(embed_dim // 4, output_dim=embed_dim // 16)
        self.text_health_proj = Projection_head(embed_dim // 4, output_dim=embed_dim // 16)

        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

        self.n_coarse_clusters = 8
        self.n_fine_clusters = 16
        self.dsd = DynamicSemanticDecoupling(
            input_dim=embed_dim,
            num_components=dsd_components,
            ema_beta=dsd_ema_beta,
        )

        self.register_buffer('coarse_centers', torch.zeros(self.n_coarse_clusters, embed_dim))
        self.register_buffer('fine_centers', torch.zeros(self.n_fine_clusters, embed_dim))

        self.implicit_start_epoch = implicit_start_epoch
        self.current_epoch = implicit_start_epoch - 1

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
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
        image_logits = self.logit_scale.clamp(max=self._log_logit_scale_max).exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            epoch: Optional[int] = -1
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if ((text is not None and image is not None) and (epoch > self.implicit_start_epoch)
                and self.dsd.decoupling_initialized.item() > 0.5):
            with torch.no_grad():
                decoupled_feature = self.dsd(text_features.detach())

                coarse_sim = torch.mm(
                    F.normalize(decoupled_feature, p=2, dim=-1),
                    F.normalize(self.coarse_centers, p=2, dim=-1).T
                )
                coarse_probs = F.softmax(coarse_sim / 0.1, dim=-1)
                implicit_category_0 = coarse_probs.detach()

                fine_sim = torch.mm(
                    F.normalize(decoupled_feature, p=2, dim=-1),
                    F.normalize(self.fine_centers, p=2, dim=-1).T
                )
                fine_probs = F.softmax(fine_sim / 0.1, dim=-1)
                implicit_category_1 = fine_probs.detach()
        elif text is not None and image is not None:
            implicit_category_0 = torch.zeros((text_features.size(0), self.n_coarse_clusters), device=text_features.device)
            implicit_category_1 = torch.zeros((text_features.size(0), self.n_fine_clusters), device=text_features.device)
        else:
            implicit_category_0 = None
            implicit_category_1 = None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "implicit_category_0": implicit_category_0,
                "implicit_category_1": implicit_category_1,
                "logit_scale": self.logit_scale.clamp(max=self._log_logit_scale_max).exp()
            }
            if image_features is not None:
                image_location_embed = self.image_location_proj(image_features)
                out_dict["image_location_proj"] = image_location_embed
                out_dict["image_health_proj"] = self.image_health_proj(image_location_embed)
            if text_features is not None:
                text_location_embed = self.text_location_proj(text_features)
                out_dict["text_location_proj"] = text_location_embed
                out_dict["text_health_proj"] = self.text_health_proj(text_location_embed)
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.clamp(max=self._log_logit_scale_max).exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.clamp(max=self._log_logit_scale_max).exp()

    def _init_centers(self, features, k):
        indices = [torch.randint(0, len(features), (1,)).item()]

        for _ in range(1, k):
            dist = torch.cdist(features, features[indices])
            min_dist = dist.min(dim=1)[0]**2
            min_dist = torch.clamp(min_dist, min=1e-10)
            prob = min_dist / (torch.sum(min_dist) + 1e-8)
            indices.append(torch.multinomial(prob, 1).item())

        return indices

    @torch.no_grad()
    def mine_implicit_clusters(self, features, explicit_labels, min_samples_per_cluster=400, max_iters=10000, stability_threshold=0.01, forced_k=None):
        features = F.normalize(features, p=2, dim=1)

        if forced_k is not None:
            cluster_probs, centers = self._balanced_clustering(
                features, explicit_labels, forced_k, max_iters, stability_threshold)
            return cluster_probs, centers, forced_k

        class_counts = explicit_labels.sum(dim=0)
        min_k = max(1, int(torch.max(class_counts) / min_samples_per_cluster))

        optimal_k = self.adaptive_cluster_count(features, min_k)

        cluster_probs, centers = self._balanced_clustering(
            features, explicit_labels, optimal_k, max_iters, stability_threshold)

        return cluster_probs, centers, optimal_k

    @torch.no_grad()
    def adaptive_cluster_count(self, features, min_k, num_trials=3):
        distortions = []
        k_values = list(range(min_k, min_k + 32, 2))

        for k in k_values:
            total_distortion = 0
            for _ in range(num_trials):
                indices = self._init_centers(features, k)
                centers = features[indices].clone()

                for _ in range(10):
                    sim_matrix = torch.mm(features, centers.t())
                    assignments = sim_matrix.argmax(dim=1)

                    new_centers = centers.clone()
                    for j in range(k):
                        mask = (assignments == j)
                        if mask.any():
                            new_centers[j] = features[mask].mean(dim=0)
                        else:
                            dist_to_centers = 1 - torch.mm(features, centers.t())
                            farthest_idx = dist_to_centers.sum(dim=1).argmax()
                            new_centers[j] = features[farthest_idx]

                    centers = new_centers

                sim_matrix = torch.mm(features, centers.t())
                max_sim = sim_matrix.max(dim=1)[0]
                distortion = (1 - max_sim).sum().item()
                total_distortion += distortion

            distortions.append(total_distortion / num_trials)

        if len(distortions) > 2:
            first_deriv = np.diff(distortions)
            second_deriv = np.diff(first_deriv)
            elbow_point = np.argmax(second_deriv) + 1
            optimal_k = k_values[elbow_point]
        else:
            optimal_k = k_values[0]

        return optimal_k

    @torch.no_grad()
    def _balanced_clustering(self, features, explicit_labels, k, max_iters, stability_threshold):
        indices = self._init_centers(features, k)
        centers = features[indices].clone()

        for _ in range(max_iters):
            sim_matrix = torch.mm(features, centers.t())
            assignments = sim_matrix.argmax(dim=1)

            new_centers = centers.clone()
            cluster_changed = False

            for j in range(k):
                mask = (assignments == j)

                if not mask.any():
                    dist_to_centers = 1 - torch.mm(features, centers.t())
                    farthest_idx = dist_to_centers.sum(dim=1).argmax()
                    new_centers[j] = features[farthest_idx]
                    cluster_changed = True
                    continue

                cluster_labels = explicit_labels[mask]
                class_distribution = cluster_labels.sum(dim=0) / mask.sum()

                max_class_ratio = torch.max(class_distribution)
                if max_class_ratio > 0.5:
                    dominant_class = torch.argmax(class_distribution)
                    non_dom_in_cluster = cluster_labels[:, dominant_class] == 0
                    if bool(non_dom_in_cluster.any()):
                        new_centers[j] = F.normalize(
                            features[mask][non_dom_in_cluster].mean(dim=0), p=2, dim=0)
                        cluster_changed = True
                        continue

                cluster_mean = features[mask].mean(dim=0)
                new_centers[j] = F.normalize(cluster_mean, p=2, dim=0)

            center_similarity = F.cosine_similarity(new_centers, centers, dim=1)
            center_shift = (1 - center_similarity).mean()
            centers = new_centers
            if center_shift < stability_threshold:
                break

        final_sim_matrix = torch.mm(features, centers.t())
        cluster_probs = F.softmax(final_sim_matrix / 0.1, dim=-1)

        return cluster_probs, centers

    def update_implicit_supervision(self, args, epoch, text, explicit_labels):
        if epoch == self.current_epoch:
            return

        self.current_epoch = epoch

        features_list = []
        with torch.no_grad():
            for i in range(args.accum_freq):
                batch_texts = text[i]
                batch_features = self.encode_text(batch_texts, normalize=self.dsd_norm_space)
                features_list.append(batch_features)

        features = torch.cat(features_list, dim=0)

        W, Winv, mean = self.dsd.compute_decoupling(features)
        self.dsd.update_decoupling(W, Winv, mean)

        decoupled = self.dsd(features)

        coarse_probs, coarse_centers, k_coarse = self.mine_implicit_clusters(
            decoupled,
            explicit_labels,
            min_samples_per_cluster=200,
            forced_k=8,
            )
        fine_probs, fine_centers, k_fine = self.mine_implicit_clusters(
            decoupled,
            explicit_labels,
            min_samples_per_cluster=100,
            forced_k=16,
            )
        logging.info(
            f"implicit supervision updated (epoch {epoch}): coarse k={k_coarse}, fine k={k_fine}"
        )

        self.n_coarse_clusters = k_coarse
        self.n_fine_clusters = k_fine
        device = coarse_centers.device
        if self.coarse_centers.shape[0] != k_coarse:
            self.register_buffer('coarse_centers', torch.zeros(k_coarse, self.coarse_centers.shape[1], device=device))

        if self.fine_centers.shape[0] != k_fine:
            self.register_buffer('fine_centers', torch.zeros(k_fine, self.fine_centers.shape[1], device=device))

        if epoch == 0:
            self.coarse_centers = coarse_centers.clone().detach()
            self.fine_centers = fine_centers.clone().detach()
        else:
            alpha = 0.996
            self.coarse_centers = alpha * self.coarse_centers + (1 - alpha) * coarse_centers
            self.fine_centers = alpha * self.fine_centers + (1 - alpha) * fine_centers


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
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
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp


def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
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
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1
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
    module.image_mean = preprocess_cfg['mean']
    module.image_std = preprocess_cfg['std']
    module.preprocess_cfg = copy.deepcopy(preprocess_cfg)


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




LEGACY_GCM_KEY_MAP = {
    'implicit_miner.eigenvectors': 'dsd.attribute_vectors',
    'implicit_miner.reconstruct_matrix': 'dsd.reconstruction_operator',
    'implicit_miner.mean_vector': 'dsd.feature_mean',
    'implicit_miner.basis_initialized': 'dsd.decoupling_initialized',
    'implicit_centers_0': 'coarse_centers',
    'implicit_centers_1': 'fine_centers',
}


def remap_legacy_gcm_state_dict(state_dict: dict):
    if not any(k in state_dict for k in LEGACY_GCM_KEY_MAP):
        return state_dict
    logging.info('Remapping legacy GCM checkpoint keys (implicit_miner.*, implicit_centers_*) '
                 'to current DynamicSemanticDecoupling naming')
    return {LEGACY_GCM_KEY_MAP.get(k, k): v for k, v in state_dict.items()}
