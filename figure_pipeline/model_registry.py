"""6+ 模型统一 adapter：encode_image / encode_text / preprocess。

设计:
- 每个模型一个 build_* 函数, 返回 ModelAdapter
- 全部离线可用（GLoRIA/MGCA 的 BERT 架构从本地 bert-base-uncased 构建, 权重由 ckpt 覆盖）
- encode_* 返回 L2 归一化 fp32 embedding
"""
import os
import sys
import types
from types import SimpleNamespace

# HF 网络不可达: 全部走本地缓存（BiomedCLIP/BiomedBERT/pubmedclip/medsiglip 均已缓存）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figure_pipeline import paths as P

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
HALF_MEAN = HALF_STD = (0.5, 0.5, 0.5)
# Bio_ClinicalBERT 本地目录: vocab 28996 与 GLoRIA/MGCA ckpt 对齐;
# pytorch_model.bin 为架构占位（真实权重由各自 ckpt 覆盖）
BERT_LOCAL = os.path.join(P.MODEL_DIR, "Bio_ClinicalBERT")


class ModelAdapter:
    def __init__(self, name, model, preprocess, embed_dim, image_size, device,
                 text_encoder="identity"):
        self.name = name
        self.model = model            # 主模型（或 dict of towers）
        self.preprocess = preprocess
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.device = device
        self._text_encoder = text_encoder

    @torch.inference_mode()
    def encode_image(self, x, normalize=True):
        feats = self._encode_image_impl(x.to(self.device))
        if feats.dtype != torch.float32:
            feats = feats.float()
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats

    @torch.inference_mode()
    def encode_text(self, texts, normalize=True):
        feats = self._encode_text_impl(list(texts))
        if feats.dtype != torch.float32:
            feats = feats.float()
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats

    # 子类覆盖
    def _encode_image_impl(self, x):
        raise NotImplementedError

    def _encode_text_impl(self, texts):
        raise NotImplementedError


def _to_device(m, device):
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def _tensor_of(out):
    """transformers 5.x 可能返回 ModelOutput, 统一抽取 tensor。"""
    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(out, attr, None)
        if torch.is_tensor(v):
            return v
    return out[0]


# ================================================================
# 1/2) open_clip CMCLIP (gcm_clip, biomedclip)
#   fork 的 CLIP 类不支持 hf-hub 文本塔, 但 CMCLIP 类与 BiomedCLIP 权重
#   结构 1:1 对齐（text.transformer.* / visual.trunk.* / text.proj.*）,
#   原始 BiomedCLIP bin 手动 strict=False 加载, implicit 缓冲区不参与编码。
# ================================================================
def _build_open_clip(name, ckpt, force_cmclip, device):
    import open_clip
    if force_cmclip and ckpt.endswith(".bin"):
        # 原始 BiomedCLIP: 先建 CMCLIP 骨架, 手动加载 bin（跳过 implicit 缓冲区）
        model, _, preprocess = open_clip.create_model_and_transforms(
            P.HUB_NAME, pretrained=None, force_CMCLIP=True,
            precision="fp32", device=device, jit=False)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        sd.pop("text.transformer.embeddings.position_ids", None)
        miss, unexp = model.load_state_dict(sd, strict=False)
        assert not unexp, f"unexpected: {unexp[:5]}"
        assert all(k.startswith(("implicit_centers", "implicit_miner")) for k in miss), \
            f"unexpected missing: {miss[:8]}"
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            P.HUB_NAME, pretrained=ckpt, force_CMCLIP=force_cmclip,
            precision="fp32", device=device, jit=False)
    model = _to_device(model, device)
    tokenizer = open_clip.get_tokenizer(P.HUB_NAME)

    class _A(ModelAdapter):
        def __init__(self):
            super().__init__(name, model, preprocess, 512, 224, device)
            self.tokenizer = tokenizer

        def _encode_image_impl(self, x):
            return self.model.encode_image(x, normalize=True)

        def _encode_text_impl(self, texts):
            tokens = self.tokenizer(texts).to(self.device)
            return self.model.encode_text(tokens, normalize=True)

    return _A()


# ================================================================
# 3) PubMedCLIP (HF CLIPModel)
# ================================================================
def _build_pubmedclip(device):
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained(P.CKPT["pubmedclip"])
    tok = CLIPTokenizer.from_pretrained(P.CKPT["pubmedclip"])
    model = _to_device(model, device)
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)])

    class _A(ModelAdapter):
        def __init__(self):
            super().__init__("pubmedclip", model, preprocess, 512, 224, device)

        def _encode_image_impl(self, x):
            return _tensor_of(self.model.get_image_features(pixel_values=x))

        def _encode_text_impl(self, texts):
            batch = tok(texts, padding=True, truncation=True, max_length=77,
                        return_tensors="pt").to(self.device)
            return _tensor_of(self.model.get_text_features(**batch))

    return _A()


# ================================================================
# 4) MedSigLIP (HF SiglipModel, 448)
# ================================================================
def _build_medsiglip(device):
    from transformers import SiglipModel, AutoTokenizer
    model = SiglipModel.from_pretrained(P.CKPT["medsiglip"])
    tok = AutoTokenizer.from_pretrained(P.CKPT["medsiglip"])
    model = _to_device(model, device)
    preprocess = transforms.Compose([
        transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(448), transforms.ToTensor(),
        transforms.Normalize(HALF_MEAN, HALF_STD)])

    class _A(ModelAdapter):
        def __init__(self):
            super().__init__("medsiglip", model, preprocess, 1152, 448, device)

        def _encode_image_impl(self, x):
            return _tensor_of(self.model.get_image_features(pixel_values=x))

        def _encode_text_impl(self, texts):
            batch = tok(texts, padding=True, truncation=True, max_length=64,
                        return_tensors="pt").to(self.device)
            return _tensor_of(self.model.get_text_features(**batch))

    return _A()


# ================================================================
# 5) GLoRIA (ResNet-50 + 4 层 BERT sum; 全局 768)
# ================================================================
def _load_gloria_modules():
    """绕过 gloria 包重型 __init__（datasets/albumentations 等）,
    用合成包只加载 text_model 与 vision_model。"""
    import importlib.util
    root = os.path.join(P.MODEL_DIR, "gloria", "gloria")
    pkg = types.ModuleType("gloria_lite")
    pkg.__path__ = [root]
    sys.modules["gloria_lite"] = pkg
    models = types.ModuleType("gloria_lite.models")
    models.__path__ = [os.path.join(root, "models")]
    sys.modules["gloria_lite.models"] = models

    for mod_name, fname in [("gloria_lite.models.cnn_backbones", "cnn_backbones.py"),
                            ("gloria_lite.models.text_model", "text_model.py"),
                            ("gloria_lite.models.vision_model", "vision_model.py")]:
        spec = importlib.util.spec_from_file_location(
            mod_name, os.path.join(root, "models", fname))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        setattr(models, fname[:-3], mod)
    return sys.modules["gloria_lite.models.text_model"], \
        sys.modules["gloria_lite.models.vision_model"]


def _build_gloria(device):
    tm, vm = _load_gloria_modules()
    BertEncoder = tm.BertEncoder

    cfg = SimpleNamespace(model=SimpleNamespace(
        vision=SimpleNamespace(model_name="resnet_50", pretrained=False, freeze_cnn=True),
        text=SimpleNamespace(bert_type=BERT_LOCAL, last_n_layers=4,
                             aggregate_method="sum", norm=False, embedding_dim=768,
                             freeze_bert=True, agg_tokens=True)))

    img_enc = vm.ImageEncoder(cfg)
    txt_enc = BertEncoder(cfg)

    # 加载 lightning ckpt
    from segmentation.encoders import _load_lightning_state_dict
    sd = _load_lightning_state_dict(P.CKPT["gloria"])
    img_sd = {k[len("gloria.img_encoder."):]: v for k, v in sd.items()
              if k.startswith("gloria.img_encoder.")}
    txt_sd = {k[len("gloria.text_encoder."):]: v for k, v in sd.items()
              if k.startswith("gloria.text_encoder.")}
    txt_sd.pop("model.embeddings.position_ids", None)  # 旧式 buffer
    miss, unexp = img_enc.load_state_dict(img_sd, strict=False)
    miss2, unexp2 = txt_enc.load_state_dict(txt_sd, strict=False)
    # 允许缺失: txt 的 emb_global/emb_local 为 None 不在 ckpt; img 无
    real_miss = [k for k in miss if not k.startswith(("emb_global", "emb_local"))]
    assert not real_miss, f"gloria missing: {real_miss}"
    assert not unexp and not unexp2, f"gloria unexpected: {unexp} {unexp2}"
    img_enc = _to_device(img_enc, device)
    txt_enc = _to_device(txt_enc, device)
    tokenizer = txt_enc.tokenizer

    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(HALF_MEAN, HALF_STD)])

    class _A(ModelAdapter):
        def __init__(self):
            super().__init__("gloria", {"img": img_enc, "txt": txt_enc},
                             preprocess, 768, 224, device)

        def _encode_image_impl(self, x):
            img_enc = self.model["img"]
            g, l = img_enc(x, get_local=True)
            emb_g, _ = img_enc.generate_embeddings(g, l)
            return emb_g

        def _encode_text_impl(self, texts):
            batch = tokenizer(texts, padding="max_length", truncation=True,
                              max_length=97, return_tensors="pt").to(self.device)
            _, emb_g, _ = self.model["txt"](batch["input_ids"],
                                            batch["attention_mask"],
                                            batch["token_type_ids"])
            return emb_g

    return _A()


# ================================================================
# 6) MGCA (ViT-B/16 + ConVIRT BERT; 全局 128)
#   文本塔直接用 HF BertModel（6 层, bert_config.json）+ ckpt 权重,
#   report_feat = CLS token last hidden（与 mgca aggregate_tokens 等价）。
# ================================================================
def _build_mgca(device):
    mgca_root = os.path.join(P.MODEL_DIR, "MGCA")
    if mgca_root not in sys.path:
        sys.path.insert(0, mgca_root)
    # 离线: 拦截 DEiT 权重下载（ckpt 会覆盖全部权重）
    import torch.hub as _hub
    _orig = _hub.load_state_dict_from_url
    _hub.load_state_dict_from_url = lambda *a, **k: {"model": {}}

    try:
        from mgca.models.backbones.encoder import ImageEncoder as MgcaImage, GlobalEmbedding
        img_enc = MgcaImage(model_name="vit_base", pretrained=False,
                            output_dim=128, hidden_dim=2048)
    finally:
        _hub.load_state_dict_from_url = _orig

    from transformers import BertConfig, BertModel
    cfg = BertConfig.from_json_file(os.path.join(mgca_root, "mgca", "configs", "bert_config.json"))
    # mgca 是 BLIP 式 cross-attention 配置; 纯文本编码不需要, 关掉并丢弃对应权重。
    # token_type_embeddings 不在 ckpt → 零初始化且前向不传 token_type_ids。
    cfg.add_cross_attention = False
    cfg.is_decoder = False
    bert = BertModel(cfg, add_pooling_layer=False)

    sd = torch.load(P.CKPT["mgca"], map_location="cpu", weights_only=False)
    sd = sd["state_dict"] if "state_dict" in sd else sd

    # 图像塔
    img_sub = {k[len("img_encoder_q."):]: v for k, v in sd.items()
               if k.startswith("img_encoder_q.")}
    miss, unexp = img_enc.load_state_dict(img_sub, strict=False)
    bad_miss = [k for k in miss if "position_ids" not in k]
    assert not bad_miss, f"mgca img missing: {bad_miss[:5]}"
    assert not unexp, f"mgca img unexpected: {unexp[:5]}"

    # 文本塔 (HF BertModel) + global_embed; 丢弃 crossattention 权重（纯文本模式不用）
    txt_sub = {k[len("text_encoder_q.model."):]: v for k, v in sd.items()
               if k.startswith("text_encoder_q.model.")
               and "crossattention" not in k}
    txt_sub.pop("embeddings.position_ids", None)
    miss, unexp = bert.load_state_dict(txt_sub, strict=False)
    # token_type_embeddings 不在 ckpt（BLIP 式无该项）→ 零初始化, 前向不传 token_type
    bad_miss = [k for k in miss if "pooler" not in k and "token_type" not in k]
    assert not bad_miss, f"mgca txt missing: {bad_miss[:5]}"
    assert not unexp, f"mgca txt unexpected: {unexp[:5]}"
    with torch.no_grad():
        bert.embeddings.token_type_embeddings.weight.zero_()

    txt_global = GlobalEmbedding(768, 2048, 128)
    g_sub = {k[len("text_encoder_q.global_embed."):]: v for k, v in sd.items()
             if k.startswith("text_encoder_q.global_embed.")}
    txt_global.load_state_dict(g_sub, strict=True)

    img_enc = _to_device(img_enc, device)   # BatchNorm eval 模式用 running stats
    bert = _to_device(bert, device)
    txt_global = _to_device(txt_global, device)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_LOCAL)

    preprocess = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(HALF_MEAN, HALF_STD)])

    class _A(ModelAdapter):
        def __init__(self):
            super().__init__("mgca", {"img": img_enc, "txt": bert, "tg": txt_global},
                             preprocess, 128, 224, device)
            self.tokenizer = tokenizer

        def _encode_image_impl(self, x):
            img_enc = self.model["img"]
            g, _ = img_enc(x, get_local=False)
            return img_enc.global_embed(g)

        def _encode_text_impl(self, texts):
            batch = self.tokenizer(texts, padding="max_length", truncation=True,
                                   max_length=97, return_tensors="pt").to(self.device)
            out = self.model["txt"](input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"])
            report_feat = out.last_hidden_state[:, 0]
            return self.model["tg"](report_feat)

    return _A()


# ================================================================
# BioViL (microsoft/BiomedVLP-CXR-BERT-specialized + ResNet50 图像塔)
#   文本: CXR-BERT(projection 128) | 图像: ResNet50(2048) -> Conv1x1 投影 128
#   输出 128 维联合空间; ms-cxr 预处理 = ImageNet 统计
# ================================================================
def _build_biovil(device):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    import torchvision
    from torchvision.models import resnet50

    d = os.path.join(P.MODEL_DIR, "BioViL")
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    cfg = AutoConfig.from_pretrained(d, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    txt = AutoModel.from_pretrained(d, config=cfg, trust_remote_code=True)

    class _ResNetWrap(nn.Module):
        def __init__(self):
            super().__init__()
            r = resnet50(weights=None)
            r.fc = nn.Identity()
            self.encoder = r

        def forward(self, x):
            return self.encoder(x)

    class _ProjWrap(nn.Module):
        # 与 ckpt 键 projector.model.{0,1,3}.* 对齐: Conv1x1(2048->128,无偏置)+BN+ReLU+Conv1x1(128->128)
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(2048, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 128, 1))

        def forward(self, f):
            return self.model(f.unsqueeze(-1).unsqueeze(-1)).flatten(1)

    class _ImgTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _ResNetWrap()
            self.projector = _ProjWrap()

        def forward(self, x):
            return self.projector(self.encoder(x))

    img = _ImgTower()
    sd = torch.load(os.path.join(d, "biovil_image_resnet50_proj_size_128.pt"),
                    map_location="cpu")
    # ckpt 保留了 resnet 原 fc(1000类)权重, 推理不使用 -> 剔除后 strict 加载
    sd.pop("encoder.encoder.fc.weight", None)
    sd.pop("encoder.encoder.fc.bias", None)
    img.load_state_dict(sd, strict=True)
    _to_device(img, device)
    _to_device(txt, device)

    preprocess = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    class _A(ModelAdapter):
        def __init__(self):
            super().__init__("biovil", {"img": img, "txt": txt}, preprocess, 128, 224, device)
            self.tokenizer = tokenizer

        def _encode_image_impl(self, x):
            return self.model["img"](x)

        def _encode_text_impl(self, texts):
            batch = self.tokenizer(texts, padding="max_length", truncation=True,
                                   max_length=128, return_tensors="pt").to(self.device)
            out = self.model["txt"](**batch)
            return out.logits  # CXRBert 投影输出 [B,128]

    return _A()


# ================================================================
# MedCLIP (RyanWangZf/MedCLIP, 官方 medclip-pretrained.zip = ResNet50 变体)
#   图像: ResNet50(fc 2048->512) | 文本: BERT-base + projection_head(768->512, 无偏置)
#   联合维 512; tokenizer/config 取本地 bert-base-uncased
# ================================================================
def _build_medclip(device):
    from transformers import AutoModel, AutoConfig, AutoTokenizer
    from torchvision.models import resnet50

    d = os.path.join(P.MODEL_DIR, "MedCLIP")
    BERT_DIR = os.path.join(P.MODEL_DIR, "Bio_ClinicalBERT")  # MedCLIP 文本塔=Bio_ClinicalBERT(vocab 28996)

    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    cfg = AutoConfig.from_pretrained(BERT_DIR)
    bert = AutoModel.from_config(cfg)

    class _ImgWrap(nn.Module):
        def __init__(self):
            super().__init__()
            r = resnet50(weights=None)
            r.fc = nn.Linear(2048, 512, bias=False)  # ckpt: fc.weight(512,2048) 无偏置
            self.model = r

        def forward(self, x):
            return self.model(x)

    class _TxtWrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = bert
            self.projection_head = nn.Linear(768, 512, bias=False)

        def forward(self, **kw):
            return self.projection_head(self.model(**kw).pooler_output)

    class _MedCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = _ImgWrap()
            self.text_model = _TxtWrap()

    net = _MedCLIP()
    sd = torch.load(os.path.join(d, "pytorch_model.bin"), map_location="cpu")
    sd.pop("logit_scale", None)
    sd.pop("text_model.model.embeddings.position_ids", None)
    net.load_state_dict(sd, strict=True)
    _to_device(net, device)

    preprocess = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225))])

    class _A(ModelAdapter):
        def __init__(self):
            super().__init__("medclip", net, preprocess, 512, 224, device)
            self.tokenizer = tokenizer

        def _encode_image_impl(self, x):
            return self.model.vision_model(x)

        def _encode_text_impl(self, texts):
            batch = self.tokenizer(texts, padding="max_length", truncation=True,
                                   max_length=128, return_tensors="pt").to(self.device)
            return self.model.text_model(input_ids=batch["input_ids"],
                                         attention_mask=batch["attention_mask"])

    return _A()


# ================================================================
# 工厂
# ================================================================
def build_adapter(name, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "gcm_clip":
        return _build_open_clip(name, P.CKPT[name], True, device)
    if name == "biomedclip":
        # fork 的 CLIP 类不支持 hf-hub 文本塔; 统一走 CMCLIP 骨架 + bin 权重
        return _build_open_clip(name, P.CKPT[name], True, device)
    if name == "pubmedclip":
        return _build_pubmedclip(device)
    if name == "medsiglip":
        return _build_medsiglip(device)
    if name == "gloria":
        return _build_gloria(device)
    if name == "mgca":
        return _build_mgca(device)
    if name == "biovil":
        return _build_biovil(device)
    if name == "medclip":
        return _build_medclip(device)
    raise ValueError(f"unknown model: {name}")
