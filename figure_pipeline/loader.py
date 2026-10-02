"""共享数据加载与特征缓存。

encode_images() 将图像特征缓存为 output/figure_data/cache/<model>/<dataset>_<modality>.npy,
键 = (model_name, dataset_name, modality), 旁路 <...>.fp 文件记录 manifest 指纹:
指纹不一致即视为陈旧缓存强制重编码(A族事故修复——换 manifest 不换 key 曾导致
特征/标签错位, 指标全部跌至 chance)。文本缓存键自带内容 hash, 天然安全。
图像读取失败直接抛错(旧行为静默补零张量, 会把缺图样本混成全零特征污染评估)。
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figure_pipeline import paths as P


class ManifestDataset(Dataset):
    """manifest CSV -> (image_tensor, text, category, location, index)。"""

    def __init__(self, manifest_csv, preprocess, text_field="original_text", image_size=224):
        self.df = pd.read_csv(manifest_csv)
        self.paths = self.df["image_path_local"].tolist()
        self.texts = self.df[text_field].fillna("").tolist() if text_field in self.df \
            else [""] * len(self.df)
        self.categories = self.df["category"].fillna("Unknown").tolist() \
            if "category" in self.df else ["Unknown"] * len(self.df)
        self.locations = self.df["location"].fillna("Unknown").tolist() \
            if "location" in self.df else ["Unknown"] * len(self.df)
        self.preprocess = preprocess
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # 缺图/坏图直接抛错: 静默补零会让该样本以全零特征混入评估, 指标不可信
        img = Image.open(self.paths[i]).convert("RGB")
        tensor = self.preprocess(img)
        return tensor, self.texts[i], self.categories[i], self.locations[i], i


class TextOnlyDataset(Dataset):
    def __init__(self, texts):
        self.texts = list(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], i


def _collate_one(batch):
    imgs = torch.stack([b[0] for b in batch])
    return (imgs, [b[1] for b in batch], [b[2] for b in batch],
            [b[3] for b in batch], [b[4] for b in batch])


def make_image_loader(manifest_csv, adapter, batch_size=64, workers=8, text_field="original_text"):
    ds = ManifestDataset(manifest_csv, adapter.preprocess, text_field, adapter.image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True, collate_fn=_collate_one), ds


def cache_path(model_name, dataset_name, modality):
    d = os.path.join(P.CACHE, model_name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{dataset_name}_{modality}.npy")


def manifest_fingerprint(manifest_csv):
    """manifest 内容代理指纹(路径+大小+mtime): 换 manifest 必变, 用于缓存失效。"""
    st = os.stat(manifest_csv)
    sig = f"{os.path.realpath(manifest_csv)}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.md5(sig.encode()).hexdigest()[:10]


def encode_images(adapter, manifest_csv, dataset_name, batch_size=64, workers=8,
                  text_field="original_text", force=False):
    """编码 manifest 全部图像 -> np.ndarray [N, D]（L2 归一化）, 带缓存。

    缓存有效性由 .fp 旁路文件保障: 记录编码时所用的 manifest 指纹,
    与当前 manifest 不一致(或旁路缺失)即重编码。
    """
    out = cache_path(adapter.name, dataset_name, "image")
    fp_file = out + ".fp"
    fp = manifest_fingerprint(manifest_csv)
    if os.path.exists(out) and not force:
        cached_fp = open(fp_file).read().strip() if os.path.exists(fp_file) else None
        if cached_fp == fp:
            return np.load(out)
        print(f"[cache-stale] {out} 的 manifest 指纹不符"
              f"(缓存 {cached_fp} vs 当前 {fp}) -> 重编码")
    loader, ds = make_image_loader(manifest_csv, adapter, batch_size, workers, text_field)
    feats = np.zeros((len(ds), adapter.embed_dim), dtype=np.float32)
    for imgs, _, _, _, idxs in tqdm(loader, desc=f"img[{adapter.name}/{dataset_name}]", unit="batch"):
        f = adapter.encode_image(imgs).cpu().numpy()
        feats[np.array(idxs)] = f
    np.save(out, feats)
    with open(fp_file, "w") as f:
        f.write(fp)
    return feats


def encode_texts(adapter, texts, dataset_name, batch_size=256, force=False):
    """编码文本列表 -> np.ndarray [N, D], 带缓存（以内容 hash 区分）。"""
    sig = hashlib.md5(("|".join(map(str, texts)) + adapter.name).encode()).hexdigest()[:16]
    out = cache_path(adapter.name, f"{dataset_name}_{sig}", "text")
    if os.path.exists(out) and not force:
        return np.load(out)
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"txt[{adapter.name}/{dataset_name}]", unit="batch"):
        f = adapter.encode_text(texts[i:i + batch_size]).cpu().numpy()
        feats.append(f)
    feats = np.concatenate(feats, axis=0)
    np.save(out, feats)
    return feats
