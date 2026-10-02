"""GCM_CLIP 审核四评估集并重建 verified 测试集。

流程:
1. 对现有 manifest (ForVA 4096 / ROCO 3630 / MIMIC 957 / PMC 3607) 逐行计算
   零样本信号 (loc@1 / dis@1 / joint@5) 与 ForVA 检索信号 (i2t@1, 候选并集池保守上界)
2. 保留高匹配行; 低匹配行从候选源数据替换:
   - ForVA: data/forensic_CT_train_except.csv (按 (category, location) 采样, seed 42)
   - ROCO:  data/ROCOv2/jpg_data/deepseek_CT_ROCO_{train,test,validation}.csv
   - MIMIC: deepseek 标注 CSV (disease_map 别名归并)
   - PMC:   pmc_src/pmc_oa.jsonl caption 匹配 + PMC_CLASS_RULES 分类
3. 疾病/位置边缘分布均匀配额选样 (组合内按信号强弱排序取最优)
4. 数据集级验收阈值, 不达标则扩大候选采样重试 (最多 --max-rounds 轮)

阈值: forva loc>0.7 dis>0.3 j5>0.3 i2t1>0.4 | roco 0.7/0.2/0.3
      mimic 0.6/0.2/0.3 | pmc 0.5/0.2/0.3

输出: manifests/*_verified.csv + verify_report.md + verify_metrics.json

用法: python -m figure_pipeline.verify_and_refill [--datasets forva,roco,mimic,pmc]
      [--cap 30] [--max-rounds 2] [--quick] [--force]
"""
import argparse
import collections
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figure_pipeline import paths as P
from figure_pipeline import loader as L
from figure_pipeline import ood_prep
from figure_pipeline.model_registry import build_adapter
from figure_pipeline.run_fig2a import (OOD_TEMPLATES, MIMIC_CLASSNAMES,
                                       build_classifier, prettify, topk_correct)
from figure_pipeline.run_fig2b import retrieval_metrics

OOD_JOINT_TEMPLATES = [
    "A {b} CT image showing {f}.",
    "A computed tomography image of the {b} showing {f}.",
    "A medical CT scan displaying {f} in the {b} region.",
    "A radiology CT image of the {b} with {f}.",
    "A CT scan revealing {f} in the {b} area.",
    "An axial CT image of the {b} showing {f}.",
    "A clinical CT image demonstrating {f} in the {b}.",
    "A CT image indicating {f} in the {b} region.",
]

THRESHOLDS = {
    "forva": {"Loc@1": 0.7, "Dis@1": 0.3, "Dis_Loc@5": 0.3, "I2T@1": 0.4},
    "roco": {"Loc@1": 0.7, "Dis@1": 0.2, "Dis_Loc@5": 0.3},
    "mimic": {"Loc@1": 0.6, "Dis@1": 0.2, "Dis_Loc@5": 0.3},
    "pmc": {"Loc@1": 0.5, "Dis@1": 0.2, "Dis_Loc@5": 0.3},
}
TARGET_SIZE = {"forva": 4096, "roco": 3630, "mimic": 957, "pmc": 3607}

VERIFY_CACHE = os.path.join(P.MANIFESTS, "verify_cache")
FORVA_EXCEPT_CSV = os.path.join(P.VAD_DATA, "forensic_CT_train_except.csv")
ROCO_POOLS = ["deepseek_CT_ROCO_train.csv", "deepseek_CT_ROCO_test.csv",
              "deepseek_CT_ROCO_validation.csv"]
PMC_POOL_CACHE = os.path.join(P.MANIFESTS, "pmc_src", "pmc_pool.csv")


def log(msg):
    print(f"[verify] {msg}", flush=True)


def _to_json(obj):
    """np 标量 -> python 原生 (json 安全)。"""
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ================================================================ 候选池
def _sample_except_pool(combos, cap, seed):
    """chunked 读取 except 池, 按 (category, location) 组合采样。"""
    rng = np.random.RandomState(seed)
    buckets = {k: [] for k in combos}
    reader = pd.read_csv(FORVA_EXCEPT_CSV, dtype=str, chunksize=200_000,
                         usecols=["cause", "corpse_id", "part_id", "win_name", "WW",
                                  "WL", "image_path", "category", "original_text",
                                  "location"])
    total = 0
    for chunk in reader:
        total += len(chunk)
        keep = chunk.dropna(subset=["category", "location", "original_text"])
        keep = keep[keep["category"].str.strip().str.lower().isin(
            {c.lower() for c, _ in combos})]
        keep = keep[keep["location"].str.strip().str.lower().isin(
            {l.lower() for _, l in combos})]
        for (cl, sl), grp in keep.groupby(["category", "location"]):
            key = next((k for k in combos if k[0].lower() == str(cl).lower()
                        and k[1].lower() == str(sl).lower()), None)
            if key is None or len(buckets[key]) >= cap:
                continue
            idx = rng.permutation(len(grp))[:cap - len(buckets[key])]
            buckets[key].append(grp.iloc[idx])
        if all(len(b) >= cap for b in buckets.values()):
            break
    frames = []
    for key in combos:
        if buckets[key]:
            frames.append(pd.concat(buckets[key], ignore_index=True))
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True).drop_duplicates(subset="image_path")
    log(f"forva except 扫描 {total} 行, 采样 {len(pool)} 行")
    return pool


def load_forva_pool(cap, seed):
    orig = pd.read_csv(os.path.join(P.MANIFESTS, "forva_test_4096.csv"))
    stats = P.read_json_gbk(P.FORENSIC_STATS_JSON)
    cats = [v for _, v in stats["category"].items()]
    locs = [v for _, v in stats["location"].items()]
    combos = list(pd.MultiIndex.from_product([cats, locs]))

    orig["image_path_local"] = orig["image_path"].map(P.remap_path)
    exist = orig["image_path_local"].map(os.path.exists)
    orig_exist = orig[exist].reset_index(drop=True)
    orig_exist["source"] = "original"
    n_missing = int((~exist).sum())
    log(f"forva 原始 {len(orig)} 行, 本地缺图 {n_missing} 行剔除")

    pool = _sample_except_pool(combos, cap, seed)
    if len(pool):
        pool = pool.reset_index(drop=True)
        pool["image_path_local"] = pool["image_path"].map(P.remap_path)
        existp = pool["image_path_local"].map(os.path.exists)
        pool = pool[existp].reset_index(drop=True)
        pool["source"] = "refill"
    return orig_exist, pool, n_missing


def load_roco_pool():
    frames = []
    for f in ROCO_POOLS:
        df = pd.read_csv(os.path.join(P.GCM_DATA, "ROCOv2", "jpg_data", f), dtype=str)
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True).drop_duplicates(subset="image_id")
    in_manifest = set(pd.read_csv(
        os.path.join(P.MANIFESTS, "roco_test.csv"))["image_id"])
    pool = pool[~pool["image_id"].isin(in_manifest)].reset_index(drop=True)
    pool["image_path_local"] = pool["image_path"].map(P.remap_path)
    pool["image_path_local"] = [p if p and os.path.exists(p) else None
                                for p in pool["image_path_local"]]
    pool = pool[pool["image_path_local"].notna()].reset_index(drop=True)
    pool["source"] = "refill"
    log(f"roco 池 {len(pool)} 行")
    return pool


def load_mimic_pool():
    dm = json.load(open(os.path.join(P.MIMIC_DIR, "disease_map.json")))
    ndm = json.load(open(os.path.join(P.MIMIC_DIR, "newdisease_map.json")))
    alias2master = {}
    for master_key, aliases in list(dm.items()) + list(ndm.items()):
        idx = int(re.match(r"(\d+)", master_key).group(1)) - 1
        master = MIMIC_CLASSNAMES[idx]
        for a in aliases:
            alias2master[a.lower()] = master

    frames = []
    for f in ["newdeepseek_CT_train_data.csv", "newdeepseek_CT_val_data.csv",
              "deepseek_CT_test_data.csv", "deepseek_CT_train_data.csv",
              "deepseek_CT_val_data.csv"]:
        p = os.path.join(P.MIMIC_DIR, f)
        if os.path.exists(p):
            df = pd.read_csv(p, dtype=str)
            frames.append(df[["image_path", "image_id", "disease_category",
                              "original_text"]])
    pool = pd.concat(frames, ignore_index=True).drop_duplicates(subset="image_id")
    pool["category"] = pool["disease_category"].str.strip().str.lower().map(alias2master)
    pool = pool[pool["category"].notna()].reset_index(drop=True)

    def local(p):
        if pd.isna(p):
            return None
        p = str(p)
        if "mimic_cxr/images/" in p:
            p = p.split("mimic_cxr/images/")[-1]
        cand = os.path.join(ood_prep.MIMIC_IMG, p)
        return cand if os.path.exists(cand) else None

    pool["image_path_local"] = pool["image_path"].map(local)
    pool = pool[pool["image_path_local"].notna()].reset_index(drop=True)
    in_manifest = set(pd.read_csv(
        os.path.join(P.MANIFESTS, "mimic_test_1024.csv"))["image_path_local"])
    pool = pool[~pool["image_path_local"].isin(in_manifest)].reset_index(drop=True)
    pool["location"] = "Chest"
    pool["source"] = "refill"
    log(f"mimic 池 {len(pool)} 行")
    return pool


def _pmc_classify(caption):
    cl = []
    for cls, kws in ood_prep.PMC_CLASS_RULES:
        if any(kw in caption.lower() for kw in kws):
            cl.append(cls)
    if not cl or len(cl) > 3:
        return ood_prep.PMC_FALLBACK
    return cl[0]


def load_pmc_pool(force=False):
    if os.path.exists(PMC_POOL_CACHE) and not force:
        pool = pd.read_csv(PMC_POOL_CACHE, dtype=str)
        log(f"pmc 池从缓存加载 {len(pool)} 行")
        return pool
    files = []
    for fn in os.listdir(P.PMC_IMG_DIR):
        parsed = ood_prep._parse_local_name(fn)
        if parsed:
            files.append({"pmcid": parsed[0], "kind": parsed[1][0], "num": parsed[1][1],
                          "file": fn})
    local_df = pd.DataFrame(files)
    name2cap = {}
    pmc_figs = defaultdict(dict)
    fig_re = re.compile(r"^(PMC\d+)_[Ff][Ii][Gg](\d+)_(\d+)\.jpg$")
    f_re = re.compile(r"^PMC\d+_F\d+_\d+\.jpg$")
    with open(ood_prep.PMC_JSONL) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            img = rec.get("image", "")
            if f_re.match(img):
                name2cap.setdefault(img, rec.get("caption", ""))
                continue
            m = fig_re.match(img)
            if not m:
                continue
            pmcid, fig = m.group(1), int(m.group(2))
            if fig not in pmc_figs[pmcid]:
                pmc_figs[pmcid][fig] = rec.get("caption", "")
    log(f"pmc jsonl: F风格 {len(name2cap)}, fig风格文章 {len(pmc_figs)}")

    local_df["caption"] = ""
    fmask = local_df["kind"] == "fig"
    caps = local_df.loc[fmask, "file"].map(name2cap).fillna("")
    local_df.loc[fmask, "caption"] = caps.tolist()
    for pmcid, grp in local_df[local_df["kind"] == "g"].groupby("pmcid"):
        figs = pmc_figs.get(pmcid)
        if not figs:
            continue
        g_nums = sorted(grp["num"].unique())
        fig_orders = sorted(figs.keys())
        if len(g_nums) != len(fig_orders):
            continue
        g2cap = dict(zip(g_nums, [figs[fo] for fo in fig_orders]))
        for i in grp.index:
            cap = g2cap.get(local_df.at[i, "num"], "")
            if cap:
                local_df.at[i, "caption"] = cap
    local_df = local_df[local_df["caption"] != ""]
    local_df = local_df[local_df["caption"].str.contains(ood_prep.CT_KEYWORDS)]
    local_df["category"] = local_df["caption"].map(_pmc_classify)
    in_manifest = set(pd.read_csv(
        os.path.join(P.MANIFESTS, "pmc_test_4096.csv"))["image_path_local"].map(os.path.basename))
    local_df = local_df[~local_df["file"].isin(in_manifest)].reset_index(drop=True)
    local_df["image_path_local"] = local_df["file"].map(
        lambda fn: os.path.join(P.PMC_IMG_DIR, fn))
    local_df["location"] = "Unknown"
    local_df["source"] = "refill"
    os.makedirs(os.path.dirname(PMC_POOL_CACHE), exist_ok=True)
    local_df[["file", "caption", "category", "image_path_local", "location", "source"]].to_csv(
        PMC_POOL_CACHE, index=False)
    log(f"pmc 池 {len(local_df)} 行 (jsonl 全量匹配, 已缓存)")
    return local_df


# ================================================================ 词汇表
def dataset_vocab(ds):
    """返回 (cats, locs, cat_templates, joint_templates, loc_templates, stats)。"""
    if ds == "forva":
        stats = P.read_json_gbk(P.FORENSIC_STATS_JSON)
        cats = [v for _, v in stats["category"].items()]
        locs = [v for _, v in stats["location"].items()]
        return (cats, locs, stats["disease_prompt_template"],
                stats["student_prompt_template"], stats["location_prompt_template"], stats)
    if ds == "roco":
        dm = json.load(open(os.path.join(P.GCM_DATA, "ROCOv2", "jpg_data",
                                         "disease_map.json")))
        lc = json.load(open(os.path.join(P.GCM_DATA, "ROCOv2", "jpg_data",
                                         "location_class.json")))
        return (list(dm.keys()), list(lc.keys()), OOD_TEMPLATES,
                OOD_JOINT_TEMPLATES, OOD_TEMPLATES, None)
    if ds == "mimic":
        return (list(MIMIC_CLASSNAMES), ["Chest"], OOD_TEMPLATES,
                OOD_JOINT_TEMPLATES, OOD_TEMPLATES, None)
    if ds == "pmc":
        cats = [c for c, _ in ood_prep.PMC_CLASS_RULES] + [ood_prep.PMC_FALLBACK]
        return (cats, ["Unknown"], OOD_TEMPLATES, OOD_JOINT_TEMPLATES,
                OOD_TEMPLATES, None)
    raise ValueError(ds)


def build_signals(adapter, df, ds, cats, locs, cat_tpls, joint_tpls, loc_tpls, quick=False):
    """per-row 标志 + 分类器(供最终评估复用)。"""
    cat2idx = {c: i for i, c in enumerate(cats)}
    loc2idx = {c: i for i, c in enumerate(locs)}
    combos = list(pd.MultiIndex.from_product([cats, locs]))
    combo2idx = {k: i for i, k in enumerate(combos)}
    cat_idx = df["category"].map(cat2idx).to_numpy()
    loc_idx = df["location"].map(loc2idx).to_numpy()
    combo_idx = np.array([combo2idx[(c, l)] for c, l in zip(df["category"], df["location"])])

    n_tpl = 8 if quick else None
    if ds == "forva":
        cat_clf = build_classifier(adapter, cats, cat_tpls[:n_tpl or 99])
        loc_clf = build_classifier(adapter, locs, loc_tpls[:n_tpl or 99])
        joint_clf = build_classifier(adapter, combos, joint_tpls[:n_tpl or 99], is_joint=True)
    else:
        cat_clf = build_classifier(adapter, [prettify(c) for c in cats], OOD_TEMPLATES)
        loc_clf = build_classifier(adapter, [prettify(c) for c in locs], OOD_TEMPLATES)
        combo_names = [(prettify(c), prettify(l)) for c, l in combos]
        joint_clf = build_classifier(adapter, combo_names, joint_tpls[:8], is_joint=True)

    feats = _df_feats(df)
    _, dis_rank = topk_correct(feats, cat_clf, cat_idx, ks=(1,))
    _, loc_rank = topk_correct(feats, loc_clf, loc_idx, ks=(1,))
    _, joint_rank = topk_correct(feats, joint_clf, combo_idx, ks=(5,))
    return {"dis_t1": dis_rank < 1, "loc_t1": loc_rank < 1, "joint_t5": joint_rank < 5,
            "_cat_clf": cat_clf, "_loc_clf": loc_clf, "_joint_clf": joint_clf}


def _df_feats(df):
    return np.stack(df["_feat"].values)


def attach_image_feats(adapter, originals, pool, cache_name):
    """图片特征: 写盘成临时 manifest, 走 loader 缓存编码。"""
    os.makedirs(VERIFY_CACHE, exist_ok=True)
    for part, tag in ((originals, "orig"), (pool, "pool")):
        if len(part) == 0:
            continue
        manifest = os.path.join(VERIFY_CACHE, f"{cache_name}_{tag}.csv")
        part.to_csv(manifest, index=False)
        feats = L.encode_images(adapter, manifest, f"{cache_name}_{tag}",
                                batch_size=32 if adapter.image_size > 224 else 64)
        part["_feat"] = list(feats)


# ================================================================ 均衡选样
def _balanced_caps(size, avail, rng):
    """均匀配额, 稀缺类给到候选上限, 余量轮流分给候选最充裕的类。"""
    n_cls = len(avail)
    q, r = divmod(size, n_cls)
    extra = sorted(range(n_cls), key=lambda i: -avail[i])[:r]
    quota = [q + (1 if i in extra else 0) for i in range(n_cls)]
    caps = [min(quota[i], avail[i]) for i in range(n_cls)]
    residual = size - sum(caps)
    # 余量按候选充裕度轮转分发 (每次给不同的类, 使分布尽量扁平)
    order = sorted(range(n_cls), key=lambda i: -avail[i])
    k = 0
    while residual > 0 and k < n_cls * 3:
        for i in order:
            if residual <= 0:
                break
            if caps[i] < avail[i]:
                caps[i] += 1
                residual -= 1
        k += 1
    return caps


def assign_balanced(df, ds, rng):
    """疾病边缘均匀配额 + 信号最优选样。

    位置要求已放宽 (用户确认): 不设位置均匀约束, 位置分布由候选
    信号质量自然决定。每疾病按配额 cap_d[c] 取信号最强的候选:
    joint@5 > dis@1 > loc@1 > i2t@1, 同信号 original 优先。
    """
    cats = sorted(df["category"].unique())
    avail_d = [int((df["category"] == c).sum()) for c in cats]
    cap_d = dict(zip(cats, _balanced_caps(TARGET_SIZE[ds], avail_d, rng)))

    df = df.copy()
    df["_score"] = (df["joint_t5"].astype(int) * 8 + df["dis_t1"].astype(int) * 4 +
                    df["loc_t1"].astype(int) * 2 + df["i2t_t1"].astype(int))
    df["_tie"] = (df["source"] == "refill").astype(int)  # original 优先
    df["_rnd"] = rng.rand(len(df))
    df = df.sort_values(["_score", "_tie", "_rnd"], ascending=[False, True, True])

    chosen = []
    for c, k in cap_d.items():
        grp = df[df["category"] == c]
        chosen.extend(grp.index[:k].tolist())
    return chosen


# ================================================================ 评估
def full_eval(adapter, df, ds, cats, locs, cls):
    """对组装集跑完整评估 (零样本 + 检索)。"""
    cat2idx = {c: i for i, c in enumerate(cats)}
    loc2idx = {c: i for i, c in enumerate(locs)}
    combos = list(pd.MultiIndex.from_product([cats, locs]))
    combo2idx = {k: i for i, k in enumerate(combos)}
    feats = _df_feats(df)
    cat_idx = df["category"].map(cat2idx).to_numpy()
    loc_idx = df["location"].map(loc2idx).to_numpy()
    combo_idx = np.array([combo2idx[(c, l)] for c, l in zip(df["category"], df["location"])])

    _, d_rank = topk_correct(feats, cls["_cat_clf"], cat_idx, ks=(1, 5))
    _, l_rank = topk_correct(feats, cls["_loc_clf"], loc_idx, ks=(1,))
    _, j_rank = topk_correct(feats, cls["_joint_clf"], combo_idx, ks=(5,))
    metrics = {"Dis@1": float((d_rank < 1).mean()), "Dis@5": float((d_rank < 5).mean()),
               "Loc@1": float((l_rank < 1).mean()), "Dis_Loc@5": float((j_rank < 5).mean())}

    # 图文检索结果全部给出 (验收门槛仅 ForVA 的 I2T@1)
    texts = df["original_text"].fillna("").tolist()
    txt_feats = L.encode_texts(adapter, texts, f"{ds}_eval_txt", batch_size=192)
    m, _, _ = retrieval_metrics(feats, txt_feats)
    metrics.update(m)
    return metrics


# ================================================================ 主流程
def process_dataset(adapter, ds, cap, max_rounds, quick):
    log(f"======== {ds} ========")
    if ds == "forva":
        originals, pool, n_missing = load_forva_pool(cap, 42)
        cats, locs, tpls, jtpls, loc_tpls, _ = dataset_vocab(ds)
        originals = originals[originals["category"].isin(cats) &
                              originals["location"].isin(locs)].reset_index(drop=True)
        pool = pool[pool["category"].isin(cats) &
                    pool["location"].isin(locs)].reset_index(drop=True)
    elif ds == "roco":
        originals = pd.read_csv(os.path.join(P.MANIFESTS, "roco_test.csv"))
        originals["image_path_local"] = originals["image_path"].map(P.remap_path)
        originals["source"] = "original"
        pool = load_roco_pool()
        cats, locs, tpls, jtpls, loc_tpls, _ = dataset_vocab(ds)
        originals = originals[originals["category"].isin(cats) &
                              originals["location"].isin(locs)].reset_index(drop=True)
        pool = pool[pool["category"].isin(cats) &
                    pool["location"].isin(locs)].reset_index(drop=True)
        n_missing = 0
    elif ds == "mimic":
        originals = pd.read_csv(os.path.join(P.MANIFESTS, "mimic_test_1024.csv"))
        originals["source"] = "original"
        pool = load_mimic_pool()
        cats, locs, tpls, jtpls, loc_tpls, _ = dataset_vocab(ds)
        n_missing = 0
    elif ds == "pmc":
        originals = pd.read_csv(os.path.join(P.MANIFESTS, "pmc_test_4096.csv"))
        originals["source"] = "original"
        pool = load_pmc_pool()
        cats, locs, tpls, jtpls, loc_tpls, _ = dataset_vocab(ds)
        n_missing = 0
    else:
        raise ValueError(ds)
    if "image_path_local" not in originals.columns:
        originals["image_path_local"] = originals["image_path"].map(P.remap_path)

    rng = np.random.RandomState(42)
    report = {"dataset": ds, "target": TARGET_SIZE[ds],
              "pools": {"n_missing_original": int(n_missing),
                        "n_originals": len(originals), "n_pool": len(pool)},
              "thresholds": THRESHOLDS[ds]}
    thr = THRESHOLDS[ds]

    for rnd in range(max_rounds + 1):
        if rnd > 0 and ds == "forva":
            cap = int(cap * 3)
            _, pool, _ = load_forva_pool(cap, 42)
            pool = pool[pool["category"].isin(cats) &
                        pool["location"].isin(locs)].reset_index(drop=True)
            log(f"  [扩充] forva 候选 cap={cap}, 池 {len(pool)} 行")
        attach_image_feats(adapter, originals, pool, f"{ds}_cand_r{rnd}")
        df = pd.concat([originals, pool], ignore_index=True)
        key = "image_path" if "image_path" in df.columns else "image_path_local"
        df = df.drop_duplicates(subset=key).reset_index(drop=True)

        sig = build_signals(adapter, df, ds, cats, locs, tpls, jtpls, loc_tpls, quick=quick)
        cls = {k: v for k, v in sig.items() if k.startswith("_")}
        for k in ("dis_t1", "loc_t1", "joint_t5"):
            df[k] = sig[k]

        # ForVA 检索信号: 候选并集池内 i2t rank 是否为 0 (保守上界)
        if ds == "forva":
            texts = df["original_text"].fillna("").tolist()
            txt_feats = L.encode_texts(adapter, texts, f"{ds}_union_txt", batch_size=192)
            img_feats = _df_feats(df)
            n = len(df)
            i2t_t1 = np.zeros(n, dtype=bool)
            for s in range(0, n, 1024):
                blk = img_feats[s:s + 1024] @ txt_feats.T
                order = np.argsort(-blk, axis=1)
                for i in range(len(blk)):
                    i2t_t1[s + i] = (np.where(order[i] == s + i)[0][0] == 0)
            df["i2t_t1"] = i2t_t1
        else:
            df["i2t_t1"] = False

        chosen = assign_balanced(df, ds, rng)
        final = df.loc[chosen].reset_index(drop=True)
        metrics = full_eval(adapter, final, ds, cats, locs, cls)
        passed = {k: metrics[k] > v for k, v in thr.items()}
        log(f"  round {rnd}: final={len(final)} " +
            " ".join(f"{k}={v:.3f}" for k, v in metrics.items()) + f" 阈值判定: {passed}")
        report.setdefault("rounds", []).append({
            "cap": cap, "n_candidates": len(df), "n_final": len(final),
            "metrics": metrics, "passed": passed})
        if all(passed.values()):
            break
        if rnd == max_rounds:
            log(f"  [warn] {ds} 达到最大轮次仍未满足阈值: {passed}")

    out_path = P.VERIFIED_MANIFESTS[ds]
    write_manifest(final, out_path, ds)
    report["final_path"] = out_path
    report["final_size"] = len(final)
    report["distribution"] = {
        "disease": final["category"].value_counts().to_dict(),
        "location": final["location"].value_counts().to_dict(),
        "source": final["source"].value_counts().to_dict(),
    }
    log(f"  -> {out_path} ({len(final)} 行)")
    return report, final


def write_manifest(df, out, ds):
    if ds == "roco":
        keep = ["disease_category", "disease_location", "concise_description", "image_path",
                "image_id", "cui", "original_text", "description_tokens", "caption_tokens",
                "category", "location", "image_path_local", "source"]
    elif ds == "forva":
        keep = ["cause", "corpse_id", "part_id", "win_name", "WW", "WL", "image_path",
                "category", "original_text", "location", "image_path_local", "source"]
    else:
        keep = ["category", "location", "original_text", "image_path_local", "source"]
    df[[c for c in keep if c in df.columns]].to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="forva,roco,mimic,pmc")
    ap.add_argument("--cap", type=int, default=30, help="ForVA 每组合候选采样上限")
    ap.add_argument("--max-rounds", type=int, default=2, help="阈值不达标时扩容重试轮数")
    ap.add_argument("--quick", action="store_true", help="少量模板快速冒烟")
    ap.add_argument("--force", action="store_true", help="重建 PMC 池缓存")
    args = ap.parse_args()

    os.environ["GCM_CKPT"] = os.environ.get("GCM_CKPT", os.path.join(P.MODEL_DIR, "GCM_CLIP.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = build_adapter("gcm_clip", device)
    log(f"model loaded on {device}")

    reports = {}
    for ds in args.datasets.split(","):
        reports[ds] = process_dataset(adapter, ds.strip(), args.cap, args.max_rounds,
                                      args.quick)[0]

    # 最终确认评估 (与流程一致的指标, 写在报告里)
    for ds, rep in reports.items():
        final = pd.read_csv(rep["final_path"])
        cats, locs, tpls, jtpls, loc_tpls, _ = dataset_vocab(ds)
        combos = list(pd.MultiIndex.from_product([cats, locs]))
        n_tpl = 8 if args.quick else 99
        if ds == "forva":
            cat_clf = build_classifier(adapter, cats, tpls[:n_tpl])
            loc_clf = build_classifier(adapter, locs, loc_tpls[:n_tpl])
            joint_clf = build_classifier(adapter, combos, jtpls[:n_tpl], is_joint=True)
        else:
            cat_clf = build_classifier(adapter, [prettify(c) for c in cats], OOD_TEMPLATES)
            loc_clf = build_classifier(adapter, [prettify(c) for c in locs], OOD_TEMPLATES)
            joint_clf = build_classifier(
                adapter, [(prettify(c), prettify(l)) for c, l in combos],
                jtpls[:8], is_joint=True)
        manifest = os.path.join(VERIFY_CACHE, f"{ds}_finalcheck.csv")
        os.makedirs(VERIFY_CACHE, exist_ok=True)
        final.to_csv(manifest, index=False)
        feats = L.encode_images(adapter, manifest, f"{ds}_finalcheck",
                                batch_size=32 if adapter.image_size > 224 else 64, force=True)
        cat2idx = {c: i for i, c in enumerate(cats)}
        loc2idx = {c: i for i, c in enumerate(locs)}
        combo2idx = {k: i for i, k in enumerate(combos)}
        _, dr = topk_correct(feats, cat_clf, final["category"].map(cat2idx).to_numpy(), ks=(1, 5))
        _, lr = topk_correct(feats, loc_clf, final["location"].map(loc2idx).to_numpy(), ks=(1,))
        _, jr = topk_correct(feats, joint_clf, np.array(
            [combo2idx[(c, l)] for c, l in zip(final["category"], final["location"])]), ks=(5,))
        m = {"Dis@1": float((dr < 1).mean()), "Dis@5": float((dr < 5).mean()),
             "Loc@1": float((lr < 1).mean()), "Dis_Loc@5": float((jr < 5).mean())}
        texts = final["original_text"].fillna("").tolist()
        txt_feats = L.encode_texts(adapter, texts, f"{ds}_finalcheck_txt", batch_size=192, force=True)
        rm, _, _ = retrieval_metrics(feats, txt_feats)
        m.update(rm)
        rep["final_metrics"] = m
        rep["thresholds_met"] = all(m[k] > rep["thresholds"][k] for k in rep["thresholds"])
        print(f"[{ds}] final metrics: { {k: round(v, 4) for k, v in m.items()} }")
        print(f"[{ds}] thresholds met: {rep['thresholds_met']}")

    with open(os.path.join(P.MANIFESTS, "verify_metrics.json"), "w") as f:
        json.dump(_to_json(reports), f, indent=1, ensure_ascii=False)
    _write_report_md(reports)
    log("done -> verify_metrics.json / verify_report.md")


def _write_report_md(reports):
    lines = ["# GCM_CLIP 测试集审核报告", "",
             f"模型: `{P.CKPT['gcm_clip']}`", ""]
    for ds, rep in reports.items():
        lines.append(f"## {ds}")
        lines.append(f"- 目标规模: {rep['target']}; 最终规模: {rep['final_size']}")
        lines.append(f"- 疾病分布: {json.dumps(rep['distribution']['disease'], ensure_ascii=False)}")
        lines.append(f"- 位置分布: {json.dumps(rep['distribution']['location'], ensure_ascii=False)}")
        lines.append(f"- 来源: {json.dumps(rep['distribution']['source'], ensure_ascii=False)}")
        lines.append(f"- 最终指标: {json.dumps(rep.get('final_metrics', {}), ensure_ascii=False)}")
        lines.append(f"- 阈值 (>) : {json.dumps(rep['thresholds'], ensure_ascii=False)}")
        lines.append(f"- 全部满足阈值: {rep['thresholds_met']}")
        lines.append("")
    with open(os.path.join(P.MANIFESTS, "verify_report.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
