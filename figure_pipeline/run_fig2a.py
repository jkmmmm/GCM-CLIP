"""Fig2a 零样本分类评估（6+ 模型 × 4 数据集）+ 审稿增强。

- ForVA: joint(40x8)/disease(40)/location(8) 分类器, 各 99 模板集成
  指标: Loc@1, Dis@1, Dis_Loc@5 (另存 top2/5/10)
- OOD (ROCOv2/MIMIC/PMC): 通用 CT 模板集成, Cat@1 (+Loc@1 for ROCO)
- 审稿增强: bootstrap 95% CI、ours vs 基线 McNemar + Holm 校正、
  单模板提示词鲁棒性（ForVA 逐模板 Dis@1/Loc@1）
输出: output/figure_data/fig2a/<model>/zero_shot_metrics.json + per_sample.npz
      fig2a/summary_all.csv  stats_tests.csv  prompt_robustness.csv

用法: python -m figure_pipeline.run_fig2a [--models m1,m2] [--quick]
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figure_pipeline import paths as P
from figure_pipeline import loader as L
from figure_pipeline.model_registry import build_adapter

OOD_TEMPLATES = [
    "A chest CT image showing {c}.",
    "A computed tomography image showing {c}.",
    "A medical CT scan displaying {c}.",
    "A radiology CT image of {c}.",
    "A CT scan revealing {c}.",
    "An axial CT image showing {c}.",
    "A clinical CT image demonstrating {c}.",
    "A CT image indicating {c}.",
]

MIMIC_CLASSNAMES = [
    "Pulmonary Infection", "Interstitial Lung Disease", "Pulmonary Neoplasm",
    "Emphysema and COPD", "Atelectasis", "Pleural Effusion and Pleural Disease",
    "Mediastinal Lesion", "Cardiovascular Abnormality", "Chest Wall and Bone Lesion",
    "Thyroid Lesion", "Diaphragmatic Abnormality and Hernia", "Pulmonary Calcification",
    "Pulmonary Hemorrhage and Contusion", "Pulmonary Embolism", "Bronchial Lesion",
    "Postoperative and Postradiation Change", "Abdominal Lesion with Thoracic Involvement",
    "Benign Pulmonary Nodule and Cyst", "Unclassified Lesion", "Normal or Uncommon Finding",
]


def prettify(cls):
    """ROCO 风格类名 -> 提示词用自然名。"""
    s = cls.replace("_", " ")
    s = s.replace("(", " (").replace(")", ")")
    return " ".join(s.split())


def build_classifier(adapter, items, templates, is_joint=False, batch=512):
    """items: 类名或 (f,b) 元组列表; 返回 [D, C] L2 归一化列。"""
    cols = []
    for i in range(0, len(items), max(1, batch // max(1, len(templates)))):
        chunk = items[i:i + max(1, batch // max(1, len(templates)))]
        texts = []
        for it in chunk:
            for tpl in templates:
                texts.append(tpl.format(f=it[0], b=it[1]) if is_joint else tpl.format(c=it))
        emb = adapter.encode_text(texts).cpu()
        emb = emb.reshape(len(chunk), len(templates), -1).mean(dim=1)
        emb = emb / emb.norm(dim=1, keepdim=True)
        cols.append(emb.T)
    return torch.cat(cols, dim=1).numpy()


def topk_correct(feats, clf, targets, ks=(1, 2, 5, 10)):
    """feats [N,D] 归一化, clf [D,C], targets [N]; 返回 ({k: 正确数}, gt 排名数组)。"""
    sim = feats @ clf                      # [N, C]
    order = np.argsort(-sim, axis=1)
    rank_of_gt = np.empty(len(targets), dtype=np.int64)
    for i, t in enumerate(targets):
        pos = np.where(order[i] == t)[0]
        rank_of_gt[i] = pos[0] if len(pos) else clf.shape[1]
    return {k: int((rank_of_gt < k).sum()) for k in ks}, rank_of_gt


def bootstrap_ci(correct: np.ndarray, n_boot=1000, seed=42, alpha=0.05):
    rng = np.random.RandomState(seed)
    N = len(correct)
    accs = correct[rng.randint(0, N, size=(n_boot, N))].mean(axis=1)
    return float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))


def mcnemar_p(a: np.ndarray, b: np.ndarray):
    """a, b: per-sample 正确 bool。双侧精确二项检验 (b01 vs b10)。"""
    from scipy.stats import binomtest
    b01 = int((~a & b).sum())
    b10 = int((a & ~b).sum())
    if b01 + b10 == 0:
        return 1.0, b01, b10
    p = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue * 1.0  # 近似双侧
    return float(p), b01, b10


def eval_forva(adapter, manifest, stats, out_dir, quick=False):
    cats = [v for _, v in stats["category"].items()]      # 40
    locs = [v for _, v in stats["location"].items()]      # 8
    cat2idx = {c: i for i, c in enumerate(cats)}
    loc2idx = {c: i for i, c in enumerate(locs)}
    combos = list(itertools.product(cats, locs))
    combo2idx = {c: i for i, c in enumerate(combos)}

    feats = L.encode_images(adapter, manifest, "forva_verified",
                            batch_size=32 if adapter.image_size > 224 else 64)
    df = pd.read_csv(manifest)
    cat_t = df["category"].map(cat2idx).to_numpy()
    loc_t = df["location"].map(loc2idx).to_numpy()
    combo_t = np.array([combo2idx[(c, l)] for c, l in zip(df["category"], df["location"])])

    n_tpl = 8 if quick else None
    joint_clf = build_classifier(adapter, combos, stats["student_prompt_template"][:n_tpl or 99], True)
    cat_clf = build_classifier(adapter, cats, stats["disease_prompt_template"][:n_tpl or 99])
    loc_clf = build_classifier(adapter, locs, stats["location_prompt_template"][:n_tpl or 99])

    jres, jrank = topk_correct(feats, joint_clf, combo_t)
    cres, crank = topk_correct(feats, cat_clf, cat_t)
    lres, lrank = topk_correct(feats, loc_clf, loc_t)
    N = len(df)
    per_sample = {
        "joint_top5": (jrank < 5), "joint_top1": (jrank < 1),
        "dis_top1": (crank < 1), "dis_top5": (crank < 5),
        "loc_top1": (lrank < 1), "loc_top5": (lrank < 5),
    }
    metrics = {
        "Loc@1": lres[1] / N, "Loc@5": lres[5] / N,
        "Dis@1": cres[1] / N, "Dis@5": cres[5] / N,
        "Dis_Loc@5": jres[5] / N, "Dis_Loc@1": jres[1] / N,
    }
    return metrics, per_sample, (feats, df, cats, locs, cat2idx, loc2idx)


def eval_ood(adapter, manifest, class_col, classes, name, loc_col=None, loc_classes=None):
    feats = L.encode_images(adapter, manifest, name,
                            batch_size=32 if adapter.image_size > 224 else 64)
    df = pd.read_csv(manifest)
    cls2idx = {c: i for i, c in enumerate(classes)}
    keep = df[class_col].isin(cls2idx).to_numpy()
    df, feats = df[keep].reset_index(drop=True), feats[keep]
    cat_clf = build_classifier(adapter, [prettify(c) for c in classes], OOD_TEMPLATES)
    cat_t = df[class_col].map(cls2idx).to_numpy()
    cres, crank = topk_correct(feats, cat_clf, cat_t)
    out = {"Cat@1": cres[1] / len(df), "Cat@5": cres[5] / len(df)}
    per_sample = {"cat_top1": crank < 1}
    if loc_col and loc_classes:
        l2i = {c: i for i, c in enumerate(loc_classes)}
        loc_clf = build_classifier(adapter, [prettify(c) for c in loc_classes], OOD_TEMPLATES)
        lres, lrank = topk_correct(feats, loc_clf, df[loc_col].map(l2i).to_numpy())
        out["Loc@1"] = lres[1] / len(df)
        per_sample["loc_top1"] = lrank < 1
    return out, per_sample, len(df)


def prompt_robustness(adapter, feats, df, stats, cats, locs, cat2idx, loc2idx, max_templates=99):
    """ForVA 逐单模板 Dis@1 / Loc@1（提示词鲁棒性, R2-b1）。"""
    rows = []
    cat_t = df["category"].map(cat2idx).to_numpy()
    loc_t = df["location"].map(loc2idx).to_numpy()
    for ti, tpl in enumerate(stats["disease_prompt_template"][:max_templates]):
        loc_tpl = stats["location_prompt_template"][ti % len(stats["location_prompt_template"])]
        cat_clf = build_classifier(adapter, cats, [tpl])
        loc_clf = build_classifier(adapter, locs, [loc_tpl])
        _, crank = topk_correct(feats, cat_clf, cat_t, ks=(1,))
        _, lrank = topk_correct(feats, loc_clf, loc_t, ks=(1,))
        rows.append({"template_idx": ti, "Dis@1": float((crank < 1).mean()),
                     "Loc@1": float((lrank < 1).mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=None)
    ap.add_argument("--quick", action="store_true", help="少量模板快速冒烟")
    ap.add_argument("--skip-robustness", action="store_true")
    args = ap.parse_args()
    models = args.models.split(",") if args.models else P.model_list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = P.read_json_gbk(P.FORENSIC_STATS_JSON)
    forva_manifest = P.VERIFIED_MANIFESTS["forva"]

    all_rows, rob_rows = [], []
    per_sample_store = {}
    for name in models:
        out_json = os.path.join(P.FIG_DATA, "fig2a", name, "zero_shot_metrics.json")
        if os.path.exists(out_json) and not args.quick:
            data = json.load(open(out_json))
            print(f"[skip] {name} 已有结果")
        else:
            adapter = build_adapter(name, device)
            print(f"=== {name} ({P.DISPLAY_NAME.get(name, name)}) ===")
            m_forva, ps_forva, ctx = eval_forva(
                adapter, forva_manifest, stats, None, quick=args.quick)
            result = {"forva": m_forva}
            per_sample = {f"forva/{k}": v for k, v in ps_forva.items()}

            roco_manifest = P.VERIFIED_MANIFESTS["roco"]
            if os.path.exists(roco_manifest):
                roco = pd.read_csv(roco_manifest, usecols=["category", "location"])
                m_roco, ps_roco, n_roco = eval_ood(
                    adapter, roco_manifest, "category", sorted(roco["category"].unique()),
                    "roco_verified", loc_col="location", loc_classes=sorted(roco["location"].unique()))
                result["roco"] = m_roco | {"n_test": n_roco}
                per_sample.update({f"roco/{k}": v for k, v in ps_roco.items()})

            mimic_manifest = P.VERIFIED_MANIFESTS["mimic"]
            if os.path.exists(mimic_manifest):
                m_mimic, ps_mimic, n_mimic = eval_ood(
                    adapter, mimic_manifest, "category", MIMIC_CLASSNAMES, "mimic_verified")
                result["mimic"] = m_mimic | {"n_test": n_mimic}
                per_sample.update({f"mimic/{k}": v for k, v in ps_mimic.items()})

            pmc_manifest = P.VERIFIED_MANIFESTS["pmc"]
            if os.path.exists(pmc_manifest):
                pmc = pd.read_csv(pmc_manifest, usecols=["category"])
                m_pmc, ps_pmc, n_pmc = eval_ood(
                    adapter, pmc_manifest, "category", sorted(pmc["category"].unique()), "pmc_verified")
                result["pmc"] = m_pmc | {"n_test": n_pmc}
                per_sample.update({f"pmc/{k}": v for k, v in ps_pmc.items()})

            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            json.dump(result, open(out_json, "w"), indent=1)
            np.savez(os.path.join(P.FIG_DATA, "fig2a", name, "per_sample.npz"), **per_sample)

            if not args.skip_robustness and not args.quick:
                feats, df, cats, locs, cat2idx, loc2idx = ctx
                rb = prompt_robustness(adapter, feats, df, stats, cats, locs,
                                       cat2idx, loc2idx, max_templates=20 if args.quick else 99)
                rb.insert(0, "model", name)
                rob_rows.append(rb)
            data = result
            del adapter
            torch.cuda.empty_cache()

        row = {"model": name, "display": P.DISPLAY_NAME.get(name, name)}
        for ds in ["forva", "roco", "mimic", "pmc"]:
            for k, v in data.get(ds, {}).items():
                row[f"{ds}:{k}"] = v
        all_rows.append(row)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(P.FIG_DATA, "fig2a", "summary_all.csv"), index=False)
    if rob_rows:
        pd.concat(rob_rows).to_csv(
            os.path.join(P.FIG_DATA, "fig2a", "prompt_robustness.csv"), index=False)

    # ---- 统计检验: bootstrap CI + McNemar (ours vs 各基线) + Holm ----
    test_rows = []
    have = [m for m in models if os.path.exists(
        os.path.join(P.FIG_DATA, "fig2a", m, "per_sample.npz"))]
    if "gcm_clip" in have:
        ours = np.load(os.path.join(P.FIG_DATA, "fig2a", "gcm_clip", "per_sample.npz"))
        for m in have:
            if m == "gcm_clip":
                continue
            base = np.load(os.path.join(P.FIG_DATA, "fig2a", m, "per_sample.npz"))
            for key in ours.files:
                if key not in base.files or len(ours[key]) != len(base[key]):
                    continue
                a, b = ours[key].astype(bool), base[key].astype(bool)
                if a.mean() == 0 and b.mean() == 0:
                    continue
                lo, hi = bootstrap_ci(a)
                p, b01, b10 = mcnemar_p(a, b)
                test_rows.append({
                    "metric": key, "baseline": m, "ours_acc": float(a.mean()),
                    "base_acc": float(b.mean()), "ours_ci": f"[{lo:.3f},{hi:.3f}]",
                    "delta": float(a.mean() - b.mean()), "mcnemar_p": p,
                    "ours_better": bool(a.mean() > b.mean())})
    if test_rows:
        tf = pd.DataFrame(test_rows)
        # Holm 校正（按 metric 分组, 逐组 p 升序乘以递减系数）
        def holm(ps):
            ps = np.asarray(ps, dtype=float)
            order_idx = np.argsort(ps)
            m = len(ps)
            adj = np.empty(m)
            running = 0.0
            for rank, idx in enumerate(order_idx):
                val = (m - rank) * ps[idx]
                running = max(running, val)
                adj[idx] = min(1.0, running)
            return adj
        tf["holm_p"] = tf.groupby("metric")["mcnemar_p"].transform(
            lambda s: pd.Series(holm(s.values), index=s.index))
        tf.to_csv(os.path.join(P.FIG_DATA, "fig2a", "stats_tests.csv"), index=False)

    print(summary.to_string(index=False)[:3000])


if __name__ == "__main__":
    main()
