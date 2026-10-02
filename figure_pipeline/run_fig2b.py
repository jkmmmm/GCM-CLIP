"""Fig2b 跨模态检索评估（ForVA 4096 对, I2T/T2I Accuracy@1/5/10）+ bootstrap CI。

复用 fig2a 的图像特征缓存; 文本特征 = manifest original_text。
输出: fig2b/<model>/retrieval_metrics.json (+ per_sample.npz, sim_matrix.npz)
      fig2b/summary_all.csv  fig2b/stats_tests.csv

用法: python -m figure_pipeline.run_fig2b [--models m1,m2]
"""
import argparse
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
from figure_pipeline.run_fig2a import bootstrap_ci, mcnemar_p


def retrieval_metrics(img_feats, txt_feats, ks=(1, 5, 10)):
    """返回 ({'i2t@k': acc}, i2t_gt_rank, t2i_gt_rank)。GT = 同索引配对。"""
    sim = img_feats @ txt_feats.T
    n = sim.shape[0]
    i2t_rank = np.empty(n, dtype=np.int64)
    t2i_rank = np.empty(n, dtype=np.int64)
    # 分块 argrank
    for s in range(0, n, 1024):
        blk = sim[s:s + 1024]
        order = np.argsort(-blk, axis=1)
        i2t_rank[s:s + 1024] = [np.where(order[i] == s + i)[0][0] for i in range(len(blk))]
    for s in range(0, n, 1024):
        blk = sim.T[s:s + 1024]        # text 查询块 × 全部图像
        order = np.argsort(-blk, axis=1)
        t2i_rank[s:s + 1024] = [np.where(order[i] == s + i)[0][0] for i in range(len(blk))]
    res = {}
    for k in ks:
        res[f"I2T@{k}"] = float((i2t_rank < k).mean())
        res[f"T2I@{k}"] = float((t2i_rank < k).mean())
    return res, i2t_rank, t2i_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=None)
    ap.add_argument("--save-sim", action="store_true")
    args = ap.parse_args()
    models = args.models.split(",") if args.models else P.model_list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = P.VERIFIED_MANIFESTS["forva"]
    df = pd.read_csv(manifest)
    texts = df["original_text"].fillna("").tolist()

    rows, per_sample = [], {}
    for name in models:
        out_json = os.path.join(P.FIG_DATA, "fig2b", name, "retrieval_metrics.json")
        if os.path.exists(out_json):
            m = json.load(open(out_json))
            print(f"[skip] {name}")
        else:
            adapter = build_adapter(name, device)
            print(f"=== {name} ===")
            img_feats = L.encode_images(adapter, manifest, "forva_verified",
                                        batch_size=32 if adapter.image_size > 224 else 64)
            txt_feats = L.encode_texts(adapter, texts, "forva_verified",
                                       batch_size=256 if name != "medsiglip" else 128)
            m, i2t_rank, t2i_rank = retrieval_metrics(img_feats, txt_feats)
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            json.dump(m, open(out_json, "w"), indent=1)
            np.savez(os.path.join(P.FIG_DATA, "fig2b", name, "per_sample.npz"),
                     i2t_at1=i2t_rank < 1, i2t_at5=i2t_rank < 5, i2t_at10=i2t_rank < 10,
                     t2i_at1=t2i_rank < 1, t2i_at5=t2i_rank < 5, t2i_at10=t2i_rank < 10)
            if args.save_sim:
                np.savez_compressed(os.path.join(P.FIG_DATA, "fig2b", name, "sim_matrix.npz"),
                                    sim=(img_feats @ txt_feats.T).astype(np.float16))
            del adapter
            torch.cuda.empty_cache()
        rows.append({"model": name, "display": P.DISPLAY_NAME.get(name, name), **m})

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(P.FIG_DATA, "fig2b", "summary_all.csv"), index=False)

    # CI + 配对检验
    test_rows = []
    have = [m for m in models if os.path.exists(
        os.path.join(P.FIG_DATA, "fig2b", m, "per_sample.npz"))]
    if "gcm_clip" in have:
        ours = np.load(os.path.join(P.FIG_DATA, "fig2b", "gcm_clip", "per_sample.npz"))
        for m in have:
            if m == "gcm_clip":
                continue
            base = np.load(os.path.join(P.FIG_DATA, "fig2b", m, "per_sample.npz"))
            for key in ours.files:
                if key not in base.files:
                    continue
                a, b = ours[key].astype(bool), base[key].astype(bool)
                lo, hi = bootstrap_ci(a)
                p, _, _ = mcnemar_p(a, b)
                test_rows.append({"metric": key, "baseline": m,
                                  "ours_acc": float(a.mean()), "base_acc": float(b.mean()),
                                  "ours_ci": f"[{lo:.3f},{hi:.3f}]",
                                  "delta": float(a.mean() - b.mean()),
                                  "mcnemar_p": p, "ours_better": bool(a.mean() > b.mean())})
    if test_rows:
        pd.DataFrame(test_rows).to_csv(
            os.path.join(P.FIG_DATA, "fig2b", "stats_tests.csv"), index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
