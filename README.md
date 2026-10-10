# GCM-CLIP Training/Evaluation Code and Test-Set Data Package

## Directory Structure

```
code_model/
├── README.md                          This document
├── biomedclip_finetuning/open_clip/   GCM-CLIP training code (modified open_clip)
│   ├── scripts/
│   │   └── run_gcm_clip_main_local.sh   Main-model training launcher (as used on the server, A800)
│   ├── src/
│   │   ├── open_clip/                   Model definitions (model.py contains GCM/DSD/ABC;
│   │   │                                loss.py contains GCMCLIPLoss — multi-task supervised
│   │   │                                contrastive loss + GOS gradient orthogonalization)
│   │   └── open_clip_train/             Training entry points (main/train/data/params/…;
│   │                                    only modules inside the training import closure are kept)
│   ├── pyproject.toml / requirements*.txt / LICENSE
├── figure_pipeline/                   Evaluation code
│   ├── run_fig2b.py                     Image–text retrieval evaluation (I2T/T2I R@1/5/10 → Fig. 2c)
│   ├── run_fig2a.py                     Zero-shot classification evaluation (99-template ensemble + Dis/Loc/Joint)
│   ├── paths.py / loader.py / model_registry.py / verify_and_refill.py / _compat/
├── config/
│   └── forensic_CT_statistics.json    Statistics configuration required by the GCM loss (--config; GBK-encoded)
├── model/
│   └── best_model.pt                  GCM-CLIP main-model weights (best checkpoint of the retraining
│                                      run gcm_clip; 761 MB — exceeds GitHub's per-file size limit, so
│                                      it is not hosted here; reviewers may request it from the authors)
└── data/                              Not hosted here (see "Data Privacy"); kept locally / shared privately
    ├── forva_test_4096_verified.csv   Paper test set (per SI Table S4)
    └── dataset_jpg/…                  All virtual-autopsy images referenced by the test set (4,025 images)
```

## Training (server configuration)

Entry point: `python -m open_clip_train.main`. See `scripts/run_gcm_clip_main_local.sh` for the full argument list.

Paper protocol (`run_gcm_clip_main_local.sh`):

- Effective batch 8,192 (core batch 128 × 64 accumulation steps), AMP; AdamW with lr 1e-4, β₁=0.9, β₂=0.999, ε=1e-8, weight decay 0.1 (weight matrices only), 300 warmup steps, cosine annealing to zero, no gradient clipping.
- Two-stage schedule over T=200 epochs: epochs 0–49 optimize only the global image–text contrastive loss; the full GCM objective (explicit location/health supervision + implicit DSD/ABC supervision combined via GOS) activates from epoch 50 (0-based; the 51st epoch) via `--implicit-start-epoch 50`.
- ABC uses ACCE (adaptive cluster counts selected from the distortion elbow on DSD-refined features, guided by both explicit anchors: disease category + anatomical location). The fixed-count ablation variant (Supplementary Table S7 Panel D) is reproduced with `--fixed-cluster-counts 8 16`.
- DSD/ABC consume the Discovery section of the report: `--csv-caption-key original_text --csv-caption-section discovery` (section header is stripped from the full report; reports already stored section-only pass through unchanged).

## Evaluation

```bash
export PYTHONPATH=<repo>/biomedclip_finetuning/open_clip/src:<repo>
export GCM_CKPT=<repo>/model/best_model.pt
# Image–text retrieval (I2T/T2I R@1/5/10)
python -m figure_pipeline.run_fig2b --models gcm_clip
# Zero-shot classification (Loc@1 / Dis@1 / Dis_Loc@5; 99 templates)
python -m figure_pipeline.run_fig2a --models gcm_clip [--quick] [--skip-robustness]
```

## Data Privacy

`data/dataset_jpg/` contains virtual-autopsy (post-mortem CT) images, which have been de-identified and constitute sensitive forensic case material. It is intended for paper review / internal research-group use only and must not be distributed publicly (consistent with the manuscript's Data Availability statement).
**For this reason `data/` is not hosted in this repository**; the images and CSV are kept locally or shared privately only.
The `model/best_model.pt` weights are likewise not hosted in the repository.
