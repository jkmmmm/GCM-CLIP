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

## Evaluation

```bash
export PYTHONPATH=<repo>/biomedclip_finetuning/open_clip/src:<repo>
export GCM_CKPT=<repo>/model/best_model.pt
# Image–text retrieval (I2T/T2I R@1/5/10)
python -m figure_pipeline.run_fig2b --models gcm_clip
# Zero-shot classification (Loc@1 / Dis@1 / Dis_Loc@5; 99 templates)
python -m figure_pipeline.run_fig2a --models gcm_clip [--quick] [--skip-robustness]
```

## Test Set and Correspondence with the Paper

`forva_test_4096_verified.csv` (= data source of SI Table S4):
- 4,096 image–text pairs; 40 classes with 62–106 pairs each, matching Table S4 class-by-class (40/40)
- 8 anatomical regions (Abdomen/Chest/Head/Lower Leg/Neck/Pelvis/Unknown/Upper Leg)
- 16 cause-of-death scenarios; 208 bodies; reports average 37.1 words
- Note: among the 4,096 pairs, 3 rows are duplicates sharing the same image (4,093 unique images); counted per the paper's "pairs" convention

**Data revision (2026-10-02)**: A file-by-file verification found that 34 referenced images were stored under over-long paths and could not be recovered locally; after substituting same-class images, their texts no longer matched the substitute images, so a total of 68 rows (including rows sharing those images) were removed (cerebral_hemorrhage_death −62, coronary_heart_disease_death −6).
The current version contains **4,028 image–text pairs / 4,025 unique images** with all 40 classes present, but the per-class counts of cerebral_hemorrhage_death and coronary_heart_disease_death no longer match SI Table S4; the paper's main-table metrics are based on the 4,096-pair version.
The removed rows are fully archived in `_removed_rows.csv` (not hosted here).

## Local Reproduction Results

| Metric | Paper (Fig. 2c) | Local reproduction |
|---|---|---|
| Loc@1 | 0.724 | **0.7239** |
| Dis@1 | 0.676 | **0.6758** |
| Dis_Loc@5 | 0.782 | **0.7822** |
| I2T R@1 | 0.565 | **0.5654** |
| I2T R@5 | 0.806 | **0.8059** |
| I2T R@10 | 0.894 | **0.8936** |

All metrics match the paper (differences ≤ 0.0005, arising from rounding when per-sample rankings tie).
Reproduction commands are in the section above; result files are written to `output/figure_data/fig2a|fig2b/gcm_clip/`
(mapped through the `/home/mmmmjk` directory junction).

## Data Privacy

`data/dataset_jpg/` contains virtual-autopsy (post-mortem CT) images, which have been de-identified and constitute sensitive forensic case material. It is intended for paper review / internal research-group use only and must not be distributed publicly (consistent with the manuscript's Data Availability statement).
**For this reason `data/` is not hosted in this repository**; the images and CSV are kept locally or shared privately only.
The `model/best_model.pt` weights are likewise not hosted in the repository.
