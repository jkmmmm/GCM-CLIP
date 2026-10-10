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
│                                      it is not hosted here; available on HuggingFace)
└── data/                              Not hosted here (see "Data and Model Weights"); available on HuggingFace
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

## Data and Model Weights

`data/dataset_jpg/` contains virtual-autopsy (post-mortem CT) images, which have been de-identified and constitute sensitive forensic case material, intended for paper review / research use only (consistent with the manuscript's Data Availability statement). For this reason the test data and the `model/best_model.pt` weights are not hosted in this repository.
Both are available at **https://huggingface.co/mmmmmjk/GCM_CLIP**.
