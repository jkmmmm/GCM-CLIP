#!/usr/bin/env bash
# GCM-CLIP training launcher (single run, no auto-retry).
#   train: forva_train_clean_full.csv / val: forva_val_4096_diverse.csv (--no-resplit)
#   Paper protocol: batch 128 x 64 accumulation = 8192, lr 1e-4, AdamW (beta2 0.999,
#   eps 1e-8), warmup 300 steps, cosine to zero over T=200 epochs; two-stage schedule:
#   epochs 0-49 optimize only the global contrastive loss, the full GCM objective
#   (explicit location/health + implicit DSD/ABC + GOS) activates from epoch 50
#   (0-based; the 51st epoch) via --implicit-start-epoch 50.
#   ABC uses ACCE (adaptive cluster counts from the distortion elbow); pass
#   --fixed-cluster-counts 8 16 to reproduce the fixed-k ablation variant.
#   DSD/ABC consume the Discovery section of the report (--csv-caption-section discovery).
set -u
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PY=/root/miniconda3/envs/gcm/bin/python
SRC=/root/gcm_clip_retrain_20260914/biomedclip_finetuning/open_clip/src
LOG_ROOT=$SRC/logs

PRETRAINED=/root/gcm_clip_retrain_20260914/biomedclip_finetuning/open_clip/src/logs/gcm_clip/checkpoints/epoch_latest.pt
TRAIN=/root/gcm_clip_retrain_20260914/output/figure_data/manifests/forva_train_clean_full.csv
VAL=/root/gcm_clip_retrain_20260914/output/figure_data/manifests/forva_val_4096_diverse.csv
STATS=/home/mmmmjk/Virtual_Anatomical_Dataset/data/forensic_CT_statistics.json
NAME=gcm_clip

cd "$SRC" || exit 1
mkdir -p "$LOG_ROOT"

CUDA_VISIBLE_DEVICES=0 "$PY" -m open_clip_train.main \
  --batch-size 128 --accum-freq 64 --workers 16 \
  --report-to tensorboard --logs "$LOG_ROOT" --name "$NAME" \
  --dataset-type csv --csv-separator "," \
  --train-data "$TRAIN" --val-data "$VAL" --no-resplit \
  --csv-img-key image_path --csv-caption-key original_text \
  --csv-caption-section discovery \
  --csv-disease-category category --csv-disease-location location \
  --force-CMCLIP --CMCLIP-loss --config "$STATS" \
  --lr 0.0001 --wd 0.1 --warmup 300 --epochs 200 \
  --beta2 0.999 --eps 1e-8 \
  --dsd-norm-space --dsd-components 32 \
  --implicit-start-epoch 50 \
  --model "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" \
  --pretrained "$PRETRAINED" \
  --seed 0 --save-best --save-most-recent --save-frequency 5 --delete-previous-checkpoint \
  >> "$LOG_ROOT/$NAME.out.log" 2>&1
