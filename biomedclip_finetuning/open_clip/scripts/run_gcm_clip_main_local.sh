#!/usr/bin/env bash
# GCM-CLIP training launcher (single run, no auto-retry).
#   train: forva_train_clean_full.csv / val: forva_val_4096_diverse.csv (--no-resplit)
#   GCM protocol: explicit location/health + implicit DSD/ABC + contrastive,
#   M=32, dsd-norm-space, implicit gating active from epoch 0 (--implicit-start-epoch -1).
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
  --csv-disease-category category --csv-disease-location location \
  --force-CMCLIP --CMCLIP-loss --config "$STATS" \
  --lr 0.0001 --wd 0.1 --warmup 200 --epochs 50 \
  --dsd-norm-space --dsd-components 32 \
  --implicit-start-epoch -1 \
  --model "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" \
  --pretrained "$PRETRAINED" \
  --seed 0 --save-best --save-most-recent --save-frequency 5 --delete-previous-checkpoint \
  >> "$LOG_ROOT/$NAME.out.log" 2>&1
