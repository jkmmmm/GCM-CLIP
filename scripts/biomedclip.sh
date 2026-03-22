cd biomedclip_finetuning/open_clip/src

## You can change the following parameters like the GPU devices, batch size, training data, epochs, and DHN-NCE loss parameters.

CUDA_VISIBLE_DEVICES=0 python3 -m open_clip_train.main \
    --batch-size 16 \
    --accum-freq 2 \
    --workers 4 \
    --report-to tensorboard \
    --save-frequency 5 \
    --logs="logs" \
    --dataset-type csv \
    --csv-separator="," \
    --train-data /root/autodl-tmp/data/forensic_CT_train.csv \
    --val-data /root/autodl-tmp/data/forensic_CT_valid.csv \
    --csv-img-key image_path \
    --csv-caption-key original_text \
    --csv-disease-category category \
    --csv-disease-location location \
    --force-CMCLIP \
    --CMCLIP-loss \
    --teacher-update-freq 306 \
    --config /root/autodl-tmp/data/forensic_CT_statistics.json \
    --lr=0 \
    --wd=0.1 \
    --warmup 10 \
    --epochs=50 \
    --model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --pretrained /root/autodl-fs/clip_log/2025_10_17-00_47_54-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_200-j_4-p_amp/checkpoints/epoch_latest.pt \
    --save-best \
    --delete-previous-checkpoint \