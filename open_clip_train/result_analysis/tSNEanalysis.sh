cd biomedclip_finetuning/open_clip/src

## You can change the following parameters like the GPU devices, batch size, training data, epochs, and DHN-NCE loss parameters.

CUDA_VISIBLE_DEVICES=0 python3 -m open_clip_train.result_analysis.tSNEanalysis \
    --model-name0 CLIP \
    --model-name1 CLIP_explicit_imlicit_8_16_32_supervised_2 \
    --checkpoint /root/autodl-fs/clip_log/2025_08_29-15_19_36-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/checkpoints/epoch_300.pt \
    --base-checkpoint /root/autodl-fs/clip_log/2025_11_26-23_26_51-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.0001-b_64-j_4-p_amp/checkpoints/best_model.pt \
    --class-names "category" \
    --feature-type 'image' \
    --model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --base-model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --force-CMCLIP \
    --config autodl-tmp/data/forensic_CT_statistics.json \
    --val-data /root/autodl-tmp/data/ROCOv2/jpg_data/selected_4096_category_balanced.csv \
    --csv-img-key image_path \
    --csv-caption-key original_text \
    --csv-disease-category disease_category \
    --csv-disease-location disease_location \
    --csv-separator="," \
    --dataset-type csv \
    --batch-size 32 \
    --workers 4 \
    --perplexity 50 \
    --n-iter 10000 \
    --random-state 42 \
    --output-dir /root/MedCLIP-SAMv2-main/biomedclip_finetuning/open_clip/src/result/t_SNE \
    