#!/bin/bash
###
 # @Author: fly
 # @Date: 2024-08-26 18:44:38
 # @FilePath: /llava_med/LLaVA-Med/llava/train/finetune_task.sh
 # @Description: 
### 

deepspeed train_mem.py \
    --deepspeed /home/lby/llava_med/LLaVA-Med/llava/train/zero3.json \
    --model_name_or_path /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
    --version v1 \
    --data_path /home/lby/llava_med/LLaVA-Med/llava/sft_data/classify_mimic_file.json \
    --image_folder /srv/lby/physionet.org/files/mimic-cxr-jpg/2.0.0/files \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/lby/llava_med/LLaVA-Med/llava/checkpoints/llava-version1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb