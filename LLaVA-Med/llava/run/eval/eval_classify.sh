python -m llava.eval.model_vqa \
    --model-path /srv/lby/llava_med/checkpoints/llava-mistral_finetuning/ \
    --question-file /home/lby/llava_med/LLaVA-Med/data/eval/test_prompt/Chest-X-ray_llava_val.jsonl \
    --image-folder "" \
    --answers-file /home/lby/llava_med/LLaVA-Med/data/eval/test_fine_tuning/Chest-X-ray_llava_val_ans.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1