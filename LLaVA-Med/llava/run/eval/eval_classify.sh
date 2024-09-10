python -m llava.run.eval.model_vqa \
    --model-path /srv/lby/llava_med/checkpoints/llava-mistral_finetuning/ \
    --question-file ./data/eval/test_prompt/Chest-X-ray_llava_val.jsonl \
    --image-folder "" \
    --answers-file ./data/eval/test_prompt/Chest-X-ray_llava_val_ans.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1