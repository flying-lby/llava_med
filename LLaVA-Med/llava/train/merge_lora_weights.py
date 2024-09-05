'''
Author: fly
Date: 2024-09-01 21:16:43
FilePath: /llava_med/LLaVA-Med/llava/train/merge_lora_weights.py
Description: 
'''
import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, default= "/srv/lby/llava_med/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, required=True, default="/home/lby/llava_med/LLaVA-Med/llava/checkpoints/llava-version1")
    parser.add_argument("--save-model-path", type=str, required=True, default= "/home/lby/llava_med/LLaVA-Med/llava/checkpoints/llava-mistral_finetuning")

    args = parser.parse_args()

    merge_lora(args)
