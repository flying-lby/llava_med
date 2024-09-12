import argparse
import json
import math
import os

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm
import random

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    # Initialize a counter for processed questions
    counter = 0
    max_count = 300  # Set the limit to 5000

    for line in tqdm(questions):
        if counter >= max_count:
            print(f"Reached {max_count} generated answers. Exiting.")
            break

        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        ############## Enable This Part for ImageWikiQA ##############
        # conv.append_message(conv.roles[1], "Let's think step by step.")
        # prompt = conv.get_prompt().replace("</s>", "")
        # print(prompt)

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
                    # "model_id": model_name,
                    # "metadata": {},
                }
            )
            + "\n"
        )
        ans_file.flush()

        # Increment the counter after each answer is processed
        counter += 1

    ans_file.close()
    
    
def get_acc():
    # 读取数据
    output_path = './data/eval/test_prompt/Chest-X-ray_llava_val_ans.jsonl'
    answers = [json.loads(line) for line in open(output_path)]

    disease_list = ['fibrosis', 'edema', 'pneumothorax', 'cardiomegaly', 'atelectasis', 'nodule', 'emphysema', 'no finding', 'mass', 'pleural_thickening', 'effusion', 'infiltration', 'pneumonia', 'hernia', 'consolidation']
    print(f"Total number of answers: {len(answers)}")
    # 随机选择 1000 行
    random.shuffle(answers)
    selected_answers = answers[:100]

    # 初始化变量
    correct_predictions = 0
    total_predictions = len(selected_answers)
    error_count = 0
    error_question_ids = []

    # 遍历每个 answer，提取 labels 和预测类别
    for item in selected_answers:
        # 获取标签（label），labels 可能包含多个标签，以逗号或其他符号分隔
        labels = ["-".join(item["question_id"].split("-")[1:])]  # 获取 label
        labels = [label.lower() for label in labels]  # 转为小写以方便比较

        # 获取预测的 text
        text = item["text"].lower()

        # 尝试在 text 中找到疾病列表中的元素作为预测结果
        predicted_categories = [disease for disease in disease_list if disease in text]

        if predicted_categories:
            predicted_category = predicted_categories[0]  # 假设预测类别为匹配到的第一个疾病
        else:
            # 如果无法提取预测类别，统计为出错
            error_count += 1
            error_question_ids.append(item["question_id"])
            continue  # 跳过此项

        # 检查预测类别是否在 labels 列表中
        if any(predicted_category in label for label in labels):
            correct_predictions += 1
        else:
            # 如果预测错误，统计出错信息
            error_count += 1
            error_question_ids.append(item["question_id"])

    # 计算准确率
    accuracy = (correct_predictions / total_predictions) * 100

    # 输出结果
    print(f"Total labels: {total_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Number of errors: {error_count}")
    print(f"Error question IDs: {error_question_ids}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/srv/lby/llava_med/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="./data/eval/test_prompt/Chest-X-ray_llava_val.jsonl")
    parser.add_argument("--answers-file", type=str, default="./data/eval/test_prompt/Chest-X-ray_llava_val_ans.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
    get_acc()