'''
Author: fly
Date: 2024-09-09 13:35:10
FilePath: /llava_med/LLaVA-Med/llava/data_analysis/extract_feature.py
Description: 
'''
import torch
from PIL import Image
import argparse
import json
from tqdm import tqdm
import random
import click
import os
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
import math
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from matplotlib import pyplot as plt
from tqdm import trange

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class LinearProbing(torch.nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LinearProbing, self).__init__()
        self.fc = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPProbing(torch.nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(MLPProbing, self).__init__()
        self.fc1 = torch.nn.Linear(feature_dim, feature_dim)
        self.fc2 = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def get_feature(model_id,num_chunks,chunk_idx,question_file,answers_file,image_folder,conv_mode):
   
    # processor = AutoProcessor.from_pretrained(model_id)
    # model = LlavaForConditionalGeneration.from_pretrained(
    #     model_id, device_map="auto", torch_dtype=torch.bfloat16
    # )

    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_id,
            model_base=None,
            model_name='llava-med-v1.5-mistral-7b'
    )
    
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    questions = get_chunk(questions, num_chunks, chunk_idx)
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = os.path.join(image_folder,line["image"])
        
        qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]
       
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        inputs = {
            "input_ids": input_ids,  # 手动处理的文本输入
            "images": image_tensor  # 手动处理的图像输入
        }
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Image Tensor shape: {image_tensor.shape}")
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            last_features = hidden_states[:, -1, :].unsqueeze(1)
            mean_features = hidden_states.mean(dim=1).unsqueeze(1)
            features = torch.cat([last_features, mean_features], dim=1)
        all_features.append(features.cpu())
        
    all_features = torch.cat(all_features, dim=0)
    
    return all_features
    

def main(dataset, model_name, probe, split, feature_type, n_epochs, all_feature):
    
    data = [json.loads(line) for line in open(f"../data/{dataset}.jsonl")]
    random.seed(1234)
    random.shuffle(data)
    classes = json.load(open(f"../data/{dataset}_classes.json"))

    labels = []
    for item in data:
        # 将字符串形式的标签转换为列表，去掉空格
        item_labels = [label.strip() for label in item["label"].split(',')]
        
        # 创建一个与 classes 等长的零向量
        label_indices = [0] * len(classes)
        
        # 遍历分割后的标签列表，并标记对应的索引为 1
        for label in item_labels:
            if label in classes:
                label_indices[classes.index(label)] = 1
            else:
                raise ValueError(f"标签 {label} 不在 classes 列表中")
        
        # 保存该样本的标签索引
        item["label_index"] = label_indices
        labels.append(label_indices)

    # 将 labels 列表转换为 PyTorch 张量
    labels = torch.tensor(labels)


    features = all_feature.float()
    print(features.shape)

    train_idxs = [i for i in range(len(data)) if data[i]["split"] == "train"]
    test_idxs = [i for i in range(len(data)) if data[i]["split"] == split]

    feature_idx = None
    if feature_type == "last":
        feature_idx = 0
    elif feature_type == "avg":
        feature_idx = 1
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    train_features = features[train_idxs, feature_idx]
    test_features = features[test_idxs, feature_idx]
    train_labels = labels[train_idxs].float()  # 标签为 float
    test_labels = labels[test_idxs].float()

    print(
        train_features.shape, test_features.shape, train_labels.shape, test_labels.shape
    )

    if probe == "linear":
        model = LinearProbing(len(train_features[0]), len(classes)).cuda()
    elif probe == "mlp":
        model = MLPProbing(len(train_features[0]), len(classes)).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()  # 多标签分类任务，使用二元交叉熵

    bsz = 512
    accs = []

    for epoch in trange(n_epochs):
        # 训练阶段
        for i in range(0, len(train_features), bsz):
            optimizer.zero_grad()
            output = model(train_features[i : i + bsz].cuda())
            loss = criterion(output, train_labels[i : i + bsz].cuda())
            loss.backward()
            optimizer.step()

        # 验证阶段，不需要梯度计算
        with torch.no_grad():
            eval_bsz = 512

            # 训练集预测
            train_preds = []
            for i in range(0, len(train_features), eval_bsz):
                output = model(train_features[i : i + eval_bsz].cuda())
                pred = torch.sigmoid(output).cpu()  # 用 sigmoid 将 logits 转化为概率
                train_preds.append(pred)
            train_preds = torch.cat(train_preds)
            
            # 测试集预测
            test_preds = []
            for i in range(0, len(test_features), eval_bsz):
                output = model(test_features[i : i + eval_bsz].cuda())
                pred = torch.sigmoid(output).cpu()
                test_preds.append(pred)
            test_preds = torch.cat(test_preds)

            # 设置阈值，将预测概率转化为 0/1 标签
            threshold = 0.5
            train_preds_bin = (train_preds > threshold).float()
            test_preds_bin = (test_preds > threshold).float()

            # 计算每个标签的准确率
            train_acc_per_class = (train_preds_bin == train_labels).float().mean(dim=0).tolist()
            test_acc_per_class = (test_preds_bin == test_labels).float().mean(dim=0).tolist()

            accs.append((train_acc_per_class, test_acc_per_class))

    # 打印每个标签的准确率
    for idx, class_name in enumerate(classes):
        print(f"Class {class_name}: Train Accuracy: {train_acc_per_class[idx]:.4f}, Test Accuracy: {test_acc_per_class[idx]:.4f}")

    # 绘制每个 epoch 的平均准确率变化图
    mean_train_acc = [torch.tensor(train_acc).mean().item() for train_acc, _ in accs]
    mean_test_acc = [torch.tensor(test_acc).mean().item() for _, test_acc in accs]

    plt.plot(mean_train_acc, label="Mean Train Accuracy")
    plt.plot(mean_test_acc, label="Mean Test Accuracy")
    plt.legend()

    # 保存准确率图和模型
    output_prefix = f"probe_outputs/{dataset}_{model_name}_{probe}_{split}_{feature_type}"
    plt.savefig(f"{output_prefix}.png")
    torch.save([accs, model.state_dict()], f"{output_prefix}.pt")


@click.command()
@click.option("--model_id", default="/srv/lby/llava_med/llava-med-v1.5-mistral-7b")
@click.option("--data_path", default="./data/eval/Chest-X-ray.jsonl")
@click.option("--num-chunks", default=1)
@click.option("--chunk-idx", default=0)
@click.option("--image-folder", default="/srv/lby")
@click.option("--conv-mode", default="vicuna_v1")
@click.option("--question-file", default="./data/eval/test_prompt/Chest-X-ray_llava_val.jsonl")
@click.option("--answers-file", default="./data/eval/test_prompt/Chest-X-ray_llava_val_ans.jsonl")
@click.option("--class_path", default="./data/eval/Chest-X-ray_classes.json")
@click.option("--seed", default=1234)
@click.option("--output_path", default="outputs")
@click.option("--batch_size", default=2)
@click.option("--prompt", default=None)
def entry(model_id, data_path, num_chunks, chunk_idx, image_folder, conv_mode, question_file, answers_file, class_path, seed, output_path, batch_size, prompt):
   
    all_feature = get_feature(model_id, num_chunks, chunk_idx, question_file, answers_file, image_folder, conv_mode)
    main(model_id, data_path, class_path, seed, output_path, batch_size, prompt, all_feature)
 
if __name__ == "__main__":
    entry()