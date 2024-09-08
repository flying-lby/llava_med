'''
Author: fly
Date: 2024-08-21 21:29:37
FilePath: /llava_med/LLaVA-Med/llava/sft_data/classify_data_generate/data.py
Description: 
'''
import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor


# # 定义旧的和新的路径前缀
# old_prefix = "/defaultShare/MIMIC-CXR/files/"
# new_prefix = "/srv/lby/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"

# # 读取 CSV 文件
# csv_file_path = '/home/lby/llava_med/LLaVA-Med/other_data/MIMIC/MIMIC/mimic-cxr-label-LLM_report-xinhuo-chexpertformat.csv'  # 替换为你的 CSV 文件路径
# df = pd.read_csv(csv_file_path)

# # 修改路径前缀
# df['Path'] = df['Path'].str.replace(old_prefix, new_prefix, regex=False)

# # 保存修改后的 CSV 文件
# output_csv_file_path = '/home/lby/llava_med/LLaVA-Med/other_data/modified_file.csv'  # 替换为保存新 CSV 文件的路径
# df.to_csv(output_csv_file_path, index=False)

# print(f"修改后的 CSV 文件已保存到 {output_csv_file_path}")
# print(df['Path'].head(5))

import os

# 读取CSV文件
csv_file_path = '/home/lby/llava_med/LLaVA-Med/other_data/cleaned_file.csv'  # 替换为实际的CSV文件路径
df = pd.read_csv(csv_file_path)

# 创建一个列表用于存放JSON数据
json_data = []

disease_columns = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "Pleural Effusion", "Pleural Other", 
    "Pneumonia", "Pneumothorax"
]
prefix = "/srv/lby/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"

# 遍历CSV的每一行并转换为JSON格式
for index, row in df.iterrows():
    
    # 跳过 "Lateral" 方向的图像
    if row["Frontal/Lateral"] == "Lateral":
        continue
    
    # 使用列表推导式提取值为1的疾病标签
    label = [disease for disease in disease_columns if row[disease] == 1]

    # 如果没有疾病特征为1，设置label为["Normal"]
    if not label:
        label = ["Normal"]
   
    # 创建conversation列表
    conversations = [
        {
            "from": "human",
            "value": "You are now acting as a knowledgeable radiologist. Please analyze the provided chest x-ray image and identify the most appropriate category or categories from the given list: ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Normal'].\n<image>"
        },
        {
            "from": "gpt",
            "value": ", ".join(label) 
        }
    ]
    
    
    # conversations = [
    #     {
    #         "from": "human",
    #         "value": "You now assume the role of a knowledgeable radiologist. What medical diagnosis can you make from this image?\n<image>"
    #     },
    #     {
    #         "from": "gpt",
    #         "value": row["caption"]
    #     },
    #     {
    #         "from": "human",
    #         "value": f"""You now assume the role of a knowledgeable radiologist. Please analyze the provided medical image in conjunction with the following medical report:'{row['caption']}' Based on this information, determine the most appropriate disease category or categories present in the patient. The diagnosis can involve one or more conditions. You must select the relevant categories from the following list: ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Normal']"""
    #     },
    #     {
    #         "from": "gpt",
    #         "value": ", ".join(label) 
    #     }
    # ]

    path = row["Path"]

    entry = {
        "id": int(row["index"]),
        "image": path[len(prefix):] if path.startswith(prefix) else path,
        "conversations": conversations
    }
    
    json_data.append(entry)

# 检查目录路径是否存在，如果不存在则创建
json_file_path = '/home/lby/llava_med/LLaVA-Med/llava/sft_data/classify_mimic_file.json'  # 替换为实际的输出JSON文件路径
json_dir = os.path.dirname(json_file_path)
if not os.path.exists(json_dir):
    os.makedirs(json_dir)

# 保存为JSON文件
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

print(f"JSON文件已保存到: {json_file_path}")
print(json_data[:5])


# # 读取JSON文件
# with open(json_file_path, 'r') as file:
#     data = json.load(file)

# # 获取前5个条目
# first_5_items = data[:5]

# # 打印前5个条目
# for item in first_5_items:
#     print(json.dumps(item, indent=4))


# csv_file_path = '/home/lby/llava_med/LLaVA-Med/other_data/cleaned_file.csv'  # 替换为实际的CSV文件路径
# df = pd.read_csv(csv_file_path)
# first_caption = df.loc[0, 'caption']

# # 打印第一行数据的完整 'caption' 内容
# print(first_caption)