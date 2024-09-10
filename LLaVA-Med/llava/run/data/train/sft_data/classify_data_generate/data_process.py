'''
Author: fly
Date: 2024-08-23 13:46:00
FilePath: /llava_med/LLaVA-Med/llava/data_process.py
Description: 
'''
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
import numpy as np
from tqdm import tqdm

MIMIC_REPORT_COL = "Report Impression"
MIMIC_XH_REPORT_COL = "xinhuo"
MIMIC_PATH_COL = "Path"


# 加载CSV文件
csv_file_path = '/home/lby/llava_med/LLaVA-Med/other_data/modified_file.csv'
df = pd.read_csv(csv_file_path)

# 新增一列用于存放去除乱码后的caption
df['caption'] = ""

to_remove = []

# 遍历数据集中的每一行
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    captions = ""

    # 检查并拼接报告内容
    if isinstance(row[MIMIC_REPORT_COL], str):
        captions += row[MIMIC_REPORT_COL]
    if isinstance(row[MIMIC_XH_REPORT_COL], str):
        captions += row[MIMIC_XH_REPORT_COL]

    # 如果报告内容为空，将该行标记为删除
    if len(captions) == 0:
        to_remove.append(idx)
        continue

    # 使用空格替换换行符
    captions = captions.replace("\n", " ")

    # 去除乱码，保留原始句子
    clean_caption = captions.replace("\ufffd\ufffd", " ")

    # 保存到新列
    df.at[idx, 'caption'] = clean_caption

    # 分割句子
    splitter = re.compile("[0-9]+\.")
    captions = splitter.split(clean_caption)
    captions = [point.split(".") for point in captions]
    captions = [sent for point in captions for sent in point]

    valid_sentences = []
    for cap in captions:
        if len(cap) == 0:
            continue

        tokenizer = RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(cap.lower())

        # 过滤无效句子
        if len(tokens) <= 1:
            continue

        valid_sentences.append(" ".join(tokens))

    # 如果该行没有有效的句子，将该行标记为删除
    if len(valid_sentences) == 0:
        to_remove.append(idx)

# 从数据集中删除空报告或无效句子的行
df.drop(to_remove, inplace=True)

# 保存清理后的数据集
df.to_csv('/home/lby/llava_med/LLaVA-Med/other_data/cleaned_file.csv', index=False)

print("共删除了 %d 行数据。" % len(to_remove))
print("数据清理完成，清理后的数据集已保存。")
