#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv

csv_path = "/Users/bytedance/PycharmProjects/VLM-parking-labeling-multiPics/outputs/csv/model_prompt_comparison_20260309_191544.csv"

print("00005.jpg 详细测试结果：")
print("-" * 120)
print(f"{'模型':<15} {'轮次':<6} {'IoU':<8} {'精度':<8} {'召回率':<8} {'TP':<4} {'FP':<4} {'FN':<4}")
print("-" * 120)

with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['图片'] == '00005.jpg':
            model = row['模型']
            run = row['轮次']
            iou = float(row['IoU'])
            precision = float(row['精度'])
            recall = float(row['召回率'])
            tp = int(row['TP'])
            fp = int(row['FP'])
            fn = int(row['FN'])
            print(f"{model:<15} {run:<6} {iou:<8.4f} {precision:<8.4f} {recall:<8.4f} {tp:<4} {fp:<4} {fn:<4}")
