#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os

csv_path = "/Users/bytedance/PycharmProjects/VLM-parking-labeling-multiPics/outputs/csv/model_prompt_comparison_20260309_191544.csv"

data = {}

with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    print(f"表头: {reader.fieldnames}")
    for row in reader:
        model = row['模型']
        image = row['图片']
        iou = float(row['IoU'])
        
        if image not in data:
            data[image] = {'qwen': [], 'seed2': []}
        
        if model == 'qwen3.5-plus':
            data[image]['qwen'].append(iou)
        elif model == 'seed2.0-pro':
            data[image]['seed2'].append(iou)

results = []
for image, vals in data.items():
    if len(vals['qwen']) == 0 or len(vals['seed2']) == 0:
        continue
    
    avg_qwen = sum(vals['qwen']) / len(vals['qwen'])
    avg_seed2 = sum(vals['seed2']) / len(vals['seed2'])
    diff = abs(avg_qwen - avg_seed2)
    
    results.append((image, avg_qwen, avg_seed2, diff))

results.sort(key=lambda x: x[3], reverse=True)

print("Qwen 与 Doubao Seed2.0 结果差别最大的图片（按 IoU 平均差值排序）：")
print("-" * 100)
print(f"{'图片':<10} {'Qwen 平均 IoU':<15} {'Seed2 平均 IoU':<18} {'差值':<10}")
print("-" * 100)

for r in results:
    print(f"{r[0]:<10} {r[1]:<15.4f} {r[2]:<18.4f} {r[3]:<10.4f}")

print("\n" + "=" * 100)
print(f"差别最大的图片: {results[0][0]}, 平均 IoU 差值: {results[0][3]:.4f}")
