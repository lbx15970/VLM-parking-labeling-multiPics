#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import statistics

csv_path = "/Users/bytedance/PycharmProjects/VLM-parking-labeling-multiPics/outputs/csv/model_prompt_comparison_20260309_191544.csv"

data = []
with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for k in ['精度', '召回率', 'IoU', '预测数', 'GT数', 'TP', 'FP', 'FN', '耗时(s)', 'Tokens']:
            try:
                row[k] = float(row[k])
            except ValueError:
                row[k] = 0.0
        data.append(row)

# 按图片分组
image_data = {}
for row in data:
    img = row['图片']
    model = row['模型']
    if img not in image_data:
        image_data[img] = {}
    if model not in image_data[img]:
        image_data[img][model] = []
    image_data[img][model].append(row)

# 计算每张图片的平均差异
results = []
for img, models in image_data.items():
    if img == '00005.jpg':
        continue
    
    if 'qwen3.5-plus' not in models or 'seed2.0-pro' not in models:
        continue
    
    qwen_rows = models['qwen3.5-plus']
    seed_rows = models['seed2.0-pro']
    
    qwen_iou = statistics.mean([r['IoU'] for r in qwen_rows])
    seed_iou = statistics.mean([r['IoU'] for r in seed_rows])
    iou_diff = abs(qwen_iou - seed_iou)
    
    qwen_precision = statistics.mean([r['精度'] for r in qwen_rows])
    seed_precision = statistics.mean([r['精度'] for r in seed_rows])
    
    qwen_recall = statistics.mean([r['召回率'] for r in qwen_rows])
    seed_recall = statistics.mean([r['召回率'] for r in seed_rows])
    
    results.append({
        'image': img,
        'qwen_iou': qwen_iou,
        'seed_iou': seed_iou,
        'iou_diff': iou_diff,
        'qwen_precision': qwen_precision,
        'seed_precision': seed_precision,
        'qwen_recall': qwen_recall,
        'seed_recall': seed_recall
    })

# 按 IoU 差异排序
results.sort(key=lambda x: x['iou_diff'], reverse=True)

print("除了 00005.jpg 以外，Qwen 与 Seed2.0 结果差异最大的图片：\n")
print("=" * 130)
print(f"{'排名':<5} {'图片':<10} {'Qwen 平均 IoU':<15} {'Seed2 平均 IoU':<18} {'IoU 差值':<12} {'Qwen 精度':<12} {'Seed2 精度':<12} {'Qwen 召回率':<12} {'Seed2 召回率':<12}")
print("=" * 130)

for i, r in enumerate(results[:5], 1):
    print(f"{i:<5} {r['image']:<10} {r['qwen_iou']:<15.4f} {r['seed_iou']:<18.4f} {r['iou_diff']:<12.4f} "
          f"{r['qwen_precision']:<12.1%} {r['seed_precision']:<12.1%} {r['qwen_recall']:<12.1%} {r['seed_recall']:<12.1%}")

print("\n" + "=" * 130)
print(f"\n差异第二的图片: {results[0]['image']}, 平均 IoU 差值: {results[0]['iou_diff']:.4f}")
print(f"差异第三的图片: {results[1]['image']}, 平均 IoU 差值: {results[1]['iou_diff']:.4f}")
