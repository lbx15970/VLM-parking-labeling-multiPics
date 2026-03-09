
import csv
import datetime
import statistics
from collections import defaultdict

# 读取 CSV
csv_path = 'outputs/csv/model_prompt_comparison_20260309_191544.csv'

data = []
with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 转换数值
        for k in ['精度', '召回率', 'IoU', '预测数', 'GT数', 'TP', 'FP', 'FN', '耗时(s)', 'Tokens']:
            try:
                row[k] = float(row[k])
            except ValueError:
                row[k] = 0.0
        data.append(row)

# 按模型分组
models = defaultdict(list)
for row in data:
    models[row['模型']].append(row)

# 按图片分组
images = defaultdict(lambda: defaultdict(list))
unique_images = set()
for row in data:
    images[row['图片']][row['模型']].append(row)
    unique_images.add(row['图片'])

# 计算总体平均值
overall = {}
for model, rows in models.items():
    stats = {
        '精度': statistics.mean([r['精度'] for r in rows]),
        '召回率': statistics.mean([r['召回率'] for r in rows]),
        'IoU': statistics.mean([r['IoU'] for r in rows]),
        '耗时(s)': statistics.mean([r['耗时(s)'] for r in rows]),
        'Tokens': statistics.mean([r['Tokens'] for r in rows]),
        'TP': sum([r['TP'] for r in rows]),
        'FP': sum([r['FP'] for r in rows]),
        'FN': sum([r['FN'] for r in rows]),
        'GT数': sum([r['GT数'] for r in rows]),
        'count': len(rows)
    }
    overall[model] = stats

# 生成报告
report = []
report.append(f"# Qwen3.5-Plus vs Doubao-Seed-2.0-Pro —— Prompt 优化 v4 测试对比报告")
report.append(f"")
report.append(f"## 概述")
report.append(f"")
report.append(f"- **对比对象**: Qwen3.5-Plus vs Doubao-Seed-2.0-Pro")
report.append(f"- **测试输入**: {len(unique_images)} 张图片，每张图片每模型运行 **3 次**，5 线程并发")
report.append(f"- **Prompt**: `prompt优化v4.md`")
report.append(f"- **测试时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
report.append(f"- **结果文件**: `{csv_path}`")
report.append(f"")
report.append(f"---")
report.append(f"")
report.append(f"## 结论摘要")
report.append(f"")

qwen_stats = overall.get('qwen3.5-plus', {})
seed_stats = overall.get('seed2.0-pro', {})

if not qwen_stats or not seed_stats:
    print("Error: Missing model data")
    exit(1)

# 比较逻辑
better_precision = "Qwen3.5" if qwen_stats['精度'] > seed_stats['精度'] else "Seed-2.0"
better_recall = "Qwen3.5" if qwen_stats['召回率'] > seed_stats['召回率'] else "Seed-2.0"
better_iou = "Qwen3.5" if qwen_stats['IoU'] > seed_stats['IoU'] else "Seed-2.0"

report.append(f"- **总体表现**: {better_precision} 在精度上略胜一筹，{better_recall} 在召回率上领先。")
report.append(f"- **Qwen3.5-Plus**: 平均精度 {qwen_stats['精度']:.1%}，平均召回率 {qwen_stats['召回率']:.1%}，平均 IoU {qwen_stats['IoU']:.3f}")
report.append(f"- **Seed-2.0-Pro**: 平均精度 {seed_stats['精度']:.1%}，平均召回率 {seed_stats['召回率']:.1%}，平均 IoU {seed_stats['IoU']:.3f}")
report.append(f"- **Token 消耗**: Qwen3.5 ({qwen_stats['Tokens']:.0f}) vs Seed-2.0 ({seed_stats['Tokens']:.0f})")
report.append(f"- **耗时**: Qwen3.5 ({qwen_stats['耗时(s)']:.1f}s) vs Seed-2.0 ({seed_stats['耗时(s)']:.1f}s)")
report.append(f"")
report.append(f"---")
report.append(f"")
report.append(f"## 总体指标对比")
report.append(f"")
report.append(f"| 模型 | 平均精度 | 平均召回率 | 平均 IoU | 平均耗时(s) | 平均 Tokens |")
report.append(f"|------|----------|------------|----------|-------------|-------------|")
report.append(f"| Qwen3.5-Plus | {qwen_stats['精度']:.1%} | {qwen_stats['召回率']:.1%} | {qwen_stats['IoU']:.3f} | {qwen_stats['耗时(s)']:.1f} | {qwen_stats['Tokens']:.0f} |")
report.append(f"| Seed-2.0-Pro | {seed_stats['精度']:.1%} | {seed_stats['召回率']:.1%} | {seed_stats['IoU']:.3f} | {seed_stats['耗时(s)']:.1f} | {seed_stats['Tokens']:.0f} |")
diff_precision = seed_stats['精度'] - qwen_stats['精度']
diff_recall = seed_stats['召回率'] - qwen_stats['召回率']
diff_iou = seed_stats['IoU'] - qwen_stats['IoU']
diff_time = seed_stats['耗时(s)'] - qwen_stats['耗时(s)']
diff_tokens = seed_stats['Tokens'] - qwen_stats['Tokens']
report.append(f"| **差异 (Seed-Qwen)** | **{diff_precision:+.1%}** | **{diff_recall:+.1%}** | **{diff_iou:+.3f}** | **{diff_time:+.1f}** | **{diff_tokens:+.0f}** |")
report.append(f"")
report.append(f"---")
report.append(f"")
report.append(f"## 分图片详细对比 (3次平均)")
report.append(f"")
report.append(f"| 图片 | 模型 | 精度 | 召回率 | IoU | TP | FP | FN | GT数 |")
report.append(f"|------|------|------|--------|-----|----|----|----|------|")

sorted_images = sorted(list(unique_images))

for img in sorted_images:
    for model in ['qwen3.5-plus', 'seed2.0-pro']:
        rows = images[img][model]
        if not rows:
            continue
        
        avg_precision = statistics.mean([r['精度'] for r in rows])
        avg_recall = statistics.mean([r['召回率'] for r in rows])
        avg_iou = statistics.mean([r['IoU'] for r in rows])
        avg_tp = statistics.mean([r['TP'] for r in rows])
        avg_fp = statistics.mean([r['FP'] for r in rows])
        avg_fn = statistics.mean([r['FN'] for r in rows])
        gt_count = int(rows[0]['GT数'])
        
        model_name = "Qwen3.5" if model == 'qwen3.5-plus' else "Seed-2.0"
        report.append(f"| {img} | {model_name} | {avg_precision:.1%} | {avg_recall:.1%} | {avg_iou:.3f} | {avg_tp:.1f} | {avg_fp:.1f} | {avg_fn:.1f} | {gt_count} |")
    # report.append(f"| | | | | | | | | |") # 空行分隔

# 输出到文件
output_path = f"docs/qwen3.5_vs_seed2.0_prompt_v4_test_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print(f"Report generated: {output_path}")
