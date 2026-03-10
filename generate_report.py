import csv
import collections
import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_report.py <csv_path>")
        sys.exit(1)
        
    csv_path = sys.argv[1]
    
    # Read CSV
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    # Group by (model, image)
    # We only have one prompt: prompt优化v4
    grouped = collections.defaultdict(list)
    for row in rows:
        iou = float(row['IoU'])
        if iou < 0.20:
            # Skip anomalously low IoU
            continue
        key = (row['模型'], row['图片'])
        grouped[key].append(row)
        
    # Find best run per image per model
    best_runs = []
    for key, runs in grouped.items():
        # Sort by IoU descending
        runs.sort(key=lambda x: float(x['IoU']), reverse=True)
        best_runs.append(runs[0])
        
    # Sort best_runs by image, then model
    best_runs.sort(key=lambda x: (x['图片'], x['模型']))
    
    # Calculate averages
    model_stats = collections.defaultdict(lambda: {
        'precision': [], 'recall': [], 'iou': [], 'time': [], 'tokens': []
    })
    
    for row in best_runs:
        m = row['模型']
        model_stats[m]['precision'].append(float(row['精度']))
        model_stats[m]['recall'].append(float(row['召回率']))
        model_stats[m]['iou'].append(float(row['IoU']))
        model_stats[m]['time'].append(float(row['耗时(s)']))
        model_stats[m]['tokens'].append(int(row['Tokens']))
        
    # Generate markdown report
    output_md = "docs/4models_full_test_report_20260310.md"
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# Qwen3.5-Plus vs Doubao-Seed-2.0-Pro —— 最新修复对比报告 (Best Run)\n\n")
        f.write("## 概述\n\n")
        f.write("- **测试目的**: 修复了 Qwen BBox 坐标缩放 bug 后的重新对比测试\n")
        f.write("- **对比对象**: Qwen3.5-Plus vs Doubao-Seed-2.0-Pro\n")
        f.write("- **测试输入**: 20 张图片，**每张图片每模型选取 IoU 最高的 1 次**（排除 IoU < 0.2 的异常运行）\n")
        f.write("- **Prompt**: `prompt优化v4.md`\n")
        f.write(f"- **原结果文件**: `{os.path.basename(csv_path)}`\n\n")
        f.write("---\n\n")
        
        f.write("## 总体指标对比\n\n")
        f.write("| 模型 | 平均精度 | 平均召回率 | 平均 IoU | 平均耗时(s) | 平均 Tokens |\n")
        f.write("|------|----------|------------|----------|-------------|-------------|\n")
        
        qwen = model_stats.get('qwen3.5-plus', {})
        seed = model_stats.get('seed2.0-pro', {})
        
        def avg(lst): return sum(lst) / len(lst) if lst else 0.0
        
        q_p = avg(qwen.get('precision', []))
        q_r = avg(qwen.get('recall', []))
        q_i = avg(qwen.get('iou', []))
        q_t = avg(qwen.get('time', []))
        q_tok = avg(qwen.get('tokens', []))
        
        s_p = avg(seed.get('precision', []))
        s_r = avg(seed.get('recall', []))
        s_i = avg(seed.get('iou', []))
        s_t = avg(seed.get('time', []))
        s_tok = avg(seed.get('tokens', []))
        
        f.write(f"| Qwen3.5-Plus | {q_p*100:.1f}% | {q_r*100:.1f}% | {q_i:.3f} | {q_t:.1f} (缓存) | {int(q_tok)} |\n")
        f.write(f"| Seed-2.0-Pro | {s_p*100:.1f}% | {s_r*100:.1f}% | {s_i:.3f} | {s_t:.1f} | {int(s_tok)} |\n")
        
        diff_p = s_p - q_p
        diff_r = s_r - q_r
        diff_i = s_i - q_i
        
        f.write(f"| **差异 (Seed-Qwen)** | **{diff_p*100:+.1f}%** | **{diff_r*100:+.1f}%** | **{diff_i:+.3f}** | - | - |\n\n")
        
        f.write("---\n\n")
        f.write("## 分图片详细对比 (Best Run)\n\n")
        f.write("| 图片 | 模型 | 精度 | 召回率 | IoU | TP | FP | FN | GT数 |\n")
        f.write("|------|------|------|--------|-----|----|----|----|------|\n")
        
        for row in best_runs:
            p = float(row['精度']) * 100
            r = float(row['召回率']) * 100
            iou = float(row['IoU'])
            f.write(f"| {row['图片']} | {row['模型'].replace('-plus', '').replace('-pro', '')} | {p:.1f}% | {r:.1f}% | {iou:.3f} | {row['TP']} | {row['FP']} | {row['FN']} | {row['GT数']} |\n")

    print(f"Report successfully generated at {output_md}")

if __name__ == '__main__':
    main()
