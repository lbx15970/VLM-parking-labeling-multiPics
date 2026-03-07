"""
多模型 + 多提示词对比测试脚本
支持 Doubao Seed 1.8、Seed 2.0 Pro 和 Qwen3-VL-Plus 的对比测试
结果文件命名: {模型}_{prompt名称}_{时间}.csv
"""
import requests
import json
import re
import os
import sys
import time
import csv

import concurrent.futures
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from collections import defaultdict
from runners.seed18_runner import Seed18Runner
from runners.seed20_runner import Seed20Runner
from runners.qwen_runner import QwenRunner

# ======================== 配置区 ========================

NUM_RUNS = 5  # 每个提示词每张图测试次数
IOU_THRESHOLD = 0.5
CALL_TIMEOUT_SECONDS = 200

# 脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# ------------ 模型配置 ------------
MODELS = {
    "seed1.8": {
        "name": "Doubao-Seed-1.8",
        "model_id": "doubao-seed-1-8-251228",
        "ep": "ep-20260304174146-7x5vk",
        "description": "豆包 Seed 1.8",
    },
    "seed2.0-pro": {
        "name": "Doubao-Seed-2.0-Pro",
        "model_id": "doubao-seed-2-0-pro-preview-260215",
        "ep": "ep-20260304172343-7thnh",
        "description": "豆包 Seed 2.0 Pro Preview",
    },
    "qwen3-vl": {
        "name": "Qwen3-VL-Plus",
        "model_id": "qwen3-vl-plus",
        "description": "通义千问 Qwen3 VL Plus",
    },
    "qwen-vl-max": {
        "name": "Qwen-VL-Max",
        "model_id": "qwen-vl-max",
        "description": "通义千问 Qwen VL Max",
    },
}

# ------------ 提示词配置 ------------
PROMPTS = {
    "prompt优化": os.path.join(script_dir, 'prompts', 'prompt优化.md'),
    "prompt优化v3": os.path.join(script_dir, 'prompts', 'prompt优化v3.md'),
    # "prompt优化_v2": os.path.join(script_dir, 'prompts', 'prompt优化_v2.md'),
    "prompt_原始": os.path.join(script_dir, 'prompts', 'prompt原始.md'),
}

# ------------ 目录配置 ------------
data_dir = os.path.join(script_dir, 'data')
outputs_dir = os.path.join(script_dir, 'outputs')
images_dir = os.path.join(data_dir, 'images')
annotations_dir = os.path.join(data_dir, 'annotations')
outputs_viz_dir = os.path.join(outputs_dir, 'visualizations')
outputs_csv_dir = os.path.join(outputs_dir, 'csv')
outputs_bboxes_dir = os.path.join(outputs_dir, 'bboxes')

os.makedirs(outputs_viz_dir, exist_ok=True)
os.makedirs(outputs_csv_dir, exist_ok=True)
os.makedirs(outputs_bboxes_dir, exist_ok=True)

# ------------ 测试图片 ------------
# 自动扫描 data/images 目录下的编号图片 (00001~00020.jpg)
image_files = sorted([
    f for f in os.listdir(images_dir)
    if f.endswith('.jpg') and f.replace('.jpg', '').isdigit()
])

# ======================== 初始化 ========================

def _load_env_file(env_path: str) -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            # 去掉行内注释（未被引号保护的 #）
            raw_value = value.strip()
            if raw_value and raw_value[0] in ('"', "'"):
                # 带引号的值：取引号内的内容
                quote = raw_value[0]
                end = raw_value.find(quote, 1)
                if end != -1:
                    raw_value = raw_value[1:end]
                else:
                    raw_value = raw_value[1:]
            else:
                # 不带引号：# 之后的是注释
                if '#' in raw_value:
                    raw_value = raw_value[:raw_value.index('#')].strip()
            value = raw_value
            if key and key not in os.environ:
                os.environ[key] = value

_load_env_file(os.path.join(script_dir, "env"))
_load_env_file(os.path.join(script_dir, ".env"))


def _pick_first(*values):
    for v in values:
        if v:
            return v
    return ""


def build_runner(model_key, model_config, args):
    base_url = args.ark_base_url
    if model_key == "seed1.8":
        api_key = _pick_first(
            args.seed18_api_key,
            os.getenv("SEED18_API_KEY", "").strip(),
            os.getenv("ARK_API_KEY", "").strip()
        )
        if not api_key:
            raise RuntimeError("缺少 seed1.8 的 API Key")
        ep = _pick_first(
            args.seed18_ep,
            os.getenv("SEED18_EP", "").strip(),
            model_config.get("ep", "")
        )
        model_id = _pick_first(
            os.getenv("SEED18_MODEL_ID", "").strip(),
            model_config["model_id"]
        )
        return Seed18Runner(api_key=api_key, ep=ep, model_id=model_id, base_url=base_url)
    if model_key == "seed2.0-pro":
        api_key = _pick_first(
            args.seed20_api_key,
            os.getenv("SEED20_API_KEY", "").strip(),
            os.getenv("ARK_API_KEY", "").strip()
        )
        if not api_key:
            raise RuntimeError("缺少 seed2.0 的 API Key")
        ep = _pick_first(
            args.seed20_ep,
            os.getenv("SEED20_EP", "").strip(),
            model_config.get("ep", "")
        )
        model_id = _pick_first(
            os.getenv("SEED20_MODEL_ID", "").strip(),
            model_config["model_id"]
        )
        return Seed20Runner(api_key=api_key, ep=ep, model_id=model_id, base_url=base_url)
    if model_key == "qwen3-vl":
        api_key = _pick_first(
            args.qwen_api_key,
            os.getenv("DASHSCOPE_API_KEY", "").strip()
        )
        if not api_key:
            raise RuntimeError("缺少 qwen3-vl 的 API Key (DASHSCOPE_API_KEY)")
        model_id = model_config["model_id"]
        return QwenRunner(api_key=api_key, model_id=model_id)
    if model_key == "qwen-vl-max":
        api_key = _pick_first(
            args.qwen_api_key,
            os.getenv("DASHSCOPE_API_KEY", "").strip()
        )
        if not api_key:
            raise RuntimeError("缺少 qwen-vl-max 的 API Key (DASHSCOPE_API_KEY)")
        model_id = model_config["model_id"]
        return QwenRunner(api_key=api_key, model_id=model_id, enable_thinking=False)
    raise RuntimeError(f"未知模型: {model_key}")


def resolve_model_display(model_key, model_config, args):
    if model_key == "seed1.8":
        return _pick_first(
            args.seed18_ep,
            os.getenv("SEED18_EP", "").strip(),
            model_config.get("ep", ""),
            os.getenv("SEED18_MODEL_ID", "").strip(),
            model_config["model_id"]
        )
    if model_key == "seed2.0-pro":
        return _pick_first(
            args.seed20_ep,
            os.getenv("SEED20_EP", "").strip(),
            model_config.get("ep", ""),
            os.getenv("SEED20_MODEL_ID", "").strip(),
            model_config["model_id"]
        )
    if model_key == "qwen3-vl":
        return model_config["model_id"]
    return model_config["model_id"]

# ======================== 工具函数 ========================

def parse_json_groundtruth(image_name):
    """从 JSON 标注文件解析所有类别的 groundtruth bboxes"""
    json_name = os.path.splitext(image_name)[0] + '.json'
    json_path = os.path.join(annotations_dir, json_name)
    
    if not os.path.exists(json_path):
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    groundtruth = []
    # 所有包含 bbox 的类别
    categories = [
        'words', 'arrows', 'tsrs', 'reverses', 'exits',
        'elevators', 'forbiddens', 'tollgates', 'entry_and_exit', 'charge',
        'parking_spots'  # 增加 parking_spots
    ]
    for cat in categories:
        cat_data = data.get(cat, {})
        if not isinstance(cat_data, dict):
            continue
        for obj_id, obj in cat_data.items():
            bbox = obj.get('bbox')
            if bbox and len(bbox) == 4:
                name = obj.get('text', obj.get('type', cat))
                groundtruth.append({
                    'name': name or cat,
                    'bbox': [int(round(v)) for v in bbox]
                })
    
    return groundtruth

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def extract_bboxes(model_output):
    results = []
    
    def extract_from_json(data, category="unknown"):
        items = []
        if isinstance(data, dict):
            if 'bbox' in data:
                content = data.get('content', data.get('direction', category))
                items.append({
                    'content': str(content),
                    'bbox': data['bbox'],
                    'category': category
                })
            else:
                for key, value in data.items():
                    items.extend(extract_from_json(value, key))
        elif isinstance(data, list):
            for item in data:
                items.extend(extract_from_json(item, category))
        return items
    
    try:
        if isinstance(model_output, str):
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', model_output)
            if json_match:
                model_output = json_match.group(1)
            parsed = json.loads(model_output)
        else:
            parsed = model_output
        results = extract_from_json(parsed)
    except json.JSONDecodeError:
        pass
    
    return results

def adjust_bboxes(bboxes, image_width, image_height):
    adjusted = []
    for item in bboxes:
        bbox = item['bbox']
        x1, y1, x2, y2 = bbox
        
        if max(x1, y1, x2, y2) <= 1.0:
            x1 = int(x1 * image_width)
            y1 = int(y1 * image_height)
            x2 = int(x2 * image_width)
            y2 = int(y2 * image_height)
        elif max(x1, y1, x2, y2) <= 1000:
            # 对于 Qwen 系列等返回 0-1000 坐标的模型，由于默认长边基准进行等比例归一化，
            # 若将宽和高分别独立的按比例还原，会导致非正方形图片发生严重拉伸(例如高度异常)。
            # 正确的做法是以最长边为基线统一还原 (x1, y1 等值在内部本质表示百分比)。
            max_side = max(image_width, image_height)
            # 不过，如果模型输出的是 [ymin, xmin, ymax, xmax] 甚至会被 extract 解析成交叉。
            # 为了尽可能保持大部分按照 width/height 独立归一化的其它模型兼容性：
            x1 = int(x1 * image_width / 1000)
            y1 = int(y1 * image_height / 1000)
            x2 = int(x2 * image_width / 1000)
            y2 = int(y2 * image_height / 1000)
            
            # 针对 qwen3 独有的以长边作为 1000 归一化特性（如果只有宽度放大，高度不变）
            # 我们应该将高度乘数修复为 max_side
            y1 = int(y1 * max_side / image_height) if max_side != image_height else y1
            y2 = int(y2 * max_side / image_height) if max_side != image_height else y2
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)
        
        adjusted.append({
            'content': item['content'],
            'category': item['category'],
            'bbox': [x1, y1, x2, y2]
        })
    return adjusted

def calculate_metrics(pred_bboxes, gt_bboxes, iou_threshold=0.5):
    if not pred_bboxes or not gt_bboxes:
        return {
            'precision': 0.0, 'recall': 0.0, 'avg_iou': 0.0,
            'tp': 0, 'fp': len(pred_bboxes) if pred_bboxes else 0,
            'fn': len(gt_bboxes) if gt_bboxes else 0
        }
    
    matched_gt = set()
    matched_pairs = []
    
    for i, pred in enumerate(pred_bboxes):
        pred_box = pred['bbox']
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt in enumerate(gt_bboxes):
            if j in matched_gt:
                continue
            iou = calculate_iou(pred_box, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            matched_pairs.append(best_iou)
    
    tp = len(matched_pairs)
    fp = len(pred_bboxes) - tp
    fn = len(gt_bboxes) - tp
    
    precision = tp / len(pred_bboxes) if pred_bboxes else 0.0
    recall = tp / len(gt_bboxes) if gt_bboxes else 0.0
    avg_iou = sum(matched_pairs) / len(matched_pairs) if matched_pairs else 0.0
    
    return {
        'precision': precision, 'recall': recall, 'avg_iou': avg_iou,
        'tp': tp, 'fp': fp, 'fn': fn
    }


def calculate_model_iou(model_bboxes, groundtruth_bboxes):
    """为模型预测的所有bbox与标准答案的bbox计算IoU值（参考 compare.py）"""
    iou_results = []
    for pred_box in model_bboxes:
        max_iou = 0.0
        matched_gt_box = None
        for gt_box in groundtruth_bboxes:
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                matched_gt_box = gt_box
        iou_results.append({
            'pred_box': pred_box,
            'gt_box': matched_gt_box,
            'iou': max_iou
        })
    return iou_results


def save_bbox_info(model_key, model_name, image_name, prompt_name, run_id,
                   pred_bboxes, gt_bboxes, timestamp):
    """保存bbox信息到txt文件，格式与 compare.py 保持一致"""
    # 计算每个预测框与GT的IoU
    gt_bbox_list = [gt['bbox'] for gt in gt_bboxes]
    pred_bbox_list = [p['bbox'] for p in pred_bboxes]
    iou_results = calculate_model_iou(pred_bbox_list, gt_bbox_list)

    safe_prompt = prompt_name.replace(" ", "_")
    img_base = os.path.splitext(image_name)[0]
    filename = f"bbox_info_{model_key}_{safe_prompt}_{img_base}_run{run_id}_{timestamp}.txt"
    bbox_path = os.path.join(outputs_bboxes_dir, filename)

    with open(bbox_path, 'w', encoding='utf-8') as f:
        f.write(f"{model_name}模型bbox结果：\n")
        for i, bbox in enumerate(pred_bbox_list):
            x1, y1, x2, y2 = bbox
            if i < len(iou_results):
                iou = iou_results[i]['iou']
                f.write(f"<bbox>{x1} {y1} {x2} {y2}</bbox> IoU: {iou:.4f}\n")
            else:
                f.write(f"<bbox>{x1} {y1} {x2} {y2}</bbox>\n")
    return bbox_path


# 模型 -> 框颜色映射
MODEL_BBOX_COLORS = {
    "seed1.8":     "#87CEEB",   # 浅蓝色 (light blue)
    "seed2.0-pro": "#00008B",   # 深蓝色 (dark blue)
    "qwen3-vl":    "red",       # 红色
    "qwen-vl-max": "#8B008B",   # 紫色 (dark magenta)
}
DEFAULT_BBOX_COLOR = "orange"


def draw_result_bboxes(image_path, pred_bboxes, gt_bboxes, model_key, model_name,
                       image_name, prompt_name, run_id, timestamp):
    """在图片上绘制预测bbox并保存为jpg（参考 compare.py）"""
    gt_bbox_list = [gt['bbox'] for gt in gt_bboxes]
    pred_bbox_list = [p['bbox'] for p in pred_bboxes]
    iou_results = calculate_model_iou(pred_bbox_list, gt_bbox_list)

    pred_color = MODEL_BBOX_COLORS.get(model_key, DEFAULT_BBOX_COLOR)

    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("Arial.ttf", 40)
        except IOError:
            font = ImageFont.load_default()

        # 图例
        draw.text((20, 20), f"{model_name}", fill=pred_color, font=font)
        draw.text((20, 70), "GroundTruth", fill="green", font=font)

        # 绘制预测bbox (线宽增加到 4 以及设置颜色)
        for i, bbox in enumerate(pred_bbox_list):
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=pred_color, width=4)
            if i < len(iou_results):
                iou = iou_results[i]['iou']
                # 文字背景
                draw.text((x1, max(0, y1 - 35)), f"IoU: {iou:.2f}", fill=pred_color, font=font)

        # 绘制GT bbox（绿色，增加线宽到 3）
        for gt in gt_bboxes:
            x1, y1, x2, y2 = gt['bbox']
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

        safe_prompt = prompt_name.replace(" ", "_")
        img_base = os.path.splitext(image_name)[0]
        out_filename = f"result_with_bboxes_{model_key}_{safe_prompt}_{img_base}_run{run_id}_{timestamp}.jpg"
        out_path = os.path.join(outputs_viz_dir, out_filename)
        img.save(out_path, "JPEG")
    return out_path

# ======================== 核心测试 ========================

def test_single(image_path, prompt_name, prompt_text, model_key, model_config, runner, run_id):
    """执行单次测试"""
    image_name = os.path.basename(image_path)
    model_short = model_key
    print(f"    [{run_id}] {model_short} | {prompt_name} | {image_name}...", end=" ", flush=True)
    
    with Image.open(image_path) as img:
        width, height = img.size
    
    gt_bboxes = parse_json_groundtruth(image_name)
    
    start_time = time.time()
    try:
        response = runner.run(
            prompt_text,
            image_path,
            CALL_TIMEOUT_SECONDS
        )
        
        inference_time = time.time() - start_time
        output = response.choices[0].message.content
        total_tokens = getattr(response.usage, 'total_tokens', 0)
        
        raw_preds = extract_bboxes(output)
        pred_bboxes = adjust_bboxes(raw_preds, width, height)
        metrics = calculate_metrics(pred_bboxes, gt_bboxes, IOU_THRESHOLD)
        
        print(f"✓ P={metrics['precision']:.0%} R={metrics['recall']:.0%} IoU={metrics['avg_iou']:.2f} T={inference_time:.1f}s GT={len(gt_bboxes)}")
        
        return {
            'model': model_key,
            'model_name': model_config['name'],
            'image': image_name,
            'image_path': image_path,
            'prompt': prompt_name,
            'run_id': run_id,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'avg_iou': metrics['avg_iou'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'pred_count': len(pred_bboxes),
            'gt_count': len(gt_bboxes),
            'pred_bboxes': pred_bboxes,
            'gt_bboxes': gt_bboxes,
            'inference_time': inference_time,
            'total_tokens': total_tokens,
            'success': True
        }
    except concurrent.futures.TimeoutError:
        inference_time = time.time() - start_time
        error_message = f"timeout after {CALL_TIMEOUT_SECONDS}s"
        print(f"✗ 错误: {error_message}")
        return {
            'success': False, 'error': error_message,
            'model': model_key, 'model_name': model_config['name'],
            'prompt': prompt_name, 'image': image_name, 'run_id': run_id,
            'inference_time': inference_time
        }
    except Exception as e:
        inference_time = time.time() - start_time
        print(f"✗ 错误: {str(e)[:60]}")
        return {
            'success': False, 'error': str(e),
            'model': model_key, 'model_name': model_config['name'],
            'prompt': prompt_name, 'image': image_name, 'run_id': run_id
        }


def save_results_csv(results, model_key, prompt_name, timestamp):
    """保存单个模型+prompt组合的结果"""
    # 文件名: {模型}_{prompt名称}_{时间}.csv
    safe_prompt_name = prompt_name.replace(" ", "_")
    filename = f"{model_key}_{safe_prompt_name}_{timestamp}.csv"
    csv_path = os.path.join(outputs_csv_dir, filename)
    
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['模型', '提示词', '图片', '轮次', '精度', '召回率', 'IoU',
                        '预测数', 'GT数', 'TP', 'FP', 'FN', '耗时(s)', 'Tokens'])
        for r in results:
            if r.get('success', False):
                writer.writerow([
                    r['model_name'], r['prompt'], r['image'], r['run_id'],
                    f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['avg_iou']:.4f}",
                    r['pred_count'], r['gt_count'], r['tp'], r['fp'], r['fn'],
                    f"{r['inference_time']:.2f}", r['total_tokens']
                ])
    
    print(f"   📁 已保存: {filename}")
    return csv_path


def print_group_stats(label, runs):
    """打印一组结果的统计信息"""
    precisions = [r['precision'] for r in runs]
    recalls = [r['recall'] for r in runs]
    ious = [r['avg_iou'] for r in runs]
    times = [r['inference_time'] for r in runs]
    tokens = [r['total_tokens'] for r in runs]
    fps = [r['fp'] for r in runs]
    fns = [r['fn'] for r in runs]
    
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    avg_iou = sum(ious) / len(ious)
    avg_t = sum(times) / len(times)
    avg_tok = sum(tokens) / len(tokens)
    avg_fp = sum(fps) / len(fps)
    avg_fn = sum(fns) / len(fns)
    
    print(f"  {label} ({len(runs)}次)")
    print(f"    精度:    {avg_p:.1%} (范围: {min(precisions):.1%} ~ {max(precisions):.1%})")
    print(f"    召回率:  {avg_r:.1%} (范围: {min(recalls):.1%} ~ {max(recalls):.1%})")
    print(f"    IoU:     {avg_iou:.3f}")
    print(f"    错标(FP): 平均 {avg_fp:.1f} 个")
    print(f"    漏标(FN): 平均 {avg_fn:.1f} 个")
    print(f"    耗时:    {avg_t:.1f}s")
    print(f"    Tokens:  {avg_tok:.0f}")
    
    return {
        'avg_precision': avg_p, 'avg_recall': avg_r, 'avg_iou': avg_iou,
        'avg_time': avg_t, 'avg_tokens': avg_tok,
        'avg_fp': avg_fp, 'avg_fn': avg_fn,
        'count': len(runs)
    }


def print_model_comparison(all_stats):
    """打印模型间对比"""
    print("\n" + "=" * 70)
    print("🏆 模型对比总结")
    print("=" * 70)
    
    # 按 (prompt, image) 分组对比
    groups = defaultdict(dict)
    for (model, prompt, image), stats in all_stats.items():
        groups[(prompt, image)][model] = stats
    
    for (prompt, image), model_stats in sorted(groups.items()):
        print(f"\n📊 {prompt} × {image}")
        print("-" * 60)
        
        header = f"  {'指标':<12}"
        models_ordered = sorted(model_stats.keys())
        for m in models_ordered:
            header += f"  {m:<18}"
        if len(models_ordered) == 2:
            header += f"  {'差异':>10}"
        print(header)
        print("  " + "-" * (12 + 20 * len(models_ordered) + (12 if len(models_ordered) == 2 else 0)))
        
        metrics_display = [
            ('精度', 'avg_precision', '{:.1%}'),
            ('召回率', 'avg_recall', '{:.1%}'),
            ('IoU', 'avg_iou', '{:.3f}'),
            ('错标(FP)', 'avg_fp', '{:.1f}'),
            ('漏标(FN)', 'avg_fn', '{:.1f}'),
            ('耗时(s)', 'avg_time', '{:.1f}'),
            ('Tokens', 'avg_tokens', '{:.0f}'),
        ]
        
        for label, key, fmt in metrics_display:
            row = f"  {label:<12}"
            values = []
            for m in models_ordered:
                val = model_stats[m].get(key, 0)
                values.append(val)
                row += f"  {fmt.format(val):<18}"
            
            if len(values) == 2:
                diff = values[1] - values[0]
                if key in ('avg_fp', 'avg_fn', 'avg_time', 'avg_tokens'):
                    # 越小越好
                    indicator = "✅" if diff < 0 else ("⚠️" if diff > 0 else "=")
                else:
                    # 越大越好
                    indicator = "✅" if diff > 0 else ("⚠️" if diff < 0 else "=")
                
                if key in ('avg_precision', 'avg_recall'):
                    row += f"  {diff:>+.1%} {indicator}"
                elif key == 'avg_iou':
                    row += f"  {diff:>+.3f} {indicator}"
                else:
                    row += f"  {diff:>+.1f} {indicator}"
            
            print(row)


def save_comparison_csv(all_results, timestamp):
    """保存所有结果的汇总对比CSV"""
    filename = f"model_prompt_comparison_{timestamp}.csv"
    csv_path = os.path.join(outputs_csv_dir, filename)
    
    success_results = [r for r in all_results if r.get('success', False)]
    
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['模型', '模型名称', '提示词', '图片', '轮次', '精度', '召回率', 'IoU',
                        '预测数', 'GT数', 'TP', 'FP', 'FN', '耗时(s)', 'Tokens'])
        for r in success_results:
            writer.writerow([
                r['model'], r['model_name'], r['prompt'], r['image'], r['run_id'],
                f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['avg_iou']:.4f}",
                r['pred_count'], r['gt_count'], r['tp'], r['fp'], r['fn'],
                f"{r['inference_time']:.2f}", r['total_tokens']
            ])
    
    print(f"\n📁 汇总结果已保存: {filename}")
    return csv_path

# ======================== 命令行参数 ========================

def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description="多模型+多提示词对比测试")
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()), default=list(MODELS.keys()),
                        help=f"要测试的模型 (默认全部: {list(MODELS.keys())})")
    parser.add_argument('--prompts', nargs='+', choices=list(PROMPTS.keys()), default=list(PROMPTS.keys()),
                        help=f"要测试的提示词 (默认全部: {list(PROMPTS.keys())})")
    parser.add_argument('--runs', type=int, default=NUM_RUNS,
                        help=f"每个组合测试次数 (默认: {NUM_RUNS})")
    parser.add_argument('--ark-base-url', default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument('--seed18-api-key', default="")
    parser.add_argument('--seed18-ep', default="")
    parser.add_argument('--seed20-api-key', default="")
    parser.add_argument('--seed20-ep', default="")
    parser.add_argument('--qwen-api-key', default="",
                        help="Qwen3 VL API Key (DASHSCOPE_API_KEY)")
    parser.add_argument('--max-concurrent', type=int, default=5,
                        help="最大并发数 (默认: 5)")
    parser.add_argument('--images', nargs='+', default=None,
                        help="指定测试图片（如 img1.jpg img2.jpg），默认全部")
    return parser.parse_args()

# ======================== 主函数 ========================

def main():
    args = parse_args()
    
    num_runs = args.runs
    max_concurrent = args.max_concurrent
    selected_models = {k: MODELS[k] for k in args.models}
    selected_prompts = {k: PROMPTS[k] for k in args.prompts}
    
    # 构建本地图片路径列表
    test_images = [os.path.join(images_dir, f) for f in image_files]
    if args.images:
        test_images = [p for p in test_images if os.path.basename(p) in args.images]
        if not test_images:
            print("⚠️ 指定的图片不在列表中，使用全部图片")
            test_images = [os.path.join(images_dir, f) for f in image_files]
    
    total_combinations = len(selected_models) * len(selected_prompts) * len(test_images) * num_runs
    
    print("\n" + "=" * 70)
    print("🔬 多模型 × 多提示词对比测试")
    print("=" * 70)
    print(f"  模型:     {', '.join(selected_models.keys())}")
    print(f"  提示词:   {', '.join(selected_prompts.keys())}")
    print(f"  图片:     {len(test_images)} 张 ({image_files[0]}~{image_files[-1]})")
    print(f"  每组轮次: {num_runs}")
    print(f"  并发数:   {max_concurrent}")
    print(f"  总测试数: {total_combinations}")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    all_stats = {}  # key: (model, prompt, image) -> stats
    
    for model_key, model_config in selected_models.items():
        print(f"\n{'='*70}")
        print(f"🤖 模型: {model_config['name']} ({model_key})")
        print(f"   Model ID: {resolve_model_display(model_key, model_config, args)}")
        print(f"{'='*70}")
        
        runner = build_runner(model_key, model_config, args)
        
        for prompt_name, prompt_path in selected_prompts.items():
            if not os.path.exists(prompt_path):
                print(f"  ⚠️ 跳过 {prompt_name}: 文件不存在 ({prompt_path})")
                continue
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()
            
            print(f"\n  📝 提示词: {prompt_name}")
            print(f"     文件: {os.path.basename(prompt_path)}")
            print(f"  " + "-" * 50)
            
            combo_results = []
            image_results_map = {os.path.basename(p): [] for p in test_images}
            
            tasks = []
            for image_path in test_images:
                for run_id in range(1, num_runs + 1):
                    tasks.append((image_path, run_id))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                future_to_task = {
                    executor.submit(
                        test_single,
                        image_path,
                        prompt_name,
                        prompt_text,
                        model_key,
                        model_config,
                        runner,
                        run_id
                    ): (image_path, run_id)
                    for image_path, run_id in tasks
                }
                
                for future in concurrent.futures.as_completed(future_to_task):
                    result = future.result()
                    all_results.append(result)
                    combo_results.append(result)
                    if result.get('success'):
                        image_results_map[result['image']].append(result)
                        # 保存 bbox info txt 和 result_with_bboxes jpg
                        try:
                            txt_path = save_bbox_info(
                                result['model'], result['model_name'],
                                result['image'], result['prompt'], result['run_id'],
                                result['pred_bboxes'], result['gt_bboxes'], timestamp
                            )
                            jpg_path = draw_result_bboxes(
                                result['image_path'], result['pred_bboxes'], result['gt_bboxes'],
                                result['model'], result['model_name'],
                                result['image'], result['prompt'], result['run_id'], timestamp
                            )
                            print(f"      📄 bbox txt: {os.path.basename(txt_path)}")
                            print(f"      🖼️  bbox jpg: {os.path.basename(jpg_path)}")
                        except Exception as e:
                            print(f"      ⚠️ 保存bbox/图片失败: {e}")
            
            for image_path in test_images:
                image_name = os.path.basename(image_path)
                image_results = image_results_map.get(image_name, [])
                if image_results:
                    stat_key = (model_key, prompt_name, image_name)
                    stats = print_group_stats(
                        f"  {model_key} | {prompt_name} | {image_name}",
                        image_results
                    )
                    all_stats[stat_key] = stats
            
            # 保存单个模型+prompt组合的CSV
            success_combo = [r for r in combo_results if r.get('success')]
            if success_combo:
                save_results_csv(success_combo, model_key, prompt_name, timestamp)
    
    # ======================== 对比报告 ========================
    success_results = [r for r in all_results if r.get('success', False)]
    
    if success_results:
        # 保存汇总CSV
        save_comparison_csv(all_results, timestamp)
        
        # 打印模型间对比
        if len(selected_models) >= 2:
            print_model_comparison(all_stats)
    
    print(f"\n{'='*70}")
    print(f"✅ 测试完成! 共 {len(success_results)}/{total_combinations} 次成功")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
