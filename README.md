# VLM-parking-labeling-multiPics

本项目用于对比评估多种大视域多模态模型（VLM）在停车场场景下的目标检测与 OCR 标注任务能力。

## 核心特性
- **多模型支持**：支持 Doubao Seed 1.8、Seed 2.0 Pro、Qwen-VL-Max、Qwen3-VL 等主流模型。
- **自动化测试流程**：全自动扫描本地图片（00001~00020.jpg）及对应的 JSON 标注。
- **高度并发**：支持多线程并发调用 API，极大缩短评测耗时。
- **坐标修正技术**：针对 Qwen3-VL 等模型的等比例归一化特性，内置了坐标自动缩放与对齐算法，消除标注漂移。
- **全维度指标统计**：自动计算 Precision、Recall、IoU、TP/FP/FN、耗时（Latency）及 Token 消耗。
- **可视化输出**：生成预测框与 GroundTruth 对比的彩色标注图及 bbox 详表。

## 目录结构
```
VLM-parking-labeling-multiPics/
├── test_prompt_comparison.py   # 主评测脚本（支持多模型、多线程并发）
├── runners/                    # 模型 API 调用适配器
│   ├── seed18_runner.py        #   Doubao Seed 1.8
│   ├── seed20_runner.py        #   Doubao Seed 2.0 Pro
│   └── qwen_runner.py          #   Qwen-VL 系列（含 Qwen3）
├── prompts/                    # 评测使用的 Prompt 模板
│   ├── prompt优化v3.md          #   针对停车位编号优化的版本
│   └── prompt原始.md            #   基础版本
├── data/
│   ├── images/                 # 测试图片 (00001.jpg ~ 00020.jpg)
│   └── annotations/            # JSON 标注文件 (00001.json ~ 00020.json)
├── outputs/
│   ├── csv/                    # 评测指标 CSV 汇总
│   ├── bboxes/                 # 详细的 bbox 坐标与 IoU 记录
│   └── visualizations/         # 结果可视化对比图
└── docs/                       # 深度评测总结报告
```

## 环境准备
### 安装依赖
```bash
pip install requests pillow openai volcengine-sdk-ark-runtime dashscope python-dotenv
```

### 配置 API Key
在 `.env` 文件中配置：
```bash
ARK_API_KEY=your_ark_key
DASHSCOPE_API_KEY=your_dashscope_key
# 各模型具体的 Endpoint ID...
```

## 快速开始
**对比 4 个模型在特定 Prompt 下的表现（每张图跑 3 次，5 线程并发）：**
```bash
python3 test_prompt_comparison.py \
  --models seed1.8 seed2.0-pro qwen-vl-max qwen3-vl \
  --prompts "prompt优化v3" \
  --runs 3 \
  --max-concurrent 5
```

## 坐标漂移修正说明 (Important)
本项目在开发过程中发现 Qwen3-VL 的坐标输出存在特殊的归一化逻辑（基于图片长边的 1000 分度值），已在 `test_prompt_comparison.py` 的 `adjust_bboxes` 函数中进行了针对性修正，确保输出图片和指标重算的准确性。

## 许可证
MIT License
