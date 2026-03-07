# VLM-parking-labeling-multiPics

用于对比多模态模型（Doubao Seed 1.8、Seed 2.0 Pro、Qwen 系列）在停车场目标检测/框选任务上的效果，并输出：

- 预测框可视化对比图
- bbox 文本结果
- CSV 汇总（包含 Precision、Recall、IoU、推理耗时、Tokens 等）
- 实验结论报告（docs）

## 目录结构

```
VLM-parking-labeling-multiPics/
├── compare.py                  # 旧版主脚本（seed-1.8 vs Qwen3 对比）
├── test_prompt_comparison.py   # 新版多模型×多提示词对比测试脚本
├── runners/                    # 模型推理 Runner
│   ├── seed18_runner.py        #   Doubao Seed 1.8 Runner
│   ├── seed20_runner.py        #   Doubao Seed 2.0 Pro Runner
│   └── qwen_runner.py          #   Qwen-VL 系列 (含 Qwen3)
├── prompts/                    # 提示词文件
│   ├── prompt优化v3.md          #   针对停车位编号优化的提示词 (最新)
│   ├── prompt优化.md           #   优化后的提示词
│   └── prompt原始.md           #   原始提示词
├── data/
│   ├── annotations/            # JSON 标注 (00001~00020.json)
│   └── images/                 # 本地图片 (00001~00020.jpg)
├── outputs/
│   ├── csv/                    # CSV 汇总结果
│   ├── bboxes/                 # 详细框坐标 txt
│   └── visualizations/         # 可视化对比图
├── docs/                       # 实验对比报告/结论
├── .env_example                # 环境变量示例
├── .env                        # 环境变量（git ignored）
└── env                         # 环境变量（git ignored）
```

## 环境准备

### Python 版本

建议使用 **Python 3.9+**。

### 安装依赖

```bash
pip install requests pillow openai volcengine-sdk-ark-runtime dashscope python-dotenv
```

## 配置 API Key

在项目根目录创建 `.env` 文件（可参考 `.env_example`）：

```bash
ARK_API_KEY=YOUR_ARK_API_KEY
DASHSCOPE_API_KEY=YOUR_DASHSCOPE_API_KEY

# 可选：为不同模型指定独立的 API Key 和 Endpoint
SEED18_API_KEY=YOUR_SEED18_API_KEY
SEED18_EP=ep-xxxx
SEED20_API_KEY=YOUR_SEED20_API_KEY
SEED20_EP=ep-xxxx
```

## 运行

### 1. 多模型 × 多提示词对比测试（推荐）

使用 `test_prompt_comparison.py`，支持多模型（Seed / Qwen3）的高性能并行评测。

#### 基本用法

```bash
python3 test_prompt_comparison.py
```

默认行为：测试全部模型 × 全部提示词 × 全部图片，并发 5 线程。

#### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--models` | 要测试的模型 | `seed1.8 seed2.0-pro qwen-vl-max qwen3-vl` |
| `--prompts` | 要测试的提示词 | `prompt优化v3` 等 |
| `--runs` | 每个组合的测试次数 | `3` |
| `--max-concurrent` | 最大并发线程数 | `5` |
| `--images` | 指定测试图片文件名 | 全部图片 (00001~00020) |

#### 使用示例

**对比 4 个模型，每组跑 3 次，5 线程并发：**

```bash
python3 test_prompt_comparison.py \
  --models seed1.8 seed2.0-pro qwen-vl-max qwen3-vl \
  --prompts prompt优化v3 \
  --runs 3 \
  --max-concurrent 5
```

## 坐标修正说明 (Crucial)
针对 Qwen3-VL 的输出框漂移问题，由于其特殊的 0-1000 长边归一化逻辑，解析代码在 `test_prompt_comparison.py:adjust_bboxes` 中进行了等比例映射修正，确保了即使在长方形图片上也能正确对齐标注框。

## 评估指标

| 指标 | 说明 |
|------|------|
| Precision | 预测框中正确的比例（TP / 预测总数） |
| Recall | 真实框中被检出的比例（TP / GT 总数） |
| IoU | 匹配框对的平均交并比 (阈值 0.5) |
| FP / FN | 错标数与漏标数 |

## 报告

实验对比结论保存在 `docs/` 下，涵盖了 qwen3-vl 解析修正前后的数据对比。
