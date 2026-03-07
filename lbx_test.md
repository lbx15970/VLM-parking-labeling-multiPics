### lbx 测试
#### 模型：--models seed1.8 seed2.0-pro 

**只测试 Qwen3-VL-Plus，跑 3 次：**
```bash
python3 test_prompt_comparison.py \
  --models qwen3-vl \
  --runs 3
```

**只测试 Seed 2.0 Pro，跑 3 次：**
```bash
python3 test_prompt_comparison.py \
  --models seed2.0-pro \
  --runs 3
```



**只用 `prompt优化` 对比两个模型，每组 3 次，5 线程并发：**
```bash
python3 test_prompt_comparison.py \
  --models qwen3-vl seed2.0-pro \
  --prompts prompt优化 \
  --runs 3 \
  --max-concurrent 5
```