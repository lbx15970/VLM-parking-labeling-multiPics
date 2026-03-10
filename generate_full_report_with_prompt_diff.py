import datetime

# 读取 Best Run 报告内容
best_report_path = "docs/qwen3.5_vs_seed2.0_prompt_v4_test_report_best_20260309_222954.md"
with open(best_report_path, "r", encoding="utf-8") as f:
    best_report_content = f.read()

# Prompt v4 与 v1 对比内容
prompt_comparison_content = """
## Prompt 优化 v4 vs v1 变更说明

Prompt 优化 v4 相比 v1（即 `prompt优化.md`）进行了大幅度的细节完善和规则明确，旨在解决漏检、误检、颗粒度不足及车位号码归属不明等问题。主要区别如下：

### 1. 新增“车位号码”独立检测 (`parking_spots`)
*   **v1**: 将地面停车位内的数字列为**排除项**，不进行提取。
*   **v4**: 新增了 **`parking_spots`** 字段，要求**必须**提取地面、墙面或立柱上的车位号码（如 "023", "B-12"），并将其作为独立类别输出，不再混入 `words` 字段。

### 2. 文字提取 (`words`) 规则更加严格细致
*   **颗粒度要求（非常重要）**: v4 严厉禁止将多行或不同物理区域的文字合并为一个大框，要求**独立、分别**提取。
*   **防漏检增强**: v4 特别列出了容易漏检的场景，如立柱上的浅色文字、远处墙面的小字区域标识、低对比度文字等。
*   **明确排除项**: v4 明确要求排除**消防设施**（灭火器、消防栓等）上的文字。
*   **去重规则**: v4 强调禁止对同一文字重复标注（例如：不能既标文字本身，又标承载文字的广告牌整体）。

### 3. 箭头检测 (`arrows`) 覆盖面扩大
*   **全场景检测**: v4 强调必须检测**所有车道**（包括对向车道）的地面箭头，以及所有方向的箭头。
*   **组合标志处理**: v4 新增了对**多箭头组合标志**（如三个箭头紧贴组成的标志）的处理规则，要求将其视为一个整体进行框选。

### 4. 功能指示牌 (`tsrs`) 框选要求微调
*   **v1**: 仅要求框选指示牌整体。
*   **v4**: 提出了更细致的要求，提到需对箭头本身和指示牌外轮廓进行框选（虽然最终 JSON 结构可能仍为一个 bbox，但提示词加强了对内部元素的关注）。

### 5. 校验规则大幅增加
*   **v4 新增了多项校验规则**，包括：
    *   检查重复标注。
    *   确认是否误标了消防设施文字。
    *   确认多箭头组合是否已合并。
    *   确认所有车道的地面箭头是否已标全。

---
"""

# 寻找插入位置
# 我们将 Prompt 对比内容插入到 "结论摘要" 之前，或者 "概述" 之后
insert_pos = best_report_content.find("## 结论摘要")

if insert_pos != -1:
    final_report_content = best_report_content[:insert_pos] + prompt_comparison_content + best_report_content[insert_pos:]
else:
    # 如果找不到插入点，就追加到最后（虽然不太可能）
    final_report_content = best_report_content + "\n" + prompt_comparison_content

# 生成新的报告文件
output_filename = f"docs/4models_full_test_report_{datetime.datetime.now().strftime('%Y%m%d')}.md"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(final_report_content)

print(f"完整测试报告已生成: {output_filename}")
