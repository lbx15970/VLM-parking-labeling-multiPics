import json
from test_prompt_comparison import adjust_bboxes

def test_bboxes():
    # 00002.jpg run2 mock
    # content: 出口, bbox: [220, 550, 240, 585] -> x1=220, y1=550 
    bboxes_00002_run2 = [{"content": "出口", "category": "word", "bbox": [220, 550, 240, 585]}]
    res1 = adjust_bboxes(bboxes_00002_run2, 3840, 1888, "qwen3.5-plus")
    print(f"00002 run2 adjusted: {res1[0]['bbox']} | Expected roughly: [844, 1038, 921, 1104]")

    # 00002.jpg run3 mock
    # content: B1, bbox: [215, 335, 228, 355] -> x1=215, y1=335
    bboxes_00002_run3 = [{"content": "B1", "category": "word", "bbox": [215, 335, 228, 355]}]
    res2 = adjust_bboxes(bboxes_00002_run3, 3840, 1888, "qwen3.5-plus")
    print(f"00002 run3 adjusted: {res2[0]['bbox']} | Expected roughly: [825, 1286, 875, 1363]")

    # 00005.jpg run2 mock
    # content: B3, bbox: [420, 478, 442, 498]
    bboxes_00005_run2 = [{"content": "B3", "category": "word", "bbox": [420, 478, 442, 498]}]
    res3 = adjust_bboxes(bboxes_00005_run2, 3840, 1888, "qwen3.5-plus")
    print(f"00005 run2 adjusted: {res3[0]['bbox']} | Expected roughly: [1612, 902, 1697, 940]")

    # 00005.jpg run3 mock
    # content: B3, bbox: [790, 485, 815, 505], 禁止右转, bbox: [1100, 485, 1125, 510]
    bboxes_00005_run3 = [
        {"content": "B3", "category": "word", "bbox": [790, 485, 815, 505]},
        {"content": "禁止右转", "category": "word", "bbox": [1100, 485, 1125, 510]}
    ]
    res4 = adjust_bboxes(bboxes_00005_run3, 3840, 1888, "qwen3.5-plus")
    print(f"00005 run3 B3 adjusted: {res4[0]['bbox']} | Expected roughly: [1491, 915, 1538, 953]")

if __name__ == "__main__":
    test_bboxes()
