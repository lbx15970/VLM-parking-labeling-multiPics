def adjust_bboxes_mock(bboxes, image_width, image_height):
    adjusted = []
    base_w, base_h = image_width, image_height
    if bboxes:
        max_x = max([max(b['bbox'][0], b['bbox'][2]) for b in bboxes])
        max_y = max([max(b['bbox'][1], b['bbox'][3]) for b in bboxes])
        if image_width > image_height:
            ratio_limit = (image_height / image_width) * 1000
            if max_x > 1005:  # small tolerance
                base_w, base_h = image_height, image_height
            elif max_y <= ratio_limit + 5: # very small tolerance
                base_w, base_h = image_width, image_width
            else:
                base_w, base_h = image_width, image_height
        else:
            ratio_limit = (image_width / image_height) * 1000
            if max_y > 1005:
                base_w, base_h = image_width, image_width
            elif max_x <= ratio_limit + 5:
                base_w, base_h = image_height, image_height
            else:
                base_w, base_h = image_width, image_height
                
    for item in bboxes:
        x1, y1, x2, y2 = item['bbox']
        px1 = int(x1 / 1000.0 * base_w)
        py1 = int(y1 / 1000.0 * base_h)
        px2 = int(x2 / 1000.0 * base_w)
        py2 = int(y2 / 1000.0 * base_h)
        adjusted.append({'bbox': [px1, py1, px2, py2]})
    return adjusted

bboxes_00002_run2 = [{"bbox": [220, 550, 240, 585]}, {"bbox": [470, 765, 505, 810]}]
print("00002 run2:", adjust_bboxes_mock(bboxes_00002_run2, 3840, 1888)[0]['bbox'], "Expect:", [844, 1038, 921, 1104])

bboxes_00002_run3 = [{"bbox": [215, 335, 228, 355]}, {"bbox": [470, 380, 505, 415]}]
print("00002 run3:", adjust_bboxes_mock(bboxes_00002_run3, 3840, 1888)[0]['bbox'], "Expect:", [825, 1286, 875, 1363])

bboxes_00005_run2 = [{"bbox": [420, 478, 442, 498]}, {"bbox": [560, 478, 575, 498]}]
print("00005 run2:", adjust_bboxes_mock(bboxes_00005_run2, 3840, 1888)[0]['bbox'], "Expect:", [1612, 902, 1697, 940])

bboxes_00005_run3 = [{"bbox": [790, 485, 815, 505]}, {"bbox": [1100, 485, 1125, 510]}]
print("00005 run3:", adjust_bboxes_mock(bboxes_00005_run3, 3840, 1888)[0]['bbox'], "Expect:", [1491, 915, 1538, 953])
