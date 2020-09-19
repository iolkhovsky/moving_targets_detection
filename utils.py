

def find_common_roi(box1, box2):
    if box1 is None or box2 is None:
        return None
    assert len(box1) == 4
    assert len(box2) == 4
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return int(min(x1, x2)), int(min(y1, y2)), \
           int(max(x1 + w1, x2 + w2)), int(max(y1 + h1, y2 + h2))