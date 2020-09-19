import numpy as np


class Cropper:

    def __init__(self, roi):
        assert len(roi) == 4
        self.x, self.y, self.w, self.h = roi
        assert type(self.w) == int
        assert type(self.h) == int
        return

    def __call__(self, *args, **kwargs):
        img = args[0]
        assert type(img) == np.ndarray
        assert 2 <= len(img.shape) <= 3
        xsz, ysz = img.shape[1], img.shape[0]
        x, y = self.x, self.y
        if (x is None) or (y is None):
            x = int((xsz - self.w) / 2)
            y = int((ysz - self.h) / 2)
        x1 = min(xsz - 1, max(0, x))
        x2 = min(xsz - 1, max(0, x + self.w - 1))
        y1 = min(ysz - 1, max(0, y))
        y2 = min(ysz - 1, max(0, y + self.h - 1))
        return img[y1:y2+1, x1:x2+1]
