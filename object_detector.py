import cv2
import numpy as np


class ObjectDetector:

    def __init__(self, min_size=(10, 10), max_size=(100, 100), min_area=50, max_area=5000):
        self.min_size = min_size
        self.max_size = max_size
        self.min_area = min_area
        self.max_area = max_area
        return

    def __str__(self):
        return "Object detector"

    def __call__(self, *args, **kwargs):
        return self.process(args[0], args[1])

    def _check_size(self, size, area):
        return (self.min_size[0] <= size[0] <= self.max_size[0]) \
               and (self.min_size[1] <= size[1] <= self.max_size[1]) \
               and (self.min_area <= area <= self.max_area)

    def process(self, idx_map, idx_cnt):
        out = []
        for i in range(idx_cnt):
            binary = (idx_map == i).astype(np.uint8) * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
            for concomp_idx in range(num_labels):
                x, y, w, h = stats[concomp_idx][cv2.CC_STAT_LEFT], stats[concomp_idx][cv2.CC_STAT_TOP], \
                             stats[concomp_idx][cv2.CC_STAT_WIDTH], stats[concomp_idx][cv2.CC_STAT_HEIGHT],
                area = stats[concomp_idx][cv2.CC_STAT_AREA]
                if self._check_size((w, h), area):
                    out.append((x, y, w, h))
        return out
