import cv2
import numpy as np

color = np.random.randint(0, 255, (100, 3))


def normalize(x):
    mean, std = x.mean(), x.std()
    return (x - mean) / std


def apply(x, color):
    return color[x]


def draw_clusters(labels):
    # out = np.zeros(shape=(labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    out = apply(labels, color)
    return out.astype(np.uint8)


class Clusterizer:

    def __init__(self, clusters_cnt, debug=False):
        self._clusters_cnt = clusters_cnt
        self._debug = debug
        return

    def __str__(self):
        return "Dense flow clusterizer"

    @staticmethod
    def convert_flow_2_img(mag, ang):
        norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv = np.zeros(shape=(mag.shape[0], mag.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = norm_mag
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return rgb

    def process(self, mag, ang):
        rgb = self.convert_flow_2_img(mag, ang)
        if self._debug:
            cv2.imshow('Optical flow', rgb)
        pixels = rgb.reshape((-1, 3))
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(pixels, self._clusters_cnt, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        clustered_img = center[label.flatten()]
        return label.reshape((rgb.shape[0], rgb.shape[1])), clustered_img.reshape(rgb.shape)

    def __call__(self, *args, **kwargs):
        return self.process(args[0], args[1])

