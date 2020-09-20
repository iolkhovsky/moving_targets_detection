from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans, AgglomerativeClustering
import numpy as np


def normalize(x):
    mean, std = x.mean(), x.std()
    return (x - mean) / std


def apply(x, color):
    return color[x]


def draw_clusters(labels):
    color = np.random.randint(0, 255, (labels.max()+1, 3))
    # out = np.zeros(shape=(labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    out = apply(labels, color)
    return out.astype(np.uint8)


class Clusterizer:

    def __init__(self):
        pass

    def __str__(self):
        pass

    def process(self, mag, ang):
        frame_shape = mag.shape
        y_coord, x_coord = np.indices(mag.shape)
        y_coord = y_coord.flatten()
        x_coord = x_coord.flatten()
        mag = mag.flatten()
        ang = ang.flatten()

        mag = normalize(mag)
        ang = normalize(ang)
        y_coord = normalize(y_coord)
        x_coord = normalize(x_coord)
        descriptors = np.asarray([y_coord, x_coord, mag, ang])
        descriptors = np.swapaxes(descriptors, 0, 1)
        descriptors = np.reshape(descriptors, (-1, 4))
        clustering = DBSCAN(eps=3, min_samples=2).fit(descriptors)
        #clustering = MiniBatchKMeans(n_clusters=5).fit(descriptors)
        #clustering = AgglomerativeClustering().fit(descriptors)
        out_map = np.reshape(clustering.labels_, frame_shape)
        return out_map, draw_clusters(out_map)

    def __call__(self, *args, **kwargs):
        return self.process(args[0], args[1])

