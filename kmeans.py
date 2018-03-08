#! /usr/bin/python
# -*- encoding:utf-8 -*-
import gdal
import math
import numpy as np
from PIL import Image


def kmeans_init_centroids(x: np.ndarray, k: int) -> np.ndarray:
    rand_idx = np.random.permutation(x.shape[0])
    centroids = x[rand_idx[:k]]
    return centroids


def find_closest_centroids(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    k = centroids.shape[0]
    tmp = np.zeros((x.shape[0], k))
    for i in range(k):
        tmp[:, i] = np.sum((x - centroids[i, :]) ** 2, 1)
    idx = np.argmin(tmp, 1)
    return idx


def compute_centroids(x: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    centroids = np.zeros((k, x.shape[1]))
    for i in range(k):
        centroids[i] = np.mean(x[idx == i], 0)
    return centroids


def kmeans(x: np.ndarray, k: int, max_iter=300):
    if k > x.shape[0]:
        print('k in bigger than x\'s size!')
        return None, None, None
    centroids = kmeans_init_centroids(x, k)
    idx = np.zeros(x.shape[0])
    pre_centroids = np.copy(centroids)
    for i in range(max_iter):
        idx = find_closest_centroids(x, centroids)
        centroids = compute_centroids(x, idx, k)
        if np.sum(np.abs(pre_centroids - centroids) ** 2) == 0:
            print('Iteration: %d' % i)
            return centroids
        pre_centroids = np.copy(centroids)
    return centroids


def main(k=4, scale=100):
    data = gdal.Open('2010-7-11')
    width = 300
    bands = data.ReadAsArray()
    h = math.ceil(bands.shape[1] / width)
    w = math.ceil(bands.shape[2] / width)
    avg_bands = np.zeros((bands.shape[0], h, w))
    for b in range(bands.shape[0]):
        for i in range(0, bands.shape[1], width):
            for j in range(0, bands.shape[2], width):
                h_width, w_width = width, width
                if bands.shape[1] - i <= width:
                    h_width = bands.shape[1] - i
                if bands.shape[2] - j <= width:
                    w_width = bands.shape[2] - j
                if np.count_nonzero(bands[b, i:i + h_width, j:j + w_width]) == 0:
                    avg = 0
                else:
                    avg = np.sum(bands[b, i:i + h_width, j:j + w_width]) / np.count_nonzero(
                        bands[b, i:i + h_width, j:j + w_width])
                avg_bands[b, math.floor(i / width), math.floor(j / width)] = avg
    points = np.zeros((avg_bands.shape[1], avg_bands.shape[2], avg_bands.shape[0]))
    for y in range(avg_bands.shape[1]):
        for x in range(avg_bands.shape[2]):
            points[y, x] = avg_bands[:, y, x]
    raw_shape = points.shape[:2]
    points = points.reshape((points.shape[0] * points.shape[1], points.shape[2]))
    total = 0
    res = list()
    for point in points:
        if not (point == np.zeros(4)).all():
            res.append(point)
            total += 1
    x = np.array(res)
    raw_max = np.max(x, 0)
    raw_min = np.min(x, 0)
    x = (x / (np.max(x, 0) - np.min(x, 0)) * scale)
    centroids = kmeans(x, k=k)
    idx = find_closest_centroids(points, centroids)
    idx[np.sum(points, 1) == 0] = k
    idx = idx.reshape(raw_shape)
    colors = np.array([(255, 136, 0), (255, 0, 204), (0, 221, 0), (255, 255, 0), (0, 68, 153), (136, 0, 34),
                       (148, 148, 148), (102, 136, 0), (153, 0, 204), (0, 204, 187), (0, 102, 0), (102, 85, 0)])
    res = np.zeros((raw_shape[0], raw_shape[1], 3), dtype='uint8')
    idx_list = idx.reshape(idx.shape[0] * idx.shape[1])
    for i in range(k):
        res[idx == i] = colors[i]
        if len(points[idx_list == i]) != 0:
            t_max = np.max(points[idx_list == i], 0)
            t_min = np.min(points[idx_list == i], 0)
            t_max = (t_max / scale) * (raw_max - raw_min)
            t_min = (t_min / scale) * (raw_max - raw_min)
            print('class %d: ' % i, 'max:', t_max, ' min:', t_min)
    img = Image.fromarray(res)
    img = img.resize((img.size[0] * 20, img.size[1] * 20))
    img.show()

if __name__ == '__main__':
    main(k=6)
