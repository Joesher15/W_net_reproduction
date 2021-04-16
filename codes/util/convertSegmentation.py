# Converts Berkeley segmentation dataset segmentation format files to .npy arrays
import numpy as np
import os

destination = "../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/test/segmentations"


def convertAndSave(filepath, filename):
    f = open(filepath, 'r')
    w, h = (0, 0)
    for line in f:
        if 'width' in line:
            w = int(line.split(' ')[1])
        if 'height' in line:
            h = int(line.split(' ')[1])
        if 'data' in line:
            break

    seg = np.zeros((h, w))
    for line in f:
        s, r, c1, c2 = map(lambda x: int(x), line.split(' '))
        seg[r, c1:c2] = s

    path = os.path.join(destination, filename)
    np.save(path, seg)


path = ""
dirs = list()
for dir, _, files in os.walk(path):
    for filename in files:
        filepath = os.path.join(dir, filename)
        print(filepath)
        convertAndSave(filepath, filename)
