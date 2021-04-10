
import os, shutil
from config import Config

import scipy.io as sc
import numpy as np
config = Config()

def BSD500gt_to_npy(test_path):
    # test_path = "../BSR/BSDS500/data/groundTruth/test"
    # test_path = "../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/test/segmentations"
    # train_path = "../BSR/BSDS500/data/groundTruth/train"
    # val_path = "../BSR/BSDS500/data/groundTruth/val"
    # destination = "../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/test/segmentations/"
    # path_files = [test_path, train_path, val_path]
    destination = "../datasets/test_set/converted_segmentaions"
    test_path=test_path+"BSR/BSDS500/data/groundTruth/"
    path_files=["test/","train/","val/"]

    for dir in path_files:
        path=test_path+dir
        # print(path)
        for file in os.listdir(path):
            if file.endswith(".mat"):
                ppath_to_file = os.path.join(path, file)
                #print(ppath_to_file)
                mat = sc.loadmat(ppath_to_file)["groundTruth"][0, 0][0][0][0]
                mat = mat.astype('int16')

                image_name = str(file).split(".")[0]
                dest_path = os.path.join(destination, image_name)
                np.save(dest_path, mat)