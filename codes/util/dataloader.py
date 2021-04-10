"""BSD500 Dataset"""
import os
import sys
import numpy as np
from PIL import Image
from scipy.io import loadmat
from config import Config

config = Config()


class DataLoader:
    def __init__(self, root=''):
        self.root = root
        self.imgs, self.gts, self.preds = _get_img_list(self.root)

    def __getitem__(self, index):
        img_path, gt_path, pred_path = self.imgs[index], self.gts[index], self.preds[index]

        raw_img = Image.open(img_path).convert('RGB')
        raw_gt, mean_gt = _loadmask(gt_path)
        raw_pred, mean_pred = _loadpred(pred_path)

        return raw_img, raw_gt, mean_gt, raw_pred, mean_pred

    def __len__(self):
        return len(self.imgs)


def _get_img_list(folder):
    img_paths = []
    gt_paths = []
    pred_paths = []

    img_directory = os.path.join(folder, config.test_path+config.test_set_image_dir)

    gt_directory = os.path.join(folder, config.test_path +config.test_set_gt_dir)
    pred_folder = os.path.join(folder, config.predictions_destination )

    #We are using for both DATAsets the ground truths of BSD500
    #Thus there is a test, train and val fodler
    segmentation_path_folders = ["test/", "train/", "val/"]

    if (config.USE_BSD500):
        # validation image folder in the BSD500 download
        image_path_folders = ["test/", "train/", "val/"]

    else:
        # no image validation folder in the BSD300 download
        image_path_folders = ["test/", "train/"]

    #looping through the image path folders
    for dir in image_path_folders:
        # Joining the constant image directory with instances [train/test/val] from the list
        img_folder=os.path.join(img_directory,dir)
        #Loop through the dir for the segmentation (different loop as for the BSDS300 we dont have a validation folder)
        for dir2 in segmentation_path_folders:
            gt_folder=os.path.join(gt_directory,dir2)


            for filename in os.listdir(img_folder):
                if filename.endswith(".jpg"):

                    imgpath = os.path.join(img_folder, filename)

                    mat_file = filename.replace('.jpg', '.mat')
                    gtpath = os.path.join(gt_folder, mat_file)
                    predpath = os.path.join(pred_folder, mat_file)

                    if (os.path.isfile(imgpath) and os.path.isfile(gtpath) and os.path.isfile(predpath)):
                        img_paths.append(imgpath)
                        gt_paths.append(gtpath)
                        pred_paths.append(predpath)
                    else:
                        print('cannot find the gt or image or pred:', imgpath, gtpath, predpath)

    print('Found {} images in the folder {}'.format(len(img_paths), img_directory))

    # img_paths.sort()
    # gt_paths.sort()
    # pred_paths.sort()

    return img_paths, gt_paths, pred_paths


def _loadmask(mask_path):
    raw_masks = []
    mean_mask = None
    mask_mat = loadmat(mask_path)['groundTruth']

    idx = mask_mat.shape[1]
    for i in range(idx):
        seg = mask_mat[0, i][0, 0][0]

        if i == 0:
            mean_mask = seg
        else:
            mean_mask = mean_mask + seg
        raw_masks.append(seg)

    mean_mask = mean_mask / idx
    raw_masks = np.array(raw_masks)

    return raw_masks, mean_mask.astype(np.int)


def _loadpred(pred_path):
    raw_preds = []
    mean_pred = None

    pred_mat = loadmat(pred_path)['segs']

    idx = pred_mat.shape[1]

    for i in range(idx):
        pred = pred_mat[0, i]

        if i == 0:
            mean_pred = pred
        else:
            mean_pred = mean_pred + pred

        raw_preds.append(pred)

    mean_pred = mean_pred / idx
    raw_preds = np.array(raw_preds)

    return raw_preds, mean_pred.astype(np.int)