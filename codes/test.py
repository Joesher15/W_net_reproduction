# Model evaluation script
# As the model is unsupervised, this script tries all possible mappings
# Of segmentation to label and takes the best one.

# Since the model works on patches, we don't need to do any transforms, instead, we just
# need to cut the image into patches and feed all of them through
# Author: Griffin Bishop

from __future__ import print_function
from __future__ import division

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import scipy.io as sc

from config import Config
import util
from model import WNet
from evaluation_dataset import EvaluationDataset


def main():
    print("PyTorch Version: ", torch.__version__)
    if torch.cuda.is_available():
        print("Cuda is available. Using GPU")

    config = Config()

    ###################################
    # Image loading and preprocessing #
    ###################################

    def BSD500gt_to_npy():
        test_path = "../BSR/BSDS500/data/groundTruth/test"
        # test_path = "../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/test/segmentations"
        train_path = "../BSR/BSDS500/data/groundTruth/train"
        val_path = "../BSR/BSDS500/data/groundTruth/val"
        destination = "../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/test/segmentations"
        path_files = [test_path, train_path, val_path]
        for dir in path_files:
            for file in os.listdir(dir):
                if file.endswith(".mat"):
                    ppath_to_file = os.path.join(dir, file)
                    print(ppath_to_file)
                    mat = sc.loadmat(ppath_to_file)["groundTruth"][0, 0][0][0][0]
                    mat = mat.astype('int16')
                    # mat = np.load(ppath_to_file)
                    # print(mat, mat.shape, type(mat))
                    image_name = str(file).split(".")[0]
                    path = os.path.join(destination, image_name)
                    np.save(path, mat)

    BSD500gt_to_npy()

    evaluation_dataset = EvaluationDataset("test")

    evaluation_dataloader = torch.utils.data.DataLoader(evaluation_dataset,
                                                        batch_size=config.test_batch_size, num_workers=4, shuffle=False)

    ###################################
    #          Model Setup            #
    ###################################

    # We will only use .forward_encoder()
    if torch.cuda.is_available():
        autoencoder = torch.load("./models/2021-04-04_23_11_33_717535")
    else:
        autoencoder = torch.load("./models/2021-04-04_23_11_33_717535", map_location=torch.device('cpu'))
    util.enumerate_params([autoencoder])

    ###################################
    #          Testing Loop           #
    ###################################

    autoencoder.eval()

    def combine_patches(image, patches):
        w, h = image[0].shape
        segmentation = torch.zeros(w, h)
        x, y = (0, 0)  # Start of next patch
        for patch in patches:
            if y + size > h:
                y = 0
                x += size
            segmentation[x:x + size, y:y + size] = patch
            y += size
        return segmentation

    # Because this model is unsupervised, our predicted segment labels do not
    # correspond to the actual segment labels.
    # We need to figure out what the best mapping is.
    # To do this, we will just count, for each of our predicted labels,
    # The number of pixels in each class of actual labels, and take the max in that image
    def count_predicted_pixels(predicted, actual):
        pixel_count = torch.zeros(config.k, config.k)
        for k in range(config.k):
            mask = (predicted == k)
            masked_actual = actual[mask]
            for i in range(config.k):
                pixel_count[k][i] += torch.sum(masked_actual == i)
        return pixel_count

    # Converts the predicted segmentation, based on the pixel counts
    def convert_prediction(pixel_count, predicted):
        map = torch.argmax(pixel_count, dim=1)
        for x in range(predicted.shape[0]):
            for y in range(predicted.shape[1]):
                predicted[x, y] = map[predicted[x, y]]
        return predicted

    def compute_iou(predicted, actual):
        intersection = 0
        union = 0
        for k in range(config.k):
            a = (predicted == k).int()
            b = (actual == k).int()
            # if torch.sum(a) < 100:
            #    continue # Don't count if the channel doesn't cover enough
            intersection += torch.sum(torch.mul(a, b))
            union += torch.sum(((a + b) > 0).int())
        return intersection.float() / union.float()

    def pixel_accuracy(predicted, actual):
        return torch.mean((predicted == actual).float())

    iou_sum = 0
    pixel_accuracy_sum = 0
    n = 0
    # Currently, we produce the most generous prediction looking at a single image
    for i, [images, segmentations, image_path] in enumerate(evaluation_dataloader, 0):
        size = config.input_size
        # Assuming batch size of 1 right now
        image = images[0]
        target_segmentation = segmentations[0]

        # NOTE: We cut the images down to a multiple of the patch size
        cut_w = (image[0].shape[0] // size) * size
        cut_h = (image[0].shape[1] // size) * size
        # print(image[0].shape)
        # f, axes = plt.subplots(1, 2, figsize=(8, 8))
        # axes[0].imshow(image[0])
        image = image[:, 0:cut_w, 0:cut_h]
        # print(image[0].shape)
        # axes[1].imshow(image[0])
        # plt.show()
        target_segmentation = target_segmentation[:, 0:cut_w, 0:cut_h]

        # NOTE: problem - the above won't get all patches, only ones that fit. (Resolved by above cutting code)
        patches = image.unfold(0, 3, 3).unfold(1, size, size).unfold(2, size, size)
        # print(patches.shape)
        patch_batch = patches.reshape(-1, 3, size, size)

        if torch.cuda.is_available():
            patch_batch = patch_batch.cuda()
        seg_batch = autoencoder.forward_encoder(patch_batch)
        seg_batch = torch.argmax(seg_batch, axis=1).float()

        predicted_segmentation = combine_patches(image, seg_batch)
        # print(predicted_segmentation.shape)
        prediction = predicted_segmentation.int()
        # print(prediction, prediction.shape)
        actual = target_segmentation[0].int()
        # print(actual, actual.shape)

        pixel_count = count_predicted_pixels(prediction, actual)
        prediction = convert_prediction(pixel_count, prediction)

        # image_path
        image_name = str(image_path).split(".")[2].split("/")[-1] + ".mat"
        image_path = "../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/test/predicted_segs_matlab/" + image_name
        print(type(prediction))
        pred_object = prediction.cpu().detach().numpy()
        numpy_object = np.empty((1, 5), dtype=object)
        numpy_object[0, 0] = pred_object
        numpy_object[0, 1] = pred_object
        numpy_object[0, 2] = pred_object
        numpy_object[0, 3] = pred_object
        numpy_object[0, 4] = pred_object
        sc.savemat(image_path, mdict={'segs': numpy_object})

        # iou = compute_iou(prediction, actual)
        # iou_sum += iou
        # accuracy = pixel_accuracy(prediction, actual)
        # pixel_accuracy_sum += accuracy
        # n += 1
        #
        # if config.verbose_testing:
        #     print(f"Intersection over union for this image: {iou}")
        #     print(f"Pixel Accuracy for this image: {accuracy}")
        #
        # if config.verbose_testing:
        #     f, axes = plt.subplots(1, 5, figsize=(8,8))
        #     axes[0].imshow(predicted_segmentation)
        #     axes[1].imshow(prediction)
        #     axes[2].imshow(image.permute(1, 2, 0))
        #     axes[3].imshow(target_segmentation[0])
        #     correctness_map = (prediction == target_segmentation)
        #     axes[4].imshow(correctness_map[0]) # Yellow = Correct, Purple = wrong
        #     plt.show()
        #
        # if n % 2 == 0:
        #     print(f"{n}")

        # print(f"Average performance on n={n} validation images:")
        # print(f"mean IoU: {iou_sum/n}   | mean pixel accuracy: {pixel_accuracy_sum/n}")

if __name__ == "__main__":
    main()
