# W-Net reproduction
This repository contains the code for reproducing the results of W-net: A Deep Model for Fully Unsupervised Image Segmentation (https://arxiv.org/abs/1711.08506)

## Train
- Download the VOC dataset for training from http://host.robots.ox.ac.uk/pascal/VOC/ and paste the downloaded folder inside /datasets/training_set
- After setting relevant debug flags and hyperparameters in the /codes/config.py file, run the train.py for training

## Test
- Download the BSD300 and BSD500 datasets from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/ and https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html respectively and place inside the datasets/test_set folder.
In the config file, set which dataset is to be selected for testing using the useBSD500 flag.
- Run prediction.py to generate the predictions for the datasets.
- Run metrics_evaluation to generate the final metrics for the selected dataset.
