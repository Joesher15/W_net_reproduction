class Config():
    def __init__(self):
        self.debug = True
        self.input_size = 128  # Side length of square image patch
        self.batch_size = 2
        self.val_batch_size = 2
        self.test_batch_size = 1
        self.verbose_testing = True

        self.k = 64  # Number of classes
        self.num_epochs = 2  # 250 for real
        self.data_dir = "../datasets/training_set/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"  # Directory of images

        self.showdata = False  # Debug the data augmentation by showing the data we're training on.

        self.useInstanceNorm = False  # Instance Normalization
        self.useBatchNorm = True  # Only use one of either instance or batch norm
        self.useDropout = True
        self.drop = 0.65

        # Each item in the following list specifies a module.
        # Each item is the number of input channels to the module.
        # The number of output channels is 2x in the encoder, x/2 in the decoder.
        self.encoderLayerSizes = [64, 128, 256, 512]
        self.decoderLayerSizes = [1024, 512, 256]

        self.showSegmentationProgress = True
        self.segmentationProgressDir = '../results/latent_images/'

        self.variationalTranslation = 0  # Pixels, 0 for off. 1 works fine

        self.loss_csvfile_destination = '../results/training_losses/'
        self.saveModel = True

        # ------------------------------Predict & Test Phase -----------------------------#
        # Flag for predicting & Metric Evaluation using BSD500
        self.USE_BSD500 = True  # <<<<============= Dataset Selection
        self.metrics_visualisation_flag = False
        self.metrics_print_flag = False
        # Model Name
        model_name = "2021-04-16_14_31_23_999059"
        # Directory from where to load the model
        self.loaded_model = "../results/saved_models/" + model_name

        # Converting the BSD500 gt frrom ".mat" files to ".npy" matrices required for predictions
        # Only need to be done once after conversion to .npy files are being loaded frrom converted_segmentations folder
        self.BSD500_preprocessing = False  # True

        # Path where both BSD500 and BSD300 are located
        self.test_path = "../datasets/test_set/"

        # Directory where to save the predictions made using the loaded model
        self.predictions_destination = "../results/test_set_predictions/"

        if self.USE_BSD500:
            # Directory to load the Images of BSD 500
            self.test_set_image_dir = 'BSR/BSDS500/data/images/'
            self.test_set_gt_dir = '/BSR/BSDS500/data/groundTruth/'
            self.predictions_destination = "../results/test_set_predictions/BSD500/"
        else:
            # Directory to load the images of BSD300
            self.test_set_image_dir = "BSDS300/images/"
            self.test_set_gt_dir = 'BSDS300/groundTruth/'
            self.predictions_destination = "../results/test_set_predictions/BSD300/"
