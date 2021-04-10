class Config():
    def __init__(self):
        self.debug = True
        self.input_size = 128  # Side length of square image patch ## TODO: Figure out
        #self.batch_size = 10  # Batch size of patches Note: 11 gig gpu will max batch of 5 ## TODO: Figure out
        #self.val_batch_size = 4  # Number of images shown in progress ## TODO: Figure out
        self.batch_size = 2
        self.val_batch_size = 2
        self.test_batch_size = 1  # We only use the first part of the model here (forward_encoder), so it can be larger ## TODO: Figure out
        self.verbose_testing = True

        self.k = 64  # Number of classes ## TODO: Figure out
        # self.num_epochs = 36  # 250 for real ## TODO: Figure out
        self.num_epochs = 2  # 250 for real ## TODO: Figure out
        #self.data_dir = "../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"  # Directory of images
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

        self.saveModel = True


        #------------------------------Predict Phase -----------------------------#

        self.BSD500_preprocessing =True #False
        self.loaded_model="../results/saved_models/2021-04-04_23_11_33_717535"
        self.test_path ="../datasets/test_set/"
        self.predictions_destination="../results/test_set_predictions/"

        # ------------------------------Test Phase -----------------------------#

