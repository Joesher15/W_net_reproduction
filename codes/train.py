# Implementation of W-Net: A Deep Model for Fully Unsupervised Image Segmentation
# in Pytorch.
# Author: Griffin Bishop

from __future__ import division
from __future__ import print_function

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
import time
from util import util
from util.autoencoder_dataset import AutoencoderDataset
from config import Config
from model import WNet
from soft_n_cut_loss import NCutLoss2D


def main():
    print("PyTorch Version: ", torch.__version__)
    if torch.cuda.is_available():
        print("Cuda is available. Using GPU")

    config = Config()
    torch.autograd.set_detect_anomaly(True)
    ###################################
    # Image loading and preprocessing #
    ###################################

    # TODO: Maybe we should crop a large square, then resize that down to our patch size?
    # For now, data augmentation must not introduce any missing pixels TODO: Add data augmentation noise
    # TODO: Change Transormations
    train_xform = transforms.Compose([
        # transforms.RandomCrop(224),
        transforms.Resize((config.input_size, config.input_size)),
        # transforms.RandomCrop(config.input_size + config.variationalTranslation),  # For now, cropping down to 224
        # transforms.RandomHorizontalFlip(),  # TODO: Add colorjitter, random erasing
        transforms.ToTensor()
    ])
    val_xform = transforms.Compose([
        # transforms.CenterCrop(224),
        transforms.Resize((config.input_size, config.input_size)),
        # transforms.CenterCrop(config.input_size),
        transforms.ToTensor()
    ])

    # TODO: Load validation segmentation maps too  (for evaluation purposes)
    train_dataset = AutoencoderDataset("train", train_xform)
    val_dataset = AutoencoderDataset("val", val_xform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4,
                                                   shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, num_workers=4, shuffle=False)

    util.clear_progress_dir()

    ###################################
    #          Model Setup            #
    ###################################

    autoencoder = WNet()
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
    optimizerE = torch.optim.Adam(autoencoder.U_encoder.parameters(), lr=0.003)
    optimizerW = torch.optim.Adam(autoencoder.parameters(), lr=0.003)
    schedulerE = torch.optim.lr_scheduler.StepLR(optimizerE, step_size=1000, gamma=0.1)
    schedulerW = torch.optim.lr_scheduler.StepLR(optimizerW, step_size=1000, gamma=0.1)
    if config.debug:
        print(autoencoder)
    util.enumerate_params([autoencoder])

    # Use the current time to save the model at end of each epoch
    modelName = str(datetime.now())

    ###################################
    #          Loss Criterion         #
    ###################################

    def reconstruction_loss(x, x_prime):
        binary_cross_entropy = F.mse_loss(x_prime, x, reduction='sum')
        return binary_cross_entropy

    ###################################
    #          Training Loop          #
    ###################################

    autoencoder.train()

    progress_images, progress_expected = next(iter(val_dataloader))
    n_iter = 1
    
    
    ################################################
    reconstruction_loss_array=[]
    soft_n_cut_loss_array=[]

    tittle='losses' + time.strftime("_%H_%M_%S", time.localtime())
    loss_dir=config.loss_csvfile_destination+tittle
    losses=[]
    np.savetxt(loss_dir+'.csv', losses, fmt='%.2f', delimiter=',', header="ReconstructionLoss,  SoftNcutLoss")
    ################################################
    for epoch in range(config.num_epochs):
        running_loss = 0.0
        for i, [inputs, outputs] in enumerate(train_dataloader, 0):

            if config.showdata:
                print(inputs.shape)
                print(outputs.shape)
                print(inputs[0])
                plt.imshow(inputs[0].permute(1, 2, 0))
                plt.show()

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                outputs = outputs.cuda()

            optimizerE.zero_grad()

            segmentations = autoencoder.forward_encoder(inputs)

            l_soft_n_cut = NCutLoss2D()(segmentations, inputs)
            l_soft_n_cut.backward(retain_graph=False)
            optimizerE.step()
            schedulerE.step()

            optimizerW.zero_grad()

            segmentations, reconstructions = autoencoder.forward(inputs)

            l_reconstruction = reconstruction_loss(
                inputs if config.variationalTranslation == 0 else outputs,
                reconstructions
            )

            # loss = (l_reconstruction + l_soft_n_cut)
            l_reconstruction.backward(
                retain_graph=False)  # We only need to do retain graph =true if we're backpropping from multiple heads
            optimizerW.step()
            schedulerW.step()

            if config.debug and (i % 50) == 0:
                print("it. ",i, " | ReconstructionLoss: ",l_reconstruction.item()," | SoftNCutLoss: ", l_soft_n_cut.item())
                # print(optimizerE.param_groups[0]['lr'])
                # print(optimizerW.param_groups[0]['lr'])

            # print statistics
            running_loss += l_reconstruction.item()

            if config.showSegmentationProgress and i == 0:  # If first batch in epoch
                util.save_progress_image(autoencoder, progress_images, epoch)
                # optimizerE.zero_grad()  # Don't change gradient on validation
            n_iter += 1
            # print(l_reconstruction)
            ##############################
            reconstruction_loss_array=np.append(reconstruction_loss_array,[[l_reconstruction.item()]])
            soft_n_cut_loss_array=np.append(soft_n_cut_loss_array,[[l_soft_n_cut.item()]])
            ##############################

        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f"Epoch {epoch} loss: {epoch_loss:.6f}")
        

        ####################################
        losses=np.vstack((reconstruction_loss_array,soft_n_cut_loss_array)).T
        
        with open(loss_dir+'.csv', 'a') as abc:
          np.savetxt(abc, losses, delimiter=",")
        
        #####################################

        if config.saveModel:
            util.save_model(autoencoder, modelName)


if __name__ == "__main__":
    main()
