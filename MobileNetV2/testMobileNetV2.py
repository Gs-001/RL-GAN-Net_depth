#.........................................................................................................................
#   Script to generate a Depth Image using a pre-trained MobileNet U-Net
#.........................................................................................................................

import os
import glob
import time
from PIL import Image
import numpy as np
import PIL
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
import torchvision.models as models
import cv2

from UtilityTest import DepthDataset
from UtilityTest import ToTensor

from Mobile_model import Model


#.........................................................................................................................
#   Hyperparameters
#.........................................................................................................................

data = "C:/Users/hunte/OneDrive/Documents/Projects/Major_Project/Depth_estimation/workspace/test_img"

pretrained_path = "C:/Users/hunte/OneDrive/Documents/Projects/Major_Project/Depth_estimation/workspace/10.pth"

batchSize = 2

#.........................................................................................................................

def showImages(data_path):
    
    depth_dataset = DepthDataset(root_dir=data_path)
    fig = plt.figure()
    for i in range(len(depth_dataset)):
        sample = depth_dataset[i]
        print(np.shape(sample))
        print(i, sample['image'].size)
        plt.imshow(sample['image'])
        plt.figure()

        if i == len(depth_dataset):
            plt.show()
            break


def testUNet():

    start_time = time.time()

    # Read the Test Images
    depth_dataset = DepthDataset(root_dir=data,transform=transforms.Compose([ToTensor()]))
    train_loader=torch.utils.data.DataLoader(depth_dataset, batchSize)
    dataiter = iter(train_loader)
    images = dataiter.next()
    
    print("\n Time taken to load Images: %s " % (time.time() - start_time))
    print("\n Test Dataset Shape: {shape}".format(shape=np.shape(depth_dataset)))

    # Import the Pre-trained Model
    #.......................................................

    model = Model().cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(pretrained_path))
    print("\n Loaded MobileNet U-Net Weights successfully\n")
    #model.eval()

    # Model Variables
    #.......................................................................

    # # Print model's state_dict
    # print("\n\nModel's state_dict:\n\n")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())


    # Generating Depth Images
    #.......................................................................

    start_time = time.time()

    for i, sample_batched1  in enumerate (train_loader):
        
        input_image = torch.autograd.Variable(sample_batched1['image'].cuda())
        
        GFV = model.module.encoder(input_image)
        out = model(input_image)

        x = out.detach().cpu().numpy()

        # The Model outputs an image of the shape (1,1,240,320)
        # The output depth image is now upscaled to get a 480 x 640 image

        img = x.reshape(240,320)
        scale_percent = 200 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # Resize the image to scaled dimensions
        depth_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        rgb_image = sample_batched1['image'].detach().cpu().numpy().reshape(3,480,640).transpose(1,2,0)

        # Saving the Images

        plt.imsave('C:/Users/hunte/OneDrive/Documents/Projects/Major_Project/Depth_estimation/workspace/generated_img/%d_depth.jpg' %i, depth_image) 
        #plt.imsave('C:/Users/hunte/OneDrive/Documents/Projects/Major_Project/Depth_estimation/workspace/generated_img/%d_depth.jpg' %i, depth_image, cmap='inferno') 

        plt.imsave('C:/Users/hunte/OneDrive/Documents/Projects/Major_Project/Depth_estimation/workspace/generated_img/%d_image.jpg' %i, rgb_image) 

    
    print("\n Time taken to generate Depth Images: %s " % (time.time() - start_time))

#.....................................................................................................................................................................

if __name__ == "__main__":
    testUNet()

#.....................................................................................................................................................................
    
