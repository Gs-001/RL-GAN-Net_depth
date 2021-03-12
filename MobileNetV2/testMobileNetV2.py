import os
import glob
import time
from PIL import Image
from icecream import ic
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


data = "/home/cse/Documents/group_17/RL-GAN-Net_depth/MobileNetV2/workspace/test_img/same"
pretrained_path = "/home/cse/Documents/group_17/RL-GAN-Net_depth/MobileNetV2/saved_models/99.pth" 

batchSize = 1

depth_dataset = DepthDataset(root_dir=data)

for i in range(len(depth_dataset)):
    sample = depth_dataset[i]
    print("Index:"+ str(i) + "\t\tDimensions: " + str(sample['image'].size))
    # plt.imshow(sample['image'])
    # plt.figure()

    if i == len(depth_dataset):
        # plt.show()
        break


# ### Reading the Test Images
start_time = time.time()

depth_dataset = DepthDataset(root_dir=data,transform=transforms.Compose([ToTensor()]))
train_loader=torch.utils.data.DataLoader(depth_dataset, batchSize)
dataiter = iter(train_loader)
images = dataiter.next()


print("\n Time taken to load Images: %s " % (time.time() - start_time))
print("\n Test Dataset Shape: {shape}".format(shape=np.shape(depth_dataset)))



# ### Importing the Model

from Mobile_model import Model
model = Model().cuda()
model = nn.DataParallel(model)

# Import the Pre-trained Model

model.load_state_dict(torch.load(pretrained_path))
print("\n Loaded MobileNet U-Net Weights successfully\n")

model.eval()


# ### Model Variables (state_dict)

# print("\n\nModel's state_dict:\n\n")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())


# ## Generating Depth Images


start_time = time.time()

for i, sample_batched1  in enumerate (train_loader):

    input_image = torch.autograd.Variable(sample_batched1['image'].cuda())

    # GFV = model.module.encoder(input_image)
    out = model(input_image)

    x = out.detach().cpu().numpy()
    ic(np.shape(x))
    # The Model outputs an image of the shape (1,1,240,320)
    # The output depth image is now upscaled to get a 480 x 640 image

    # NOTE : reshape to half of RGB (or predicted depth) shape
    img = x.reshape(540,960)

    ## Scaling code, not really required, toBeDeleted
    # scale_percent = 200 # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)

    # NOTE : resize to desired size, but make sure to maintain ratio
    depth_image = cv2.resize(img, (1280,720), interpolation = cv2.INTER_AREA)
    rgb_image = sample_batched1['image'].detach().cpu().numpy().reshape(3,1080,1920).transpose(1,2,0)

    # Saving the Images

    #plt.imsave('/home/cse/Documents/group_17/RL-Net_depth/MobileNetV2/workspace/generated_img/%d_depth.png' %i, depth_image) 
    plt.imsave('/home/cse/Documents/group_17/RL-GAN-Net_depth/MobileNetV2/workspace/generated_img/%d_depth.png' %i, depth_image) 

    plt.imsave('/home/cse/Documents/group_17/RL-GAN-Net_depth/MobileNetV2/workspace/generated_img/%d_image.png' %i, rgb_image) 


print("\n Time taken to generate Depth Images: %s " % (time.time() - start_time))


# #### Saving Results in a ZIP File

# import zipfile

# zf = zipfile.ZipFile("myzipfile.zip", "w")
# for dirname, subdirs, files in os.walk("C:/Users/hunte/OneDrive/Documents/Projects/Major_Project/Depth_estimation/workspace/generated_img"):
#     zf.write(dirname)
#     for filename in files:
#         zf.write(os.path.join(dirname, filename))
# zf.close()

