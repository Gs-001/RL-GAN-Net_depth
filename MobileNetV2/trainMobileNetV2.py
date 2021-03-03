
# coding: utf-8

# # Script to train a MobileNetV2 based U-Net to generate Depth Images


import pandas as pd
import numpy as np
import torch
import os
import cv2
import kornia 
from glob import glob
from PIL import Image
from io import BytesIO
import random
from zipfile import ZipFile

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.utils import shuffle
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from DepthData_mob import DepthDataset
from DepthData_mob import Augmentation
from DepthData_mob import ToTensor


# ## Hyperparameters


epochs=100
lr=0.0001
batch_size=16


# ## Read Train Dataset

rgb_path = "/home/cse/Documents/group_17/Test Images/combined/rgb/"
# rgb_path = "/home/cse/Documents/group_17/RL-GAN-Net_depth/Depth_Dataset/train/rgb_100"
# rgb_path = "/home/cse/Documents/group_17/Test Images/720pTest/rgb/_resized/"

depth_path = "/home/cse/Documents/group_17/Test Images/combined/depth_GH/"
# depth_path = "/home/cse/Documents/group_17/RL-GAN-Net_depth/Depth_Dataset/train/depth_100"
# depth_path = "/home/cse/Documents/group_17/Test Images/720pTest/depth_GH/"

def prepare_dataset():
    rgb_files = os.listdir(rgb_path)
    depth_files = os.listdir(depth_path)
    rgb_files.sort()
    depth_files.sort()

    dataset_files_list = [image_pair for image_pair in zip(rgb_files, depth_files)]

    # dataset_files_list = []
    # i=0
    # for image_pair in zip(rgb_files, depth_files):
    #     dataset_files_list.append(image_pair)
    #     i+=1
    #     if i == 64:
    #         break
    return dataset_files_list

def prepare_test_dataset():
    # return first 10 images
    rgb_files = os.listdir(rgb_path)
    dataset_files_list = []
    i=0
    for rgbImg in rgb_files:
        dataset_files_list.append(rgbImg)
        i+=1
        if i == 10:
            break
    return dataset_files_list

traincsv = prepare_dataset()

#loading the mobilNetDepth model
from Mobile_model import Model


from icecream import ic

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    ssim = kornia.losses.SSIM(window_size=11,max_val=val_range,reduction='none')
    return ssim(img1, img2)


import matplotlib
import matplotlib.cm
import numpy as np


def DepthNorm(depth, maxDepth=1000.0):
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1))


# In[6]:


def LogProgress(model, writer, test_loader, epoch):
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
    if epoch == 0: 
        writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 0: 
        writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)
    output = DepthNorm( model(image) )
    writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
    writer.add_image('Train.3.Diff', colorize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)), epoch)
    del image
    del depth
    del output
    

import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
from tensorboardX import SummaryWriter

writer = SummaryWriter()

# from data import getTrainingTestingData
# from utils import AverageMeter, DepthNorm, colorize

# 

model = Model().cuda()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)
#load trained model if needed
#model.load_state_dict(torch.load('/workspace/1.pth'))
print('Model created.')


# [['rr-1.ppm', 'd-1.jpg'], [], []]

depth_dataset = DepthDataset(traincsv=traincsv, transform=transforms.Compose([Augmentation(0.5),ToTensor()]))
train_loader = DataLoader(depth_dataset, batch_size, shuffle=True)

## Additions
testcsv = prepare_test_dataset()
test_dataset = DepthDataset(traincsv=traincsv, transform=transforms.Compose([Augmentation(0.5),ToTensor()]))
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

l1_criterion = nn.L1Loss()

optimizer = torch.optim.Adam( model.parameters(), lr )

# Start training...
for epoch in range(epochs):
    # path='/workspace/'+str(epoch)+'.pth'        
    path='/home/cse/Documents/group_17/RL-GAN-Net_depth/MobileNetV2/saved_models/' + str(epoch) + '.pth' 
    torch.save(model.state_dict(), path)
    batch_time = AverageMeter()
    losses = AverageMeter()
    N = len(train_loader)

    # Switch to train mode
    model.train()

    end = time.time()
    ic("Running New Epoch...", epoch)
    for i, sample_batched in enumerate(train_loader):
        optimizer.zero_grad()

        #Prepare sample and target
        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

        # Normalize depth
        depth_n = DepthNorm( depth )

        ic(np.shape(image))        
        # Predict
        output = model(image)
        ic(np.shape (output))

        # Compute the loss
        l_depth = l1_criterion(output, depth_n)
        l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

        loss = (1.0 * l_ssim.mean().item()) + (0.1 * l_depth)

        # Update step
       
        losses.update(loss.data.item(), image.size(0))
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))

        # Log progress
        niter = epoch*N+i
        if i % 5 == 0:
            # Print to console
            print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
            'ETA {eta}\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})'
            .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

            # Log to tensorboard
            #writer.add_scalar('Train/Loss', losses.val, niter)

            # path='/workspace/'+str(epoch)+'.pth'
            path='/home/cse/Documents/group_17/RL-GAN-Net_depth/MobileNetV2/saved_models/'+str(epoch)+'.pth'
            torch.save(model.state_dict(), path)    
        if i % 300 == 0:
            LogProgress(model, writer, test_loader, niter)

    # Record epoch's intermediate results
    LogProgress(model, writer, test_loader, niter)
    writer.add_scalar('Train/Loss.avg', losses.avg, epoch)



# In[25]:


#Evaluations


# model = Model().cuda()
# model = nn.DataParallel(model)
#load the model if needed
#model.load_state_dict(torch.load('/workspace/3.pth'))
# model.eval()
# batch_size=1

# depth_dataset = DepthDataset(traincsv=traincsv, 
#                                 root_dir='/workspace/',
#                                 transform=transforms.Compose([Augmentation(0.5),ToTensor()]))

# train_loader = DataLoader(depth_dataset, batch_size, shuffle=True)


# for sample_batched1  in (train_loader):
#     image1 = torch.autograd.Variable(sample_batched1['image'].cuda())
    
#     outtt=model(image1 )
#     break

    


# In[ ]:


#ploting the evaluated images

# x=outtt.detach().cpu().numpy()
# ic(x.shape)
# x=x.reshape(240,320)
# plt.imshow(x)
# plt.figure()
# plt.imshow(sample_batched1['image'].detach().cpu().numpy().reshape(3,480,640).transpose(1,2,0))

