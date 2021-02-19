# RL-Depth-Net
This repository is a successor to the original CVPR 2019 Paper : [RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion](https://arxiv.org/abs/1904.12304). 

We propose a novel approach of replacing backpropogation of an Autoencoder by a Reinforcement Learning agent, for Monocular Depth Estimation from RGB Images.


### Prerequisites

* Python 3.6
* Linux / Windows
* Anaconda (optional)

The packages for the project  are listed in requirements_conda.txt and requirements_pip.txt files. Only install the ones needed or you can clone the whole environment. 


### Steps

1. Download depth data from [Link]()
2. Process Depth Images to remove noise
3. Train the MobileNetV2 model on depth data using trainMobileNetV2.ipynb
4. Train RL by using pre-trained MobileNetV2 model, by running trainRL.py
5. Test with new RGB images by running testRL.py

### Credits:
1. https://github.com/sfujim/TD3



If you use this work for your projects, please take the time to cite the original CVPR paper:

```
@InProceedings{Sarmad_2019_CVPR,
author = {Sarmad, Muhammad and Lee, Hyunjoo Jenny and Kim, Young Min},
title = {RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
