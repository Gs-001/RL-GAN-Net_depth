# import torch.utils.data as data
# import os
# import os.path
# #from plyfile import PlyData, PlyElement
# from Datasets.plyfile.plyfile import PlyData
# import numpy as np

# from PIL import Image

# def image_loader(root, image_path):
#     Image.open(os.path.join(root, image_path))
    

# class DepthDataset(data.Dataset):

#     def __init__(self, root, data_list):
#         self.root = root
#         self.data_list = data_list
#         self.loader = image_loader

#     def __getitem__(self, index):
#         rgb_path, depth_path = self.data_list[index]

#         rgb_image = self.loader(self.root, rgb_path, 'rgb')
#         depth_image = self.loader(self.root, depth_path, 'depth')

#         return rgb_image, depth_image

#     def __len__(self):
#         return len(self.data_list)




# from torchvision.datasets.folder import default_loader
# from torchvision.datasets.vision import VisionDataset
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
import glob
import os


def make_dataset(root):
    path = os.path.join(root, 'r-*')
    rgb_images = [os.path.basename(img_path) for img_path in glob.glob(path)]
    depth_images = [os.path.basename(img_path) for img_path in glob.glob(path)]
    num_of_images = len(rgb_images) if len(rgb_images) <= len(depth_images) else len(depth_images)

    return [(rgb_images[i], depth_images[i]) for i in range(num_of_images)]


def image_loader(root, image_path):
    return torch.from_numpy(np.array(Image.open(os.path.join(root, image_path))))

class DepthDataset(data.Dataset):

    def __init__(self, root, loader=None, rgb_transform=None, gt_transform=None):
        # super().__init__(root, transform=rgb_transform, target_transform=gt_transform)

        self.root = root
        self.image_pairs = make_dataset(self.root)
        self.loader = image_loader

    def __getitem__(self, index):
        """Returns a data sample from our dataset.
        """
        rgb_path, depth_path = self.image_pairs[index]

        # import each image using loader (by default it's PIL)
        rgb_image = self.loader(self.root, rgb_path)
        depth_image = self.loader(self.root, depth_path)

        # here goes tranforms if needed
        # maybe we need different tranforms for each type of image
        # if self.transform is not None:
        #     rgb_sample = self.transform(rgb_sample)
        # if self.target_transform is not None:
        #     gt_sample = self.target_transform(gt_sample)      

        # now we return the right imported pair of images (tensors)
        return rgb_image, depth_image

    def __len__(self):
        return len(self.image_pairs)
