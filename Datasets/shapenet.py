import os.path
import glob
from .util import split2list
from .listdataset import ListDataset
from random import shuffle

def make_dataset(input_dir,split,net_name,target_dir=None):
    # print("make_dataset: ", locals())
    plyfiles = []
    if(net_name== 'GAN'):
        for dirs in os.listdir(input_dir):
            tempDir = os.path.join(input_dir, dirs)
            for input in glob.iglob(os.path.join(tempDir, '*.npy')):
                input = os.path.basename(input)
                root_filename = input[:-4]
                plyinput = dirs + '/' + root_filename + '.npy'
                plyfiles.append([plyinput])


    if(net_name == "auto_encoder"):
        rgb_path = os.path.join(input_dir, 'r-*')
        depth_path = os.path.join(input_dir, 'd-*')

        rgb_images = [os.path.basename(img_path) for img_path in glob.glob(rgb_path)]
        depth_images = [os.path.basename(img_path) for img_path in glob.glob(depth_path)]
        num_of_images = len(rgb_images) if len(rgb_images) <= len(depth_images) else len(depth_images)

        train_dataset = [[rgb_images[i], depth_images[i]] for i in range(num_of_images)]

        return train_dataset, train_dataset




    # if(net_name == 'auto_encoder'):
    #     target_dir = input_dir
    #     for dirs in os.listdir(target_dir):
    #         # print(dirs)
    #         tempDir = os.path.join(input_dir,dirs)
    #         # print(tempDir)
    #         for target in glob.iglob(os.path.join(tempDir,'*.ply')):
    #             target = os.path.basename(target)
    #             # print(target)
    #             root_filename = target[:-4]
    #             plytarget = dirs + '/' + root_filename + '.ply'

    #             # print(plytarget)
    #             # exit()
    #             plyinput = plytarget
    #             plyfiles.append([[plyinput],[plytarget]])

    if (net_name == 'shape_completion'): # TODO remove this sometime ?

        for dirs in os.listdir(input_dir):
            temp_In_Dir = os.path.join(input_dir, dirs)
            temp_Tgt_Dir = os.path.join(target_dir, dirs)

            for target in glob.iglob(os.path.join(temp_In_Dir, '*.ply')):
                target = os.path.basename(target)
                root_filename = target[:-9]
                plytarget = dirs + '/' + root_filename + '.ply'

                plyin = dirs + '/' + target

                plyfiles.append([[plyin], [plytarget]])

    if split== None:
        return plyfiles, plyfiles
    else:
        return split2list(plyfiles, split, default_split=split)

def shapenet(input_root, target_root, split, net_name='auto_encoder', co_transforms= None, input_transforms = None, target_transforms= None, args=None,give_name=False):

    # print("--------------------") 

    

    # print("shapenet: ", locals())

    [train_list,valid_list] = make_dataset(input_root, split,net_name, target_root)

    # print(train_list)
    # print("Dataset Type: \n{}".format(type(train_list[0])))
    # print("Train Dataset: \n {}\n {}".format(train_list[0], train_list[1]))

    train_dataset = ListDataset(input_root,target_root,train_list,net_name, co_transforms, input_transforms, target_transforms,args,mode='train',give_name=give_name)

    shuffle(valid_list)

    valid_dataset = ListDataset(input_root,target_root,valid_list,net_name, co_transforms, input_transforms, target_transforms,args,mode='valid',give_name=give_name)

    return  train_dataset,valid_dataset