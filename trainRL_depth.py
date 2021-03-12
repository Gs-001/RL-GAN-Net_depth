# Written by Muhammad Sarmad
# Date : 23 August 2018

from RL_params import *

import kornia

# import tensorflow as tf
import tensorflow
from glob import glob

from tensorflow.python.keras.models import Model 

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model

from tensorflow.python.keras.preprocessing.image import load_img
import tensorflow as tf

from icecream import ic

import cv2
import matplotlib.pyplot as plt

np.random.seed(5)
# torch.manual_seed(5)

dataset_names = sorted(name for name in Datasets.__all__)
model_names = sorted(name for name in models.__all__)


def evaluate_policy(policy,valid_loader,env,args, eval_episodes=6,render = False):

    print("\n#################################################################")
    print("Validation Phase starting...")
    print("#################################################################\n")

    avg_reward = 0.
    env.reset(epoch_size=len(valid_loader),figures=8) # reset the visdom and set number of figures

    # for i,(input) in enumerate(valid_loader):
    for i in range (0,eval_episodes):
        try:
            imagePair = next(dataloader_iterator)
            # imagePair = torch.from_numpy(imagePair)

            #input = torch.from_numpy(input)
        except:
            dataloader_iterator = iter(valid_loader)
            imagePair = next(dataloader_iterator)
            # imagePair = torch.from_numpy(imagePair)
            #input = torch.from_numpy(input)

        # data_iter = iter(valid_loader)
        # input = data_iter.next()
        # action_rand = torch.randn(args.batch_size, args.z_dim)
        
        obs = env.agent_input(imagePair[0]) # env(input, action_rand)
        done = False

        while not done:
          # Action By Agent and collect reward
            action = policy.select_action(obs)
            # ic(type(action))
            # ic(np.shape(action))
            action = torch.tensor(action).cuda().unsqueeze(dim=0)
            # ic(type(action))
            # ic(np.shape(action))

            new_state, _, reward, done, _,pred_depth = env( imagePair, action,render=render,disp =True)
            #pred_depth = pred_depth.numpy()
            pred_depth = torch.reshape(pred_depth, (pred_depth.shape[3],pred_depth.shape[2])).numpy()##
            cv2.resize(pred_depth, (1280,720), interpolation = cv2.INTER_AREA)
            ic(pred_depth.shape)
            avg_reward += reward

        if i+1 >= eval_episodes:
            break

    avg_reward /= eval_episodes

    plt.imsave('/home/cse/Documents/group_17/pred_depth/pred_depth.png', pred_depth)

    print("\n####################################################")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("######################################################\n")

    return avg_reward

def test_policy(policy,valid_loader,env,args, eval_episodes=12,render = True):

    print("\n#################################################################")
    print("Policy Testing Phase starting...")
    print("#################################################################\n")

    avg_reward = 0.
    env.reset(epoch_size=len(valid_loader),figures=12) # reset the visdom and set number of figures

    #for i,(input) in enumerate(valid_loader):
    for i in range (0,eval_episodes):
        try:
            imagePair = next(dataloader_iterator)
        except:
            dataloader_iterator = iter(valid_loader)
            imagePair = next(dataloader_iterator)

       # data_iter = iter(valid_loader)
       # input = data_iter.next()
        #action_rand = torch.randn(args.batch_size, args.z_dim)
        obs = env.agent_input(imagePair[0])# env(input, action_rand)
        done = False

        while not done:
          # Action By Agent and collect reward
            action = policy.select_action(np.array(obs))
            action= torch.tensor(action).cuda().unsqueeze(dim=0)
            new_state, _, reward, done, _,_ = env( imagePair, action,render=render,disp =True)
            avg_reward += reward

        if i+1 >= eval_episodes:
            break

    avg_reward /= eval_episodes

    print("\n####################################################")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("######################################################\n")

    return avg_reward

def readImages(folder_path, imtype='d'):
    images = []
    for i in range(len(folder_path)):
        img = load_img(folder_path[i])
        img = np.array(img, dtype=np.float32)

        if imtype == 'r':
            img = cv2.resize(img, (640,480), interpolation = cv2.INTER_AREA)
        # TODO remove below break
        if i ==100:
            break

        images.append(img)
    
    images = np.array(images)
    return images

def main(args):
    #
    """ Transforms/ Data Augmentation Tec """
    # co_transforms = pc_transforms.Compose([])

    # input_transforms = transforms.Compose([
    #     pc_transforms.ArrayToTensor()
    # ])

    # target_transforms = transforms.Compose([
    #     pc_transforms.ArrayToTensor()
    # ])


    """##################------------Data Loader##################-----------------"""

    def merge_loader(rgb, depth):
        combined = []
        for i in zip(rgb, depth):
            combined.append(i)

        return combined

    train_path = glob("/home/cse/Documents/group_17/Test Images/combined/rgb/*")
    depth_path = glob("/home/cse/Documents/group_17/Test Images/combined/depth_GH/*")

    print("\n Loading Dataset...")
    start_time = time.time()

    rgb_images = readImages(train_path, 'r')
    depth_images = readImages(depth_path)
    train_loader = merge_loader(rgb_images, depth_images)
    valid_loader = train_loader
    test_loader = train_loader

    print(" Time taken to load dataset: %s" % (time.time() - start_time))

    """----------------Model Settings##################------------"""

    pretrained_AE = torch.load("/home/cse/Documents/group_17/OG_RL-Net_depth/MobileNetV2/saved_models/99.pth")

    # ### Importing the Model

    from MobileNetV2.Mobile_model import Model
    model = Model().cuda()
    model = nn.DataParallel(model)

    # Import the Pre-trained Model

    model.load_state_dict(pretrained_AE)
    print("\n Loaded MobileNet U-Net Weights successfully\n")

    model.eval()
    
    model_encoder = model.module.encoder
    model_decoder = model.module.decoder

    ic("Successfully loaded Encoder, Decoder")
    
    """----------------Error Metrics##################------------"""
    norm = Norm(dims=args.z_dim)
    epoch = 0
    
    test_loss = trainRL(train_loader, valid_loader, test_loader, model_encoder, model_decoder, epoch, args, norm)
    print('Average Loss :{}'.format(test_loss))

def trainRL(train_loader, valid_loader, test_loader, model_encoder, model_decoder, epoch, args, norm):
    ic(epoch)
    epoch_size = len(valid_loader)

    file_name = "%s_%s" % (args.policy_name, args.env_name)

    if args.save_models and not os.path.exists("./pytorch_models_test"):
        os.makedirs("./pytorch_models_test")

    env = envs(args, model_encoder, model_decoder, epoch_size)

    state_dim = args.state_dim
    # ic(state_dim)
    action_dim = args.z_dim
    # ic(action_dim)
    max_action = args.max_action

    # Initialize policy
    if args.policy_name == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "OurDDPG":
        policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action, args.device)

    replay_buffer = utils.ReplayBuffer()
    # evaluations = [evaluate_policy(policy,valid_loader,env,args)]
    evaluations = [evaluate_policy(policy, valid_loader,env,args, render=False)]
    ic.enable()

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    env.reset(epoch_size=len(train_loader))
    ic.enable()

    while total_timesteps < args.max_timesteps:
        if done:
            try:
                dataloader_iterator = iter(train_loader)
                input = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_loader)
                input = next(dataloader_iterator)

            if total_timesteps != 0:
                # print("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward)
                if args.policy_name == "TD3":
                    ic("TD3")
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau,
                                 args.policy_noise, args.noise_clip, args.policy_freq)
                else:
                    # ic("else")
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)
            
            

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq

                # evaluations.append(evaluate_policy(policy,valid_loader,env,args,render = False))

                if args.save_models: policy.save(file_name, directory="./pytorch_models_test")

                env.reset(epoch_size=len(test_loader))
                test_policy(policy, test_loader, env, args, render=True)

                env.reset(epoch_size=len(train_loader))

            # Reset environment
            # obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        obs = env.agent_input(input[0])

        if total_timesteps < args.start_timesteps:
            # action_t = torch.rand(args.batch_size, args.z_dim) # TODO checked rand instead of randn
            action_t = torch.FloatTensor(args.batch_size, args.z_dim).uniform_(-args.max_action, args.max_action)
            action = action_t.detach().cpu().numpy().squeeze(0)
            # obs, _, _, _, _ = env(input, action_t)
        else:
            # action_rand = torch.randn(args.batch_size, args.z_dim)
            #
            # obs, _, _, _, _ = env( input, action_rand)
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=args.z_dim)).clip(-args.max_action*np.ones(args.z_dim,), args.max_action*np.ones(args.z_dim,))
                action = np.float32(action)
            action_t = torch.tensor(action).cuda().unsqueeze(dim=0)

        # Perform action
        # env.render()

        new_obs, _, reward, done, _,_ = env(input, action_t,disp = True)

        # new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == args.max_episodes_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer

        # ic(type(obs))
        # ic(np.shape(obs))
        # ic(type(new_obs))
        # ic(np.shape(new_obs))
        # ic(type(action))
        # ic(np.shape(action))
        # ic(type(reward))
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1


    # for i,(input) in enumerate(valid_loader):
    #
    #
    #
    #      if np.shape(input)[0]< args.batch_size:
    #         break;#print(np.shape(input)[0])
    #
    #      action = torch.randn(args.batch_size, args.z_dim)
    #      action_np = action.detach().cpu().numpy()
    #      new_state, _, reward,done, _ = env1(i,input,action)
    #
    #
    #
    # return reward

class envs(nn.Module):
    def __init__(self,args,model_encoder,model_decoder,epoch_size):
        super(envs,self).__init__()

        self.norm = Norm(dims=args.z_dim)
        self.epoch = 0
        self.epoch_size =epoch_size

        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.j = 1
        self.figures = 3
        self.attempts = args.attempts
        self.end = time.time()
        self.batch_time = AverageMeter()
        self.lossess = AverageMeter()
        self.attempt_id =0
        self.state_prev = np.zeros([4,])
        self.iter = 0
    
    def reset(self,epoch_size,figures =3):
        self.j = 1
        self.i = 0
        self.figures = figures
        self.epoch_size= epoch_size
    
    def agent_input(self,input):
        with torch.no_grad():
            input = torch.from_numpy(input)
            input = torch.reshape(input,(-1,input.shape[2], input.shape[0], input.shape[1]))
            encoder_out = self.model_encoder(input.cuda())            
        return encoder_out[-1].cpu().data.numpy().ravel()
    
    def forward(self,input,action,render=False, disp=False):
        with torch.no_grad():
            # Encoder Input
            input_rgb = torch.reshape(torch.from_numpy(input[0]),(-1,input[0].shape[2], input[0].shape[0], input[0].shape[1]))
            target_depth = torch.reshape(torch.from_numpy(input[1]),(-1,input[1].shape[2], input[1].shape[0], input[1].shape[1]))

            #ic(input_rgb.shape)
            #ic(target_depth.shape)

            #ic('exiting')
            
            # Encoder  output
            encoder_out = self.model_encoder(input_rgb.cuda())

            # RL Generated Action
            z = Variable(action, requires_grad=True).cuda()

            # Reshape z to dimensions of last element of encoding
            z = torch.reshape(z, (encoder_out[-1].shape[0],encoder_out[-1].shape[1],encoder_out[-1].shape[2],encoder_out[-1].shape[3]))

            # Replacing original encoding with RL agent output
            encoder_out[-1] = z

            # Decoder Output
            pred_depth = self.model_decoder(encoder_out)
            
            pred_depth = pred_depth.reshape(240,320)
            pred_depth = pred_depth.detach().cpu().numpy()

            # Upscaling Predicted Depth to 720p
            pred_depth = cv2.resize(pred_depth, (480,640), interpolation = cv2.INTER_AREA)
            
            # Coverting 3 Channel Depth to Single Channel
            gray_truth = cv2.cvtColor(input[1], cv2.COLOR_RGB2GRAY)
            # ic(np.shape(gray_truth))
            gray_truth = gray_truth.transpose()
            # ic(np.shape(gray_truth))
            gray_truth = cv2.resize(gray_truth, (480,640), interpolation = cv2.INTER_AREA)
            # ic(np.shape(gray_truth))
            
            # Reshape?
            gray_truth = gray_truth.reshape(-1, np.shape(gray_truth)[0], np.shape(gray_truth)[1])
            pred_depth = pred_depth.reshape(-1, np.shape(pred_depth)[0], np.shape(pred_depth)[1])
            # ic(type(gray_truth))
            # ic(gray_truth.shape)
            
            # Converting Depth Images to Torch Tensors
            gray_truth = torch.from_numpy(gray_truth).reshape(-1, np.shape(gray_truth)[0], np.shape(gray_truth)[1], np.shape(gray_truth)[2])
            pred_depth = torch.from_numpy(pred_depth).reshape(-1, np.shape(pred_depth)[0], np.shape(pred_depth)[1], np.shape(pred_depth)[2])

            # ic(type(pred_depth))
            # ic(pred_depth.shape)
            # ic(type(gray_truth))
            # ic(gray_truth.shape)
            # ic('################## CKPT : Saving Images')
            # import matplotlib.pyplot as plt
            # plt.imsave('/home/cse/Documents/group_17/gray_truth.png', gray_truth)
            # plt.imsave('/home/cse/Documents/group_17/pred_depth.png', pred_depth)

        # Compute the Losses

        # SSIM Loss

        l1_criterion = nn.L1Loss()

        def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
            ssim = kornia.losses.SSIM(window_size=11,max_val=val_range,reduction='none')
            return ssim(img1, img2)

        def compute_errors(gt, pred):
            gt = gt.numpy()
            pred = pred.numpy()
            thresh = np.maximum((gt / pred), (pred / gt))
            δ1 = (thresh < 1.25   ).mean()
            rmse = np.sqrt(np.mean((gt - pred) ** 2))
            for i in gt[0][0]:
                i[i<0.7]=0.7
            log10_err = np.mean(np.absolute(np.log10(gt) - np.log10(pred)))
            return rmse, log10_err, δ1

        # ic(pred_depth.shape)
        # ic(gray_truth.shape)
        
        l_depth = l1_criterion(pred_depth, gray_truth)
        l_ssim = torch.clamp((1 - ssim(pred_depth, gray_truth, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

        batch_size = 1

        rmse        = np.zeros(batch_size, np.float32)
        log10_err   = np.zeros(batch_size, np.float32)
        δ1          = np.zeros(batch_size, np.float32)

        rmse, log10_err, δ1  = compute_errors(gray_truth, pred_depth) 


        loss = (1.0 * l_ssim.mean().item()) + (0.1 * l_depth)
        # ic(loss)
        
        # State Formulation
        state_curr = np.array([loss.cpu().data.numpy(),rmse,log10_err,δ1])
        reward_SSIM =- state_curr[0]
        reward_rmse =- state_curr[1]
        reward_log  =- state_curr[2]
        reward_delta = state_curr[3]
        
        # Reward Formulation
        reward = (reward_SSIM*100+reward_rmse*1+reward_log*1+reward_delta*1)

        # measured elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()    
    
        
        if disp:
            print('[{4}][{0}/{1}]\t Reward: {2}\t States: {3}'.format(self.i, self.epoch_size,reward, state_curr,self.iter))
            # print('[{}][{}/{}]\t Reward: {}\t States: {}\t  MinSoFar: {}\t  MaxSoFar: {}'.format(self.iter, self.i, self.epoch_size, reward, state_curr, round(min_reward, 2), round(max_reward, 2)))
            self.i += 1
            if(self.i>=self.epoch_size):
                self.i=0
                self.iter +=1

        done = True

        # The New State is the improved encoding
        #state = encoder_out
        state = encoder_out[-1].cpu().data.numpy().ravel()

        return state, None, reward, done, self.lossess.avg, pred_depth

if __name__ == '__main__':
    args = get_parameters()
    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")  # for selecting device for chamfer loss
    torch.cuda.set_device(args.gpu_id)
    print('Using TITAN XP GPU # :', torch.cuda.current_device())
    # print(args)
    main(args)







