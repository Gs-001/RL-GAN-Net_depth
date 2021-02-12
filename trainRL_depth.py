# Written by Muhammad Sarmad
# Date : 23 August 2018

from RL_params import *

# import tensorflow as tf
import tensorflow
from glob import glob

from tensorflow.python.keras.models import Model 

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model

from tensorflow.python.keras.preprocessing.image import load_img
import tensorflow as tf

from icecream import ic

np.random.seed(5)
# torch.manual_seed(5)

dataset_names = sorted(name for name in Datasets.__all__)
model_names = sorted(name for name in models.__all__)


max_reward = -10000000000
min_reward = 10000000000

def evaluate_policy(policy,valid_loader,env,args, eval_episodes=6,render = False):
    avg_reward = 0.
    env.reset(epoch_size=len(valid_loader),figures=8) # reset the visdom and set number of figures

    # for i,(input) in enumerate(valid_loader):
    for i in range (0,eval_episodes):
        try:
            input = next(dataloader_iterator)
            input = tf.convert_to_tensor(input)
            #input = torch.from_numpy(input)
        except:
            dataloader_iterator = iter(valid_loader)
            input = next(dataloader_iterator)
            input = tf.convert_to_tensor(input)
            #input = torch.from_numpy(input)

        # data_iter = iter(valid_loader)
        # input = data_iter.next()
        # action_rand = torch.randn(args.batch_size, args.z_dim)
        obs = env.agent_input(input[0]) # env(input, action_rand)
        done = False

        while not done:
          # Action By Agent and collect reward
            action = policy.select_action(np.array(obs))
            action= torch.tensor(action).cuda().unsqueeze(dim=0)
            new_state, _, reward, done, _ = env( input, action,render=render,disp =True)
            avg_reward += reward

        if i+1 >= eval_episodes:
            break

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")

    return avg_reward

def test_policy(policy,valid_loader,env,args, eval_episodes=12,render = True):
    avg_reward = 0.
    env.reset(epoch_size=len(valid_loader),figures=12) # reset the visdom and set number of figures

    #for i,(input) in enumerate(valid_loader):
    for i in range (0,eval_episodes):
        try:
            input = next(dataloader_iterator)
        except:
            dataloader_iterator = iter(valid_loader)
            input = next(dataloader_iterator)

       # data_iter = iter(valid_loader)
       # input = data_iter.next()
        #action_rand = torch.randn(args.batch_size, args.z_dim)
        obs =env.agent_input(input[0])# env(input, action_rand)
        done = False

        while not done:
          # Action By Agent and collect reward
            action = policy.select_action(np.array(obs))
            action= torch.tensor(action).cuda().unsqueeze(dim=0)
            new_state, _, reward, done, _ = env( input, action,render=render,disp =True)
            avg_reward += reward

        if i+1 >= eval_episodes:
            break

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")

    return avg_reward

def readImages(folder_path):
    images = []
    for i in range(len(folder_path)):
        img = load_img(folder_path[i], target_size=(224, 224, 3))
        # img = Image.open(folder_path[i])
        img = np.array(img, dtype=np.float32)
        images.append(img)
    
    images = np.array(images)
    return images

def main(args):
    #
    """ Transforms/ Data Augmentation Tec """
    co_transforms = pc_transforms.Compose([])

    input_transforms = transforms.Compose([
        pc_transforms.ArrayToTensor()
    ])

    target_transforms = transforms.Compose([
        pc_transforms.ArrayToTensor()
    ])


    """-----------------------------------------------Data Loader----------------------------------------------------"""

    def merge_loader(rgb, depth):
        combined = []
        for i in zip(rgb, depth):
            combined.append(i)

        return combined

    if (args.net_name == 'auto_encoder'):

        train_path = glob("/home/cse/Documents/group_17/NYU_Depth_V2/basements/train/rgb/*")
        depth_path = glob("/home/cse/Documents/group_17/NYU_Depth_V2/basements/train/depth_jpg/*")

        rgb_images = readImages(train_path)
        depth_images = readImages(depth_path)
        train_loader = merge_loader(rgb_images, depth_images)
        valid_loader = train_loader
        test_loader = train_loader

    """----------------Model Settings-----------------------------------------------"""

    print('Encoder Model: {0}, Decoder Model : {1}'.format(args.model_encoder,args.model_decoder))

    pretrained_AE = load_model(os.path.join(os.path.dirname(__file__), "AE", 'trainedAutoencoder'))
    model_encoder = load_model(os.path.join(os.path.dirname(__file__),"AE", 'trainedEncoder'))
    model_decoder = load_model(os.path.join(os.path.dirname(__file__), "AE", 'trainedDecoderr'))

    """----------------Error Metrics-----------------------------------------------"""
    nll = NLL()
    mse = MSE(reduction = 'elementwise_mean')
    norm = Norm(dims=args.z_dim)
    epoch = 0

    test_loss = trainRL(train_loader, valid_loader, test_loader, model_encoder, model_decoder, epoch, args, nll, mse, norm)
    print('Average Loss :{}'.format(test_loss))

def trainRL(train_loader, valid_loader, test_loader, model_encoder, model_decoder, epoch, args, nll, mse, norm):
    epoch_size = len(valid_loader)

    file_name = "%s_%s" % (args.policy_name, args.env_name)

    if args.save_models and not os.path.exists("./pytorch_models_test"):
        os.makedirs("./pytorch_models_test")

    env = envs(args, model_encoder, model_decoder, epoch_size)

    state_dim = args.state_dim
    action_dim = args.z_dim
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

    print("$$$ BRUCE IS BATMAN ----------------------------")

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    env.reset(epoch_size=len(train_loader))

    while total_timesteps < args.max_timesteps:
        if done:
            try:
                input = next(dataloader_iterator)
                input = tf.convert_to_tensor(input)  
            except:
                dataloader_iterator = iter(train_loader)
                input = next(dataloader_iterator)
                input = tf.convert_to_tensor(input)

            if total_timesteps != 0:
                # print("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward)
                if args.policy_name == "TD3":
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau,
                                 args.policy_noise, args.noise_clip, args.policy_freq)
                else:
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq

                evaluations.append(evaluate_policy(policy,valid_loader,env,args,render = False))

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
        new_obs, _, reward, done, _ = env(input, action_t,disp = True)

        # new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == args.max_episodes_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
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

        self.nll = NLL()
        self.mse = MSE(reduction='elementwise_mean')
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
        self.j = 1;
        self.i = 0;
        self.figures = figures;
        self.epoch_size= epoch_size
    
    def agent_input(self,input):
        with torch.no_grad():
            input = tf.reshape(input,(-1,input.shape[0], input.shape[1], input.shape[2]))
            encoder_out = self.model_encoder.predict(input).squeeze()
        return encoder_out
    
    def forward(self,input,action,render=False, disp=False):
        with torch.no_grad():
            # Encoder Input
            input_var = tf.reshape(input[0],(-1,input[0].shape[0], input[0].shape[1], input[0].shape[2]))
            target_var = tf.reshape(input[1],(-1,input[1].shape[0], input[1].shape[1], input[1].shape[2]))
        
            # Encoder  output
            encoder_out = self.model_encoder.predict(input_var)
            encoder_out = encoder_out[0]

            # print(f"action shape: {np.shape(action)}")

            # Generator Input
            z = Variable(action, requires_grad=True).cuda()
            z = z.detach().cpu().numpy().squeeze()

            # D Decoder Output
            # new_z = np.reshape(z, 1, np.shape(z))
            new_z = np.expand_dims(z, axis=0)
            # print(f"z shape: {np.shape(z)}")
            # print(f"new z shape: {np.shape(new_z)}")
            pred_depth = self.model_decoder.predict(new_z)
            
        # LOSSES ------------------------------------------------------------------

        # *** TODO variable loss_GFV is only AE loss

        ic()

        loss_GFV = self.mse(torch.from_numpy(pred_depth), torch.from_numpy(input[1].numpy()))   # saving to pytorch_models_test
        # loss_GFV = self.mse(torch.from_numpy(pred_depth), torch.from_numpy(input[1]))
        # loss_GFV = self.mse(torch.from_numpy(pred_depth), torch.from_numpy(pred_depth))
        # loss_GFV = self.mse(torch.from_numpy(input[1]), torch.from_numpy(input[1]))

        # loss_GFV = self.mse(torch.from_numpy(pred_depth), torch.from_numpy(target_var.numpy()))

        # ic(type(pred_depth))
        # ic(type(input[1].numpy()))
        # ic(type(target_var.numpy()))
        
        # Norm Loss
        loss_norm = self.norm(torch.from_numpy(z))

        # States Formulation -------------------------------------------------------
        state_curr = np.array([loss_GFV.cpu().data.numpy(), loss_norm.cpu().data.numpy()])
        # state_prev = self.state_prev

        reward_GFV =- state_curr[0]      # - state_curr[1] + self.state_prev[1]
        reward_norm =- state_curr[1]     # - state_curr[3] + self.state_prev[3]
        
        # Reward Formulation --------------------------------------------------------
        reward = (reward_GFV * 10.0 + reward_norm*1/10)      
        # if(reward < min_reward):
        #     min_reward = reward
        # if(reward > max_reward):
        #     max_reward = reward
        
        # reward = reward * 100
        # self.state_prev = state_curr

        # measured elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()    
    
        if disp:
            print('[{4}][{0}/{1}]\t Reward: {2}\t States: {3}'.format(self.i, self.epoch_size,reward,state_curr,self.iter))
            # print('[{}][{}/{}]\t Reward: {}\t States: {}\t  MinSoFar: {}\t  MaxSoFar: {}'.format(self.iter, self.i, self.epoch_size, reward, state_curr, round(min_reward, 2), round(max_reward, 2)))
            self.i += 1
            if(self.i>=self.epoch_size):
                self.i=0
                self.iter +=1

        done = True

        # *** Switching it up, with encoder_out, then with z
        state = z

        return state, None, reward, done, self.lossess.avg

if __name__ == '__main__':
    args = get_parameters()
    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")  # for selecting device for chamfer loss
    torch.cuda.set_device(args.gpu_id)
    print('Using TITAN XP GPU # :', torch.cuda.current_device())
    print(args)
    main(args)







