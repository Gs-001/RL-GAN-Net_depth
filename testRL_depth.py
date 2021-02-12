# Written by Muhammad 'Tintan' Sarmad
# Date : 23 August 2018

from RL_params import *

np.random.seed(5)
#torch.manual_seed(5)

dataset_names = sorted(name for name in Datasets.__all__)
model_names = sorted(name for name in models.__all__)

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
    """ Transforms/ Data Augmentation Techniques """
    co_transforms = pc_transforms.Compose([])

    input_transforms = transforms.Compose([
        pc_transforms.ArrayToTensor()
    ])

    target_transforms = transforms.Compose([
        pc_transforms.ArrayToTensor()
    ])

    """-----------------------------------------------Data Loader----------------------------------------------------"""

    """----------------Model Settings-----------------------------------------------"""
    
    def merge_loader(rgb, depth):
        combined = []
        for i in zip(rgb, depth):
            combined.append(i)

        return combined

    if (args.net_name == 'auto_encoder'):

        train_path = glob("/home/cse/Documents/group_17/RL-GAN-Net_depth/rgb/rgb/*")
        depth_path = glob("/home/cse/Documents/group_17/RL-GAN-Net_depth/target_images/current/depth_cv/*")

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

    
    network_data_Actor = torch.load(args.pretrained_Actor)
    network_data_Critic = torch.load(args.pretrained_Critic)

    model_actor = models.__dict__['actor_net'](args, data=network_data_Actor).cuda()
    model_critic = models.__dict__['critic_net'](args, data=network_data_Critic).cuda()

    """----------------Error Metrics-----------------------------------------------"""
    nll = NLL()
    mse = MSE(reduction = 'elementwise_mean')
    norm = Norm(dims=args.z_dim)
    epoch = 0

    test_loss = testRL(test_loader, model_encoder, model_decoder, model_actor,model_critic,epoch, args, nll, mse, norm)
    print('Average Loss :{}'.format(test_loss))


def testRL(test_loader,model_encoder,model_decoder, model_actor,model_critic,epoch,args,nll, mse,norm):
    # *** .eval() is similar to forward function (?)
    # model_encoder.eval()
    # model_decoder.eval()
    # model_actor.eval()
    # model_critic.eval()

    epoch_size = len(test_loader)

    env = envs(args, model_encoder, model_decoder, epoch_size)

    for i, (input,fname) in enumerate(test_loader):
        obs = env.agent_input(input)                # env(input, action_rand)
        done = False
        while not done:
            # Action By Agent and collect reward
            action = model_actor(np.array(obs))     # policy.select_action(np.array(obs))
            action = torch.tensor(action).cuda().unsqueeze(dim=0)
            new_state, _, reward, done, _ = env(input, action, render=True, disp=True,fname=fname, filenum = str(i))
            # action # sarmad you c*nt


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
        self.i = 0;
        self.figures = 25
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
            input = input.cuda(async=True)
            input_var = Variable(input, requires_grad=True)
            encoder_out = self.model_encoder(input_var, )
            out = encoder_out.detach().cpu().numpy().squeeze()
        return out

    def forward(self,input,action,render=False, disp=False,fname=None,filenum = None):
        with torch.no_grad():
            # Encoder Input
            input = input.cuda(async=True)
            input_var = Variable(input, requires_grad=True)

            # Encoder  output
            encoder_out = self.model_encoder(input_var, )

            # *** RL generated GFV
            z = Variable(action, requires_grad=True).cuda()

            # *** Note for self
            depth_image = self.model_decoder(z)
            # *** TODO use reconstuction code, and save the image 
        
        # LOSSES ------------------------------------------------------------------
        # ***TODO variable loss_GFV is only AE loss
        # loss_GFV = 10*self.mse(out_G, encoder_out) # ORIGINAL
        loss_GFV = 10*self.mse(encoder_out)

        # Norm Loss
        loss_norm = 0.1*self.norm(z)

        # States Formulation -------------------------------------------------------
        state_curr = np.array([loss_GFV.cpu().data.numpy(), loss_norm.cpu().data.numpy()])
        # state_prev = self.state_prev

        reward_GFV =- state_curr[0]     # -state_curr[1] + self.state_prev[1]
        reward_norm =- state_curr[1]    # - state_curr[3] + self.state_prev[3]
        
        # Reward Formulation --------------------------------------------------------
        reward = reward_D + reward_GFV  
        
        # reward = reward * 100
        # self.state_prev = state_curr

        # measured elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()

        # *** all below is probably visdom
        # # --------------------------------------------------------
        # test1 = trans_input_temp.detach().cpu().numpy()
        # test2 = pc_1_temp.detach().cpu().numpy()
        # test3 = pc_1_G_temp.detach().cpu().numpy()

        # fname = fname[0]
        # if  not os.path.exists('test/'+fname[:8]):
        #     os.makedirs('test/'+fname[:8])


        # np.savetxt('test/'+fname[:9]+filenum+'_input.xyz', np.c_[test1[0,:],test1[1,:],test1[2,:]],  header='x y z', fmt='%1.6f',
        #        delimiter=' ')
        # np.savetxt('test/' +fname[:9]+ filenum + '_AE.xyz', np.c_[test2[0, :], test2[1, :], test2[2, :]], header='x y z', fmt='%1.6f',
        #            delimiter=' ')
        # np.savetxt('test/' +fname[:9]+ filenum + '_agent.xyz', np.c_[test3[0, :], test3[1, :], test3[2, :]], header='x y z', fmt='%1.6f',
        #            delimiter=' ')
        # visuals = OrderedDict(
        #     [('Input_pc', trans_input_temp.detach().cpu().numpy()),
        #      ('AE Predicted_pc', pc_1_temp.detach().cpu().numpy()),
        #      ('GAN Generated_pc', pc_1_G_temp.detach().cpu().numpy())])
        
        # if render==True and self.j <= self.figures:
        #  vis_Valida[self.j].display_current_results(visuals, self.epoch, self.i)
        #  self.j += 1

        if disp:
            print('[{4}][{0}/{1}]\t Reward: {2}\t States: {3}'.format(self.i, self.epoch_size,reward,state_curr,self.iter))
            self.i += 1
            if(self.i>=self.epoch_size):
                self.i=0
                self.iter +=1

        done = True
        # *** switching to encoder_out, then z
        state = z.detach().cpu().data.numpy().squeeze()
        return state, _, reward, done, self.lossess.avg



if __name__ == '__main__':
    args = get_parameters()
    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")  # for selecting device for chamfer loss
    torch.cuda.set_device(args.gpu_id)
    print('Using TITAN XP GPU # :', torch.cuda.current_device())
    print(args)
    main(args)







