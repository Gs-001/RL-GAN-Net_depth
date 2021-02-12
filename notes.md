# Check these out
- `encoder_out` variable in trainRL.py `Line:383`
- `Generator Output` section, to be replaced
- Losses at 414
- Scam Alert, NLL just returns mean
- action equation, how is it decided ? `line:257` 
- replace *encoder_out* `line:327` with model.predict 
- new function in class envs, `loadTFModels`
- re-evaluate *states Formulation* Section `line:358`
- **loss_GFV is just AE loss** right now

---

# 02 Feb
- changes in line 336, removed `Variable()`call
- to put in `.squeeze()` or not in `agent_input()`
- changed `z-dim` to 128
- detach cpu etc on z

# 03 Feb
- DDPG.py: 50 passing both x and u throught l1
- train AE on depth_cv data
- Add new loss function to RL


# RL commands
- python trainRL.py --pretrained /home/cse/Documents/group_17/RL-GAN-Net_depth/ckpts/shapenet/01-08-18:33/ae_pointnet,Adam,1epochs,b24,lr0.001/checkpoint.pth.tar -d /home/cse/Documents/group_17/RL-GAN-Net_depth/shape_net_core_uniform_samples_2048_split/train

- No Evaluation iterations in trainRL_depth.py: 200 + 177

RGB -> Encoder ..(latent vector/encoding) -> RL Agent -> New Encoding (Z) --> Decoder --> Predicted Depth Image (Computer loss w.r.t target depth image)
 +                                                                                      ^
depth                                                                                   |
  |......................................................................................



 z = RLsomething(encoder_out)
 decoder_out = Decoder(z)
 loss = find_loss(decoder_out, target_image)


# 11FEB
  - changed while loop at 209, converted to tensor

# 13 Feb
  - Saving RL models to pytorch_models_test

# 14 Feb
  - Make changes to testRL_depth.py and test trained RL agent
  - Test the trained AE
  - check trained trainRL_depth.py

