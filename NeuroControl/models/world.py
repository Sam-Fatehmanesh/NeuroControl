import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NeuroControl.models.neuraloperator import FNOPredictor
from NeuroControl.models.transcnn import TransCNN
from NeuroControl.models.cnn import CNNLayer, DeCNNLayer
from NeuroControl.models.transformer import Transformer
from NeuroControl.models.encoder import SpikePositionEncoding
from NeuroControl.models.moe import MoEPredictor
from NeuroControl.models.mlp import MLP
import pdb
import csv
from soft_moe_pytorch import SoftMoE, DynamicSlotsSoftMoE
from mamba_ssm import Mamba2



# A transformer based world model
class WorldModelT(nn.Module):
    def __init__(self, image_n, num_frames_per_step, latent_size, state_size, action_dim, cnn_kernel_size=3):
        super(WorldModelT, self).__init__()

        self.num_frames_per_step = num_frames_per_step
        self.image_n = image_n

        self.encoder_CNN_out_ch = 4

        num_2pools = 3

        self.encoder_CNN = nn.Sequential(
            CNNLayer(1, 8, cnn_kernel_size),
            CNNLayer(8, 64, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(64, 128, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(128, 32, cnn_kernel_size),
             nn.MaxPool2d(2, stride=2),
            CNNLayer(32, self.encoder_CNN_out_ch, cnn_kernel_size),
        )
        self.postCNN_image_n = int(image_n/(2**num_2pools))
        pixles_num = int(self.postCNN_image_n**2)
        self.per_image_dim = self.encoder_CNN_out_ch * pixles_num
        self.predim = self.per_image_dim * num_frames_per_step
        self.transformer_in_dim = self.predim + state_size + action_dim
        
        self.state_decoder = nn.Linear(state_size, self.per_image_dim)
        self.state_encoder = nn.Linear(latent_size, state_size)

        self.action_decoder = nn.Linear(action_dim, self.per_image_dim)
        self.action_flat = nn.Flatten(start_dim=0)

        self.flat = nn.Flatten()

        # Latent Representation Transformer
        self.lr_transformer = Transformer(self.per_image_dim, latent_size, heads=8, ff_dim=latent_size*2, out_dim=latent_size, layer_num=8)

        self.decoder_transformer = Transformer(latent_size, latent_size, heads=8, ff_dim=latent_size*2, out_dim=pixles_num, layer_num=4)
        # Uses deconvolutions to generate an image
        # Upscales from 35x35 to 280x280
        self.decoder_DCNN = nn.Sequential(
            DeCNNLayer(1, 32, kernel_size=4, stride=2, padding=1),
            DeCNNLayer(32, 8, kernel_size=4, stride=2, padding=1),
            DeCNNLayer(8, 1, kernel_size=4, stride=2, padding=1),
            #DeCNNLayer(8, 1, kernel_size=4, stride=2, padding=1),
        )
        # self.decoder_DCNN = nn.Sequential(
        #     CNNLayer(1, 32, cnn_kernel_size),
        #     CNNLayer(32, 128, cnn_kernel_size),
        #     CNNLayer(128, 64, cnn_kernel_size),
        #     CNNLayer(64, 8, cnn_kernel_size),
        #     CNNLayer(8, 1, cnn_kernel_size),
        # )


    def forward(self, observation_t, action_t, state_t):
        batch_dim, _, image_n, _ = observation_t.shape
        
        xt = self.encoder_CNN(observation_t)
        
        xt = self.flat(xt)
        
        #xt = xt.view(1,1, batch_dim, self.per_image_dim)
        
        state_t = self.state_decoder(state_t).view(1, self.per_image_dim)
        action_t = self.action_flat(action_t)
        action_t = self.action_decoder(action_t).view(1, self.per_image_dim)
        obs_action_state_cat = torch.cat((xt, state_t, action_t), dim=0)
        
       # xt = xt.view(1,1,1,self.predim)
        zt = self.lr_transformer(obs_action_state_cat)

        
        state_t = self.state_encoder(zt[0])
        # to get rid of state and action "token"
        zt = zt[2:]


        imglat = self.decoder_transformer(zt)
        #pdb.set_trace()

        #pdb.set_trace()
        imglat = imglat.view(batch_dim, 1, self.postCNN_image_n, self.postCNN_image_n)

        #pdb.set_trace()
        observation_t_hat = self.decoder_DCNN(imglat)
        #print(observation_t_hat.size())
        #pdb.set_trace()

        return observation_t_hat, state_t

# A neural operator based world model
class WorldModelNO(nn.Module):
    def __init__(self, image_n, num_frames_per_step, latent_size, state_size, action_dim, cnn_kernel_size=3, no_layers=8):
        super(WorldModelNO, self).__init__()

        self.state_size = state_size
        self.num_frames_per_step = num_frames_per_step
        self.image_n = image_n

        self.encoder_CNN_out_ch = 4

        num_2pools = 3

        self.encoder_CNN = nn.Sequential(
            CNNLayer(1, 8, cnn_kernel_size),
            CNNLayer(8, 64, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(64, 128, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(128, 32, cnn_kernel_size),
             nn.MaxPool2d(2, stride=2),
            CNNLayer(32, self.encoder_CNN_out_ch, cnn_kernel_size),
        )
        self.postCNN_image_n = int(image_n/(2**num_2pools))
        pixles_num = int(self.postCNN_image_n**2)
        self.per_image_dim = self.encoder_CNN_out_ch * pixles_num
        self.predim = self.per_image_dim * num_frames_per_step
        self.action_dim = action_dim
        self.transformer_in_dim = self.predim + state_size + action_dim
        
        #self.state_decoder = nn.Linear(state_size, self.per_image_dim)
        self.state_encoder = nn.Linear(latent_size, state_size)

        self.action_flat = nn.Flatten(start_dim=0)

        self.flat = nn.Flatten()

        # Latent Representation Transformer
        #self.lr_transformer = Transformer(self.per_image_dim, latent_size, heads=8, ff_dim=latent_size*2, out_dim=latent_size, layer_num=8)
        self.image_NO = FNOPredictor(self.predim, latent_size, out_dim=latent_size, no_layers=no_layers)
        self.state_NO = FNOPredictor(state_size, latent_size, out_dim=latent_size, no_layers=no_layers)
        self.action_NO = FNOPredictor(action_dim, latent_size, out_dim=latent_size, no_layers=no_layers)
        self.lr_NO = FNOPredictor(latent_size, latent_size, out_dim=latent_size, no_layers=no_layers)
        
        #self.decoder_transformer = Transformer(latent_size, latent_size, heads=8, ff_dim=latent_size*2, out_dim=pixles_num, layer_num=4)
        self.decoder_NO = FNOPredictor(latent_size, latent_size, out_dim=pixles_num * num_frames_per_step)
        # Uses deconvolutions to generate an image
        # Upscales from 35x35 to 280x280
        self.decoder_DCNN = nn.Sequential(
            DeCNNLayer(1, 32, kernel_size=4, stride=2, padding=1),
            DeCNNLayer(32, 8, kernel_size=4, stride=2, padding=1),
            DeCNNLayer(8, 1, kernel_size=4, stride=2, padding=1),
            #DeCNNLayer(8, 1, kernel_size=4, stride=2, padding=1),
        )
        # self.decoder_DCNN = nn.Sequential(
        #     CNNLayer(1, 32, cnn_kernel_size),
        #     CNNLayer(32, 128, cnn_kernel_size),
        #     CNNLayer(128, 64, cnn_kernel_size),
        #     CNNLayer(64, 8, cnn_kernel_size),
        #     CNNLayer(8, 1, cnn_kernel_size),
        # )


    def forward(self, observation_t, action_t, state_t):
        batch_dim, _, image_n, _ = observation_t.shape
        
        xt = self.encoder_CNN(observation_t)
        
        xt = self.flat(xt)
        
        xt = xt.view(1,1, 1, self.predim)
        
        state_t = (state_t).view(1,1,1,self.state_size)
        action_t = self.action_flat(action_t).view(1,1,1,self.action_dim)
        #obs_action_state_cat = torch.cat((xt, state_t, action_t), dim=3)
        state_t = self.state_NO(state_t)
        action_t = self.action_NO(action_t)
        xt = self.image_NO(xt)

        obs_action_state = xt + state_t + action_t
        #pdb.set_trace()

       # xt = xt.view(1,1,1,self.predim)
        #pdb.set_trace()
        zt = self.lr_NO(obs_action_state)
        state_t = self.state_encoder(zt)
        
        #pdb.set_trace()

        imglat = self.decoder_NO(zt)
        #pdb.set_trace()

        #pdb.set_trace()
        imglat = imglat.view(batch_dim, 1, self.postCNN_image_n, self.postCNN_image_n)
        
        #pdb.set_trace()
        observation_t_hat = self.decoder_DCNN(imglat)
        #print(observation_t_hat.size())
        #pdb.set_trace()

        return observation_t_hat, state_t

# A Mixture of Experts based world model
class WorldModelMoE(nn.Module):
    def __init__(self, image_n, num_frames_per_step, latent_size, state_size, action_size, cnn_kernel_size=3):
        super(WorldModelMoE, self).__init__()

        self.action_size = action_size
        self.num_frames_per_step = num_frames_per_step
        self.image_n = image_n

        self.encoder_CNN_out_ch = 4

        num_2pools = 3

        self.encoder_CNN = nn.Sequential(
            CNNLayer(1, 8, cnn_kernel_size),
            CNNLayer(8, 64, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(64, 128, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(128, 32, cnn_kernel_size),
             nn.MaxPool2d(2, stride=2),
            CNNLayer(32, self.encoder_CNN_out_ch, cnn_kernel_size),
        )
        self.postCNN_image_n = int(image_n/(2**num_2pools))
        pixles_num = int(self.postCNN_image_n**2)
        self.per_image_dim = self.encoder_CNN_out_ch * pixles_num
        self.predim = self.per_image_dim * num_frames_per_step
        self.transformer_in_dim = self.predim + state_size
        
        self.state_decoder = nn.Linear(state_size, self.per_image_dim)
        self.state_encoder = nn.Linear(latent_size, state_size)

        self.action_decoder = nn.Linear(action_size, self.per_image_dim)
        self.action_flat = nn.Flatten(start_dim=0)

        self.flat = nn.Flatten()

        # Latent Representation Transformer
        #self.lr_transformer = Transformer(self.per_image_dim, latent_size, heads=8, ff_dim=latent_size*2, out_dim=latent_size, layer_num=8)
        self.lr_MoE = MoEPredictor(self.per_image_dim, latent_size, latent_size, num_experts=8, num_layers=8)

        #self.decoder_transformer = Transformer(latent_size, latent_size, heads=8, ff_dim=latent_size*2, out_dim=pixles_num, layer_num=4)
        self.decoder_MoE = MoEPredictor(latent_size, pixles_num, latent_size, num_experts=8, num_layers=4)
        # Uses deconvolutions to generate an image
        # Upscales from 35x35 to 280x280
        self.decoder_DCNN = nn.Sequential(
            DeCNNLayer(1, 32, kernel_size=4, stride=2, padding=1),
            DeCNNLayer(32, 8, kernel_size=4, stride=2, padding=1),
            DeCNNLayer(8, 1, kernel_size=4, stride=2, padding=1),
            #DeCNNLayer(8, 1, kernel_size=4, stride=2, padding=1),
        )
        # self.decoder_DCNN = nn.Sequential(
        #     CNNLayer(1, 32, cnn_kernel_size),
        #     CNNLayer(32, 128, cnn_kernel_size),
        #     CNNLayer(128, 64, cnn_kernel_size),
        #     CNNLayer(64, 8, cnn_kernel_size),
        #     CNNLayer(8, 1, cnn_kernel_size),
        # )


    def forward(self, observation_t, action_t, state_t):
        batch_dim, _, image_n, _ = observation_t.shape

        
        
        xt = self.encoder_CNN(observation_t)
        
        xt = self.flat(xt)
        
        #xt = xt.view(1,1, batch_dim, self.per_image_dim)
        
        state_t = self.state_decoder(state_t).view(1, self.per_image_dim)
        action_t = self.action_flat(action_t)
        action_t = self.action_decoder(action_t).view(1, self.per_image_dim)
        obs_state_cat = torch.cat((xt, state_t, action_t), dim=0)
        #obs_state_cat=obs_state_cat.view(1, batch_dim + 1, self.per_image_dim, 1)
        
       # xt = xt.view(1,1,1,self.predim)
        zt = self.lr_MoE(obs_state_cat)

        
        state_t = self.state_encoder(zt[0])
        zt = zt[2:]


        imglat = self.decoder_MoE(zt)
        #pdb.set_trace()

        #pdb.set_trace()
        imglat = imglat.view(batch_dim, 1, self.postCNN_image_n, self.postCNN_image_n)

        #pdb.set_trace()
        observation_t_hat = self.decoder_DCNN(imglat)
        #print(observation_t_hat.size())
        #pdb.set_trace()

        return observation_t_hat, state_t

# A Mixture of Experts and neural operator based world model
class WorldModelMoENO(nn.Module):
    def __init__(self, image_n, num_frames_per_step, latent_size, state_size, action_size, cnn_kernel_size=3):
        super(WorldModelMoENO, self).__init__()

        self.action_size = action_size
        self.num_frames_per_step = num_frames_per_step
        self.image_n = image_n

        self.encoder_CNN_out_ch = 4

        num_2pools = 3

        self.encoder_CNN = nn.Sequential(
            CNNLayer(1, 8, cnn_kernel_size),
            CNNLayer(8, 64, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(64, 128, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(128, 32, cnn_kernel_size),
             nn.MaxPool2d(2, stride=2),
            CNNLayer(32, self.encoder_CNN_out_ch, cnn_kernel_size),
        )
        self.postCNN_image_n = int(image_n/(2**num_2pools))
        pixles_num = int(self.postCNN_image_n**2)
        self.per_image_dim = self.encoder_CNN_out_ch * pixles_num
        self.predim = self.per_image_dim * num_frames_per_step
        self.transformer_in_dim = self.predim + state_size
        
        self.state_decoder = nn.Linear(state_size, self.per_image_dim)
        self.state_encoder = nn.Linear(latent_size, state_size)

        self.action_decoder = nn.Linear(action_size, self.per_image_dim)
        self.action_flat = nn.Flatten(start_dim=0)

        self.flat = nn.Flatten()

        # Latent Representation Transformer
        #self.lr_transformer = Transformer(self.per_image_dim, latent_size, heads=8, ff_dim=latent_size*2, out_dim=latent_size, layer_num=8)
        self.lr_MoE = MoEPredictor(self.per_image_dim, latent_size, latent_size, num_experts=8, num_layers=8)

        #self.decoder_transformer = Transformer(latent_size, latent_size, heads=8, ff_dim=latent_size*2, out_dim=pixles_num, layer_num=4)
        self.decoder_MoE = MoEPredictor(latent_size, pixles_num, latent_size, num_experts=8, num_layers=4)
        # Uses deconvolutions to generate an image
        # Upscales from 35x35 to 280x280
        self.decoder_DCNN = nn.Sequential(
            DeCNNLayer(1, 32, kernel_size=4, stride=2, padding=1),
            DeCNNLayer(32, 8, kernel_size=4, stride=2, padding=1),
            DeCNNLayer(8, 1, kernel_size=4, stride=2, padding=1),
            #DeCNNLayer(8, 1, kernel_size=4, stride=2, padding=1),
        )
        # self.decoder_DCNN = nn.Sequential(
        #     CNNLayer(1, 32, cnn_kernel_size),
        #     CNNLayer(32, 128, cnn_kernel_size),
        #     CNNLayer(128, 64, cnn_kernel_size),
        #     CNNLayer(64, 8, cnn_kernel_size),
        #     CNNLayer(8, 1, cnn_kernel_size),
        # )
        self.no_flat = nn.Flatten()
        self.no = FNOPredictor(10*latent_size, latent_size, 10*latent_size, no_layers=16)
        self.latent_size = latent_size

    def forward(self, observation_t, action_t, state_t):
        batch_dim, _, image_n, _ = observation_t.shape

        
        
        xt = self.encoder_CNN(observation_t)
        
        xt = self.flat(xt)
        # xt = xt.view(1, xt.size()[0])
        
        #xt = xt.view(1,1, batch_dim, self.per_image_dim)
        
        state_t = self.state_decoder(state_t).view(1, self.per_image_dim)
        action_t = self.action_flat(action_t)
        action_t = self.action_decoder(action_t).view(1, self.per_image_dim)
        #pdb.set_trace()
        obs_state_cat = torch.cat((xt, state_t, action_t), dim=0)
        #obs_state_cat=obs_state_cat.view(1, batch_dim + 1, self.per_image_dim, 1)
        
       # xt = xt.view(1,1,1,self.predim)
        zt = self.lr_MoE(obs_state_cat)


        shape = zt.size()
        zt = self.no_flat(zt)
        zt = zt.view(1,1,1,10 * self.latent_size)

        zt = self.no(zt)

        zt = zt.view(10, self.latent_size)
        
        state_t = self.state_encoder(zt[0])
        zt = zt[2:]


        imglat = self.decoder_MoE(zt)
        #pdb.set_trace()

        #pdb.set_trace()
        imglat = imglat.view(batch_dim, 1, self.postCNN_image_n, self.postCNN_image_n)

        #pdb.set_trace()
        observation_t_hat = self.decoder_DCNN(imglat)
        #print(observation_t_hat.size())
        #pdb.set_trace()

        return observation_t_hat, state_t

# A neural operator based world model that is internally recurrent
class WorldModelNOR(nn.Module):
    def __init__(self, image_n, num_frames_per_step, latent_size, state_size, action_dim, cnn_kernel_size=3, no_layers=8):
        super(WorldModelNOR, self).__init__()

        self.state_size = state_size
        self.num_frames_per_step = num_frames_per_step
        self.image_n = image_n

        self.encoder_CNN_out_ch = 4

        num_2pools = 3

        self.encoder_CNN = nn.Sequential(
            CNNLayer(1, 8, cnn_kernel_size),
            CNNLayer(8, 64, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(64, 128, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(128, 32, cnn_kernel_size),
             nn.MaxPool2d(2, stride=2),
            CNNLayer(32, self.encoder_CNN_out_ch, cnn_kernel_size),
        )
        self.postCNN_image_n = int(image_n/(2**num_2pools))
        pixles_num = int(self.postCNN_image_n**2)
        self.per_image_dim = self.encoder_CNN_out_ch * pixles_num
        self.predim = self.per_image_dim * num_frames_per_step
        self.action_dim = action_dim
        self.transformer_in_dim = self.predim + state_size + action_dim
        
        #self.state_decoder = nn.Linear(state_size, self.per_image_dim)
        self.state_encoder = nn.Linear(latent_size, state_size)

        self.action_flat = nn.Flatten(start_dim=0)

        self.flat = nn.Flatten()

        # Latent Representation Transformer
        #self.lr_transformer = Transformer(self.per_image_dim, latent_size, heads=8, ff_dim=latent_size*2, out_dim=latent_size, layer_num=8)
        self.image_mlp = MLP(4, self.predim, latent_size, latent_size)#FNOPredictor(self.predim, latent_size, out_dim=latent_size, no_layers=no_layers)
        self.state_mlp = MLP(4, state_size, latent_size, latent_size)#FNOPredictor(state_size, latent_size, out_dim=latent_size, no_layers=no_layers)
        self.action_mlp = MLP(4, action_dim, latent_size, latent_size) #FNOPredictor(action_dim, latent_size, out_dim=latent_size, no_layers=no_layers)
        self.lr_mlp = MLP(4, latent_size, latent_size, latent_size)#FNOPredictor(latent_size, latent_size, out_dim=latent_size, no_layers=2)

        self.recurrent_predictor = nn.Sequential(
            MLP(2, latent_size, latent_size, latent_size),
            FNOPredictor(latent_size, latent_size, out_dim=latent_size, no_layers=no_layers),
            MLP(2, latent_size, latent_size, latent_size),
        )
        
        #self.decoder_transformer = Transformer(latent_size, latent_size, heads=8, ff_dim=latent_size*2, out_dim=pixles_num, layer_num=4)
        # self.decoder_NO = FNOPredictor(latent_size, latent_size, out_dim=pixles_num)
        # Uses deconvolutions to generate an image
        # Upscales from 35x35 to 280x280
        self.preimage_mlp = MLP(1, latent_size, latent_size, pixles_num)
        self.decoder_DCNN = nn.Sequential(
            DeCNNLayer(1, 32, kernel_size=4, stride=2, padding=1),
            DeCNNLayer(32, 8, kernel_size=4, stride=2, padding=1),
            DeCNNLayer(8, 1, kernel_size=4, stride=2, padding=1),
            #DeCNNLayer(8, 1, kernel_size=4, stride=2, padding=1),
        )
        # self.decoder_DCNN = nn.Sequential(
        #     CNNLayer(1, 32, cnn_kernel_size),
        #     CNNLayer(32, 128, cnn_kernel_size),
        #     CNNLayer(128, 64, cnn_kernel_size),
        #     CNNLayer(64, 8, cnn_kernel_size),
        #     CNNLayer(8, 1, cnn_kernel_size),
        # )


    def forward(self, observation_t, action_t, state_t):
        batch_dim, _, image_n, _ = observation_t.shape
        
        xt = self.encoder_CNN(observation_t)
        
        xt = self.flat(xt)
        
        xt = xt.view(1,1, 1, self.predim)
        
        state_t = (state_t).view(1,1,1,self.state_size)
        action_t = self.action_flat(action_t).view(1,1,1,self.action_dim)
        #obs_action_state_cat = torch.cat((xt, state_t, action_t), dim=3)
        state_t = self.state_mlp(state_t)
        action_t = self.action_mlp(action_t)
        xt = self.image_mlp(xt)

        obs_action_state = xt + state_t + action_t
        #pdb.set_trace()

       # xt = xt.view(1,1,1,self.predim)
        #pdb.set_trace()
        zt = self.lr_mlp(obs_action_state)


        pred_zt = []
        for i in range(self.num_frames_per_step):
            zt = self.recurrent_predictor(zt)
            pred_zt.append(zt)

        pred_zt = torch.cat(pred_zt, dim=0)

        
        #pdb.set_trace()

        

        #pdb.set_trace()
        
        
        #pdb.set_trace()
        #print(observation_t_hat.size())
        #pdb.set_trace()

        imglat = self.preimage_mlp(pred_zt)
        #pdb.set_trace()
        imglat = imglat.view(self.num_frames_per_step, 1, self.postCNN_image_n, self.postCNN_image_n)
        pred_observations = self.decoder_DCNN(imglat)

        state_tp1 = self.state_encoder(zt)
        return pred_observations, state_tp1

class WorldModelMamba(nn.Module):
    def __init__(self, image_n, num_frames_per_step, latent_size, state_size, action_size, cnn_kernel_size=3):
        super(WorldModelMamba, self).__init__()

        self.loss = nn.MSELoss()
        self.action_size = action_size
        self.num_frames_per_step = num_frames_per_step
        self.image_n = image_n
        self.latent_size = latent_size
        self.state_size = state_size

        self.encoder_CNN_out_ch = 4

        num_2pools = 3

        self.encoder_CNN = nn.Sequential(
            CNNLayer(1, 8, cnn_kernel_size),
            CNNLayer(8, 64, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(64, 128, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(128, 32, cnn_kernel_size),
             nn.MaxPool2d(2, stride=2),
            CNNLayer(32, self.encoder_CNN_out_ch, cnn_kernel_size),
        )
        self.postCNN_image_n = int(image_n/(2**num_2pools))
        pixles_num = int(self.postCNN_image_n**2)
        self.per_image_dim = self.encoder_CNN_out_ch * pixles_num
        self.predim = self.per_image_dim * num_frames_per_step
        self.transformer_in_dim = self.predim + state_size + action_size
        
        self.state_encoder = nn.Linear(latent_size, state_size)

        self.action_flat = nn.Flatten(start_dim=0)

        self.flat = nn.Flatten()

        # Latent Representation Transformer
        #self.lr_transformer = Transformer(self.per_image_dim, latent_size, heads=8, ff_dim=latent_size*2, out_dim=latent_size, layer_num=8)
        self.lr_mlp = MLP(4, self.transformer_in_dim, latent_size, self.num_frames_per_step*latent_size)#MoEPredictor(self.per_image_dim, latent_size, latent_size, num_experts=8, num_layers=8)

        self.mamba = nn.Sequential(
            Mamba2(latent_size),
            Mamba2(latent_size),
            Mamba2(latent_size),
            Mamba2(latent_size),
        )

        #self.decoder_transformer = Transformer(latent_size, latent_size, heads=8, ff_dim=latent_size*2, out_dim=pixles_num, layer_num=4)
        self.decoder_mlp = MLP(2, latent_size, latent_size, pixles_num)#MoEPredictor(latent_size, pixles_num, latent_size, num_experts=8, num_layers=4)

        # Uses deconvolutions to generate an image
        # Upscales from 35x35 to 280x280
        self.decoder_DCNN = nn.Sequential(
            DeCNNLayer(1, 32, kernel_size=4, stride=2, padding=1),
            DeCNNLayer(32, 8, kernel_size=4, stride=2, padding=1),
            DeCNNLayer(8, 1, kernel_size=4, stride=2, padding=1),
            #DeCNNLayer(8, 1, kernel_size=4, stride=2, padding=1),
        )
        # self.decoder_DCNN = nn.Sequential(
        #     CNNLayer(1, 32, cnn_kernel_size),
        #     CNNLayer(32, 128, cnn_kernel_size),
        #     CNNLayer(128, 64, cnn_kernel_size),
        #     CNNLayer(64, 8, cnn_kernel_size),
        #     CNNLayer(8, 1, cnn_kernel_size),
        # )


    def forward(self, observation_t, action_t, state_t):
        
        xt = self.encoder_CNN(observation_t)
        xt = self.flat(xt)
        
        xt = xt.view(1,1, 1, self.predim)
        state_t = state_t.view(1, 1, 1, self.state_size)
        action_t = self.action_flat(action_t).view(1, 1, 1, self.action_size)

        obs_state_cat = torch.cat((xt, state_t, action_t), dim=3)
        #obs_state_cat=obs_state_cat.view(1, batch_dim + 1, self.per_image_dim, 1)
        
       # xt = xt.view(1,1,1,self.predim)
        zt = self.lr_mlp(obs_state_cat)
        zt = zt.view(1,self.num_frames_per_step, self.latent_size)

        zt = self.mamba(zt)
        
        state_t = self.state_encoder(zt[:,-1])


        zt = torch.transpose(zt, 0, 1)
        imglat = self.decoder_mlp(zt)
        #pdb.set_trace()

        #pdb.set_trace()
        imglat = imglat.view(self.num_frames_per_step, 1, self.postCNN_image_n, self.postCNN_image_n)
        observation_t_hat = self.decoder_DCNN(imglat)
        #print(observation_t_hat.size())
        #pdb.set_trace()

        return observation_t_hat, state_t