import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from LFNeuroControl.models.neuraloperator import FNOPredictor
from LFNeuroControl.models.transcnn import TransCNN
from LFNeuroControl.models.cnn import CNNLayer, DeCNNLayer
from LFNeuroControl.models.transformer import Transformer
from LFNeuroControl.models.encoder import SpikePositionEncoding
import pdb
import csv
# A Recurrent World Model 
class WorldModelT(nn.Module):
    def __init__(self, image_n, num_frames_per_step, latent_size, state_size, cnn_kernel_size=3):
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
        self.transformer_in_dim = self.predim + state_size
        
        self.state_decoder = nn.Linear(state_size, self.per_image_dim)
        self.state_encoder = nn.Linear(latent_size, state_size)

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


    def forward(self, observation_t, state_t):
        batch_dim, _, image_n, _ = observation_t.shape
        
        xt = self.encoder_CNN(observation_t)
        
        xt = self.flat(xt)
        
        #xt = xt.view(1,1, batch_dim, self.per_image_dim)
        
        state_t = self.state_decoder(state_t).view(1, self.per_image_dim)
        obs_state_cat = torch.cat((xt, state_t), dim=0)
        
       # xt = xt.view(1,1,1,self.predim)
        zt = self.lr_transformer(obs_state_cat)

        
        state_t = self.state_encoder(zt[0])
        zt = zt[1:]


        imglat = self.decoder_transformer(zt)
        #pdb.set_trace()

        #pdb.set_trace()
        imglat = imglat.view(batch_dim, 1, self.postCNN_image_n, self.postCNN_image_n)

        #pdb.set_trace()
        observation_t_hat = self.decoder_DCNN(imglat)
        #print(observation_t_hat.size())
        #pdb.set_trace()

        return observation_t_hat, state_t


class WorldModelNO(nn.Module):
    def __init__(self, image_n, num_frames_per_step, latent_size, state_size, cnn_kernel_size=3):
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
        self.transformer_in_dim = self.predim + state_size
        
        #self.state_decoder = nn.Linear(state_size, self.per_image_dim)
        self.state_encoder = nn.Linear(latent_size, state_size)

        self.flat = nn.Flatten()

        # Latent Representation Transformer
        #self.lr_transformer = Transformer(self.per_image_dim, latent_size, heads=8, ff_dim=latent_size*2, out_dim=latent_size, layer_num=8)
        self.lr_NO = FNOPredictor(self.transformer_in_dim, latent_size, out_dim=latent_size)
        
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


    def forward(self, observation_t, state_t):
        batch_dim, _, image_n, _ = observation_t.shape
        
        xt = self.encoder_CNN(observation_t)
        
        xt = self.flat(xt)
        
        xt = xt.view(1,1, 1, self.predim)
        
        state_t = (state_t).view(1,1,1,self.state_size)
        obs_state_cat = torch.cat((xt, state_t), dim=3)
        
       # xt = xt.view(1,1,1,self.predim)
        #pdb.set_trace()
        zt = self.lr_NO(obs_state_cat)
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
