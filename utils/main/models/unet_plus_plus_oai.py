import torch

import numpy as np
import torch.nn as nn
torch.cuda.empty_cache() 
import torch
import monai
from monai.networks.nets import UNetPlusPlus

class AttentionUNetPlusPlus(UNetPlusPlus):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_blocks = torch.nn.ModuleList([
            monai.networks.blocks.AttentionBlock(spatial_dims=2, 
                                                 in_channels=cfg.MODEL.IN_CHANNELS, 
                                                 gating_channels=c, 
                                                 inter_channels=c // 2)
            for c in self.channels[:-1]  # Skip connections need attention
        ])

    def two_d_softmax(self,x):
        exp_y = torch.exp(x)
        return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)

    def forward(self, x):
        enc_features = self.encoder(x)
        dec_features = []
        
        # Apply Attention before feeding into decoder
        for i, attn in enumerate(self.attention_blocks):
            enc_features[i] = attn(enc_features[i], enc_features[-1])
        
        for idx in range(len(self.decoder)):
            dec_features.append(self.decoder[idx](enc_features[-(idx + 2)], dec_features[-1] if dec_features else None))
        
        return self.two_d_softmax(self.final(dec_features[-1]))
    

class unet_plus_plus_monai(nn.Module):
    def __init__(self, cfg):
        super(unet_plus_plus_monai, self).__init__()
        # self.unet = smp.UnetPlusPlus(
        #     encoder_name=cfg.MODEL.ENCODER_NAME,
        #     encoder_weights=cfg.MODEL.ENCODER_WEIGHTS,
        #     decoder_channels=cfg.MODEL.DECODER_CHANNELS,
        #     decoder_use_batchnorm=cfg.MODEL.BATCH_NORM_DECODER,
        #     decoder_attention_type=cfg.MODEL.ATTENTION,
        #     in_channels=cfg.MODEL.IN_CHANNELS,
        #     classes=cfg.MODEL.OUT_CHANNELS,
        # )

        self.unet = AttentionUNetPlusPlus(
            spatial_dims=2,  # Use 3 for 3D images
            in_channels=1,  # Number of input channels
            out_channels=2,  # Number of segmentation classes
            channels=(16, 32, 64, 128, 256),  # Encoder channels
            strides=(2, 2, 2, 2),  # Downsampling factors
            num_res_units=2,  # Residual units
        ).to("cuda")  # Move to GPU
                    

    
    def forward(self, x, meta):
        return self.two_d_softmax(self.unet(x))
    

