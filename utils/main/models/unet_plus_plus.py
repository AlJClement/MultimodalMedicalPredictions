import torch

import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp
torch.cuda.empty_cache() 

# @misc{Iakubovskii:2019,
#   Author = {Pavel Iakubovskii},
#   Title = {Segmentation Models Pytorch},
#   Year = {2019},
#   Publisher = {GitHub},
#   Journal = {GitHub repository},
#   Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
# }

class unet_plus_plus(nn.Module):
    def __init__(self, cfg):
        super(unet_plus_plus, self).__init__()
        self.unet = smp.UnetPlusPlus(
            encoder_name=cfg.MODEL.ENCODER_NAME,
            encoder_weights=cfg.MODEL.ENCODER_WEIGHTS,
            decoder_channels=cfg.MODEL.DECODER_CHANNELS,
            decoder_use_batchnorm=cfg.MODEL.BATCH_NORM_DECODER,
            in_channels=cfg.MODEL.IN_CHANNELS,
            classes=cfg.MODEL.OUT_CHANNELS,
        )
    def two_d_softmax(self,x):
        exp_y = torch.exp(x)
        return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)
    
    def forward(self, x, meta):
        return self.two_d_softmax(self.unet(x))
    

