import torch

import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp
from collections import OrderedDict
torch.cuda.empty_cache() 

# @misc{Iakubovskii:2019,OrderedDict
class unet_plus_plus_meta(nn.Module):
    def __init__(self, cfg):
        super(unet_plus_plus_meta, self).__init__()

        self.in_channels=cfg.MODEL.IN_CHANNELS
        self.out_channels=cfg.MODEL.OUT_CHANNELS
        self.init_features=cfg.MODEL.INIT_FEATURES
        self.bs = cfg.TRAIN.BATCH_SIZE
        self.im_size = cfg.DATASET.CACHED_IMAGE_SIZE

        self.meta_features=cfg.MODEL.META_FEATURES
        self.num_meta_features= sum(cfg.MODEL.META_FEATURES)
                
        self.stride = 2
        self.pool = 2
        self.features = int(self.init_features)
        self.features=self.init_features
        self.im_size_h = self.im_size[0]
        self.im_size_w = self.im_size[1]

        self.device = cfg.MODEL.DEVICE

        self.unet = smp.UnetPlusPlus(
            encoder_name=cfg.MODEL.ENCODER_NAME,
            encoder_weights=cfg.MODEL.ENCODER_WEIGHTS,
            decoder_channels=cfg.MODEL.DECODER_CHANNELS,
            decoder_use_batchnorm=cfg.MODEL.BATCH_NORM_DECODER,
            in_channels=cfg.MODEL.IN_CHANNELS,
            classes=cfg.MODEL.OUT_CHANNELS,
        )
        self.lin_end = self._lin_end()

    def _lin_end(self):
        #print("pred; ",xx.shape) torch.Size([1, 5, 512, 512]) 
        feat = (self.bs, self.out_channels, self.im_size_h,self.im_size_w)
        #print(feat)
        feats = self.bs*self.im_size_h*self.im_size_w
        
        #xx is input from previous model so ,bs, img size w, im size h
        xx = torch.tensor([self.out_channels, self.bs*self.im_size_h*self.im_size_w])
        xx=xx.view(-1,feat[1])
        #print('xx before meta: ',xx.size()) torch.Size([262144, 5])
        
        #print(meta.shape) torch.Size([1, 1, 3, 100]) 
        meta_shape = self.meta_features.shape[1]
        
        meta_flat=self.meta_features.view(-1,meta_shape)
        meta_flat=meta_flat.repeat(1,self.out_channels)
        
        xx=torch.cat((xx,meta_flat),dim=0)
        #print(xx.shape)torch.Size([262444, 5]) 
        #flatten to linear layer
        xx=torch.flatten(xx.view(xx.size(1), -1))
        #print('xx:', xx.shape)torch.Size([1312220])
        
        in_features_lin = list(xx.shape)[0] #feats + self.num_meta_feats*self.bs
        _out_features_lin = self.num_meta_features*self.bs #list(xx.shape)[0] #
        out_features_lin = feats * self.out_channels
        name = "meta"
        lin = nn.Sequential(OrderedDict(
                [(name+'_1_linear', nn.Linear(in_features=in_features_lin, out_features=_out_features_lin)),
                 (name+'_2_linear', nn.Linear(in_features=_out_features_lin, out_features=out_features_lin)),
                ]))
        #print('in_features_lin:', in_features_lin)
        #print('_out_feats:', _out_features_lin)
        #print('out_feats:', out_features_lin)
        
        _lin=lin.to(self.device)
        x_lin=_lin(xx)
        #print('out:',x_lin.shape)
    
        un_flat=nn.Squential(nn.Unflatten(0, (bs,self.out_channels,self.im_size_h,self.im_size_w)))

        x=un_flat(x_lin)

        return x/home/scratch/allent/MultimodalMedicalPredictions/utils/main/models/unet_plus_plus_meta.py

    def two_d_softmax(self, x):
        exp_y = torch.exp(x)
        return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)
    
    def forward(self, im, meta):
        xx = self.unet(im)
        print(xx.shape)
        x = self.lin_end(xx, meta)
        return self.two_d_softmax(x)
    

