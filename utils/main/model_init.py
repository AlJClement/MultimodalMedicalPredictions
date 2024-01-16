import torch    
from .models import *
from preprocessing.metadata_import import MetadataImport
from torchsummary import summary
from torchinfo import summary
import numpy as np
class model_init():
    def __init__(self,cfg) -> None:
        self.cfg=cfg
        self.net = eval(cfg.MODEL.NAME)
        #self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_channels=cfg.MODEL.IN_CHANNELS
        self.out_channels=cfg.MODEL.OUT_CHANNELS
        self.init_feats=cfg.MODEL.INIT_FEATURES
        self.bs = cfg.TRAIN.BATCH_SIZE
        self.im_size = cfg.DATASET.CACHED_IMAGE_SIZE

        self.meta_features=cfg.MODEL.META_FEATURES
        self.num_meta_features= sum(cfg.MODEL.META_FEATURES)

        #
        self.meta_func = eval("MetadataImport(cfg)." + cfg.MODEL.NAME)
        self.device = cfg.MODEL.DEVICE

        return
    
    def get_net_parameters(self, net):
        params = [p.numel() for p in net.parameters()]
        params_total = sum(params)
        print('Trainable params: ', params_total)
        return
    
    def get_net_info(self,net):
        '''add the info to logger'''
        input_im_size = torch.tensor([self.bs, self.in_channels,self.im_size[0],self.im_size[1]])
        input_meta_shape = torch.tensor([self.bs,self.in_channels,len(self.meta_features),int(np.average(self.meta_features))])
        
        #no bs for torch summary
        #net_summary_torchsummary= summary(net,[tuple(input_im_size.detach().numpy()), tuple(input_meta_shape.detach().numpy())])
        net_summary= summary(net,[tuple(input_im_size.detach().numpy()), tuple(input_meta_shape.detach().numpy())])

        return  net_summary
            
    def get_net_from_conf(self, get_net_info=True):
        net = self.net(self.cfg)
        net = net.to(self.device)
        net.train()
        if get_net_info == True:
            return net, self.get_net_info(net)
        else:
            return net
    
    def get_modelspecific_feature_structure(self):
        return self.meta_func
    
    