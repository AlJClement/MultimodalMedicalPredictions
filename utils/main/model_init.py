import torch    
from .models import *
from preprocessing.metadata_import import MetadataImport
from torchsummary import summary
import numpy as np
class model_init():
    def __init__(self,cfg) -> None:
        self.net = eval(cfg.MODEL.NAME)
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_channels=cfg.MODEL.IN_CHANNELS
        self.out_channels=cfg.MODEL.OUT_CHANNELS
        self.init_feats=cfg.MODEL.INIT_FEATURES
        self.bs = cfg.TRAIN.BATCH_SIZE
        self.im_size = cfg.DATASET.CACHED_IMAGE_SIZE

        self.meta_features=cfg.MODEL.META_FEATURES
        self.num_meta_features= sum(cfg.MODEL.META_FEATURES)

        #get from met        self.num_meta_features = ls
        self.meta_func = eval("MetadataImport(cfg)." + cfg.MODEL.NAME)

        return
    def get_net_info(self,net):
        '''add the info to logger'''
        input_im_size = torch.tensor([self.in_channels,self.im_size[0],self.im_size[1]])
        input_meta_shape = torch.tensor([self.in_channels,len(self.meta_features),int(np.average(self.meta_features))])
        net_summary= summary(net,[tuple(input_im_size.detach().numpy()), tuple(input_meta_shape.detach().numpy())])
        return  net_summary
            
    def get_net_from_conf(self, get_net_info=True):
        net = self.net(self.bs, self.im_size, self.num_meta_features, out_channels = self.out_channels)
        net = net.to(self.device)
        if get_net_info == True:
            return net, self.get_net_info(net)
        else:
            return net
    
    def get_modelspecific_feature_structure(self):
        return self.meta_func
    
    