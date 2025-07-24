import torch
import numpy as np
import torch.nn as nn
from yacs.config import CfgNode as CN
import sys
import os
import importlib

hrnet_lib_path = '/home/scratch/allent/HigherHRNet-Human-Pose-Estimation/lib'
pose_hrnet_path = os.path.join(hrnet_lib_path, 'models', 'pose_higher_hrnet.py')

spec = importlib.util.spec_from_file_location("hrnet_pose_hrnet", pose_hrnet_path)
hrnet_pose_hrnet = importlib.util.module_from_spec(spec)
sys.modules["hrnet_pose_hrnet"] = hrnet_pose_hrnet
spec.loader.exec_module(hrnet_pose_hrnet)
import torch.nn.functional as F

from hrnet_pose_hrnet import PoseHigherResolutionNet, get_pose_net, BasicBlock
torch.cuda.empty_cache() 
#https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
#weights https://drive.google.com/drive/folders/1NYOZm3oGnuP0KlmQlIeP5sQvlHkRzu3l

class hrnet(nn.Module):
    def __init__(self, cfg):
        super(hrnet, self).__init__()
        self.num_landmarks = cfg.MODEL.OUT_CHANNELS
        self.preload_weigths_path = '/home/scratch/allent/MultimodalMedicalPredictions/utils/main/models/weights/hrnet_w32-36af842e.pth'

        self.hrnet_cfg=self.get_hrnet_w32_config()

        self.backbone = self.get_hrnet_backbone(pretrained=True)
        self.backbone.eval()
        self.config = cfg

        self.get_pose_net = hrnet_pose_hrnet.HighResolutionModule(
                                num_branches=4,  # example number, replace with your config
                                blocks=BasicBlock,  # or Bottleneck, depends on your model
                                num_blocks=[4,4,4,4],  # list of blocks per branch
                                num_inchannels=[64,128,256,512],  # input channels per branch
                                num_channels=[64,128,256,512],  # output channels per branch
                                fuse_method='SUM' #OR'concat but then need to change accross config
                            )
        
        #480 because of current w32 from four brnaches at these resolutins 32,64,128,256
        self.head = nn.Conv2d(480, self.num_landmarks, kernel_size=1, bias=True)
    
    def get_hrnet_w32_config(self):

        cfg = CN()

        cfg.MODEL = CN()
        cfg.MODEL.EXTRA = CN()

        cfg.MODEL.INIT_WEIGHTS = True
    
        cfg.MODEL.PRETRAINED = self.preload_weigths_path 

        cfg.MODEL.EXTRA.STAGE1 = CN()
        cfg.MODEL.EXTRA.STAGE1.NUM_MODULES = 1
        cfg.MODEL.EXTRA.STAGE1.NUM_BRANCHES = 1
        cfg.MODEL.EXTRA.STAGE1.BLOCK = 'BOTTLENECK'
        cfg.MODEL.EXTRA.STAGE1.NUM_BLOCKS = [4]
        cfg.MODEL.EXTRA.STAGE1.NUM_CHANNELS = [64]
        cfg.MODEL.EXTRA.STAGE1.FUSE_METHOD = 'SUM'

        cfg.MODEL.EXTRA.STAGE2 = CN()
        cfg.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
        cfg.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
        cfg.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
        cfg.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
        cfg.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [32, 64]
        cfg.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

        cfg.MODEL.EXTRA.STAGE3 = CN()
        cfg.MODEL.EXTRA.STAGE3.NUM_MODULES = 4
        cfg.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
        cfg.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
        cfg.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
        cfg.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [32, 64, 128]
        cfg.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

        cfg.MODEL.EXTRA.STAGE4 = CN()
        cfg.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
        cfg.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
        cfg.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
        cfg.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
        cfg.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
        cfg.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'

        cfg.MODEL.EXTRA.FINAL_CONV_KERNEL = 1

        cfg.MODEL.TAG_PER_JOINT = False  # or True, depending on loaded model
        #multi-person pose estimation methods like HigherHRNet or Associative Embedding.
        #estimate is false because this is all one joint
        cfg.MODEL.OUT_CHANNELS = self.num_landmarks 
        cfg.MODEL.NUM_JOINTS = 7
        cfg.MODEL.INPUT_CHANNELS = 1 

        cfg.MODEL.EXTRA.DECONV = CN()
        cfg.MODEL.EXTRA.DECONV.NUM_DECONVS = 1
        cfg.MODEL.EXTRA.DECONV.NUM_CHANNELS = [32]
        cfg.MODEL.EXTRA.DECONV.NUM_LAYERS = 1
        cfg.MODEL.EXTRA.DECONV.NUM_FILTERS = [256, 256, 256]
        cfg.MODEL.EXTRA.DECONV.KERNEL_SIZE = [4, 4, 4]

        cfg.LOSS = CN()
        cfg.LOSS.TYPE = 'MSELoss'
        cfg.LOSS.WITH_AE_LOSS = [False,False]
        # cfg.LOSS.HEATMAP_LOSS_TYPE = 'MSELoss'
        # cfg.LOSS.TAG_LOSS_TYPE = 'AssociativeEmbeddingLoss' # depends on your framework
        cfg.INPUT = CN()
        cfg.INPUT.IMAGE_SIZE = [512, 512]  # width, height
        cfg.MODEL.EXTRA.DECONV.CAT_OUTPUT = [True, True, True]
        cfg.MODEL.EXTRA.DECONV.NUM_BASIC_BLOCKS = 4
        cfg.MODEL.EXTRA.PRETRAINED_LAYERS = ['*']
        return cfg
    
    def get_hrnet_backbone(self, pretrained=True):
        # Load config, e.g., hrnet_w32 config
        # Build HRNet backbone
        model = get_pose_net(self.hrnet_cfg, is_train=True)
        if pretrained:
            # Load pretrained weights
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained = torch.load(self.preload_weigths_path, map_location=device)
            model.load_state_dict(pretrained, strict=False)
            model.to(device)
            print("Loaded pretrained ImageNet weights")

        return model

    def two_d_softmax(self, x):
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x, dim=(2,3), keepdim=True)

    def forward(self, x, meta=None):
        features = self.backbone(x)

        # Upsample
        f0_up = F.interpolate(features[0], size=(512, 512), mode='bilinear', align_corners=False)
        f1_up = F.interpolate(features[1], size=(512, 512), mode='bilinear', align_corners=False)

        # Now you can concat or sum:
        # combined = torch.cat([f0_up, features[1]], dim=1)  # along channels
        # # or
        heatmaps = f0_up + f1_up # element-wise sum, if channels match
        heatmaps = self.two_d_softmax(heatmaps)
        return heatmaps