import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class dpt(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.in_channels = cfg.MODEL.IN_CHANNELS
        self.out_channels = cfg.MODEL.OUT_CHANNELS
        self.im_size = cfg.DATASET.CACHED_IMAGE_SIZE
        self.device = cfg.MODEL.DEVICE

        self.dpt = smp.DPT(
            encoder_name=cfg.MODEL.ENCODER_NAME,
            encoder_weights=cfg.MODEL.ENCODER_WEIGHTS,
            in_channels=cfg.MODEL.IN_CHANNELS,
            classes=cfg.MODEL.OUT_CHANNELS,
            activation=None,
        )

        self.bn = nn.BatchNorm2d(self.out_channels)

    def two_d_softmax(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        x = torch.softmax(x, dim=-1)
        return x.view(b, c, h, w)

    def forward(self, im, meta=None):
        x = self.dpt(im)  # [B, C, H, W]

        # Make sure output matches input spatial size
        if x.shape[-2:] != im.shape[-2:]:
            x = F.interpolate(x, size=im.shape[-2:], mode="bilinear", align_corners=False)

        x = self.bn(x)
        return self.two_d_softmax(x)
