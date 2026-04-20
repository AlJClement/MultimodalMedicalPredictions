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
            dynamic_img_size=cfg.MODEL.DYNAMIC_IMG_SIZE,
        )
        self._configure_dynamic_input_size()

    def _configure_dynamic_input_size(self):
        if not getattr(self.dpt, "encoder", None):
            return

        encoder = self.dpt.encoder
        img_size = tuple(self.im_size)
        self._set_module_input_size(encoder, img_size)

    def _set_module_input_size(self, module, img_size):
        if module is None:
            return

        # SMP+timm compatibility varies by version, so update both public APIs
        # and the internal attributes used by Swin/ViT patch embedding code.
        if hasattr(module, "set_input_size"):
            try:
                module.set_input_size(img_size=img_size)
            except TypeError:
                pass

        if hasattr(module, "dynamic_img_size"):
            module.dynamic_img_size = True
        if hasattr(module, "strict_img_size"):
            module.strict_img_size = False
        if hasattr(module, "img_size"):
            module.img_size = img_size

        patch_embed = getattr(module, "patch_embed", None)
        if patch_embed is not None and patch_embed is not module:
            self._set_module_input_size(patch_embed, img_size)

        model = getattr(module, "model", None)
        if model is not None and model is not module:
            self._set_module_input_size(model, img_size)

        for child in module.children():
            self._set_module_input_size(child, img_size)

    def _get_encoder_input_size(self):
        encoder = getattr(self.dpt, "encoder", None)
        if encoder is None:
            return None

        candidates = [
            getattr(getattr(encoder, "model", None), "patch_embed", None),
            getattr(encoder, "patch_embed", None),
            getattr(encoder, "model", None),
            encoder,
        ]

        for module in candidates:
            if module is None:
                continue
            img_size = getattr(module, "img_size", None)
            if img_size is None:
                continue
            if isinstance(img_size, int):
                return (img_size, img_size)
            return tuple(img_size)

        return None

    def _sync_encoder_device(self, device):
        encoder = getattr(self.dpt, "encoder", None)
        if encoder is not None:
            encoder.to(device)

    def two_d_softmax(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1).float()
        x = x - x.amax(dim=-1, keepdim=True)
        x = torch.softmax(x, dim=-1)
        return x.reshape(b, c, h, w)

    def forward(self, im, meta=None):
        if getattr(self.dpt, "encoder", None):
            self._set_module_input_size(self.dpt.encoder, tuple(im.shape[-2:]))
            self._sync_encoder_device(im.device)
        try:
            x = self.dpt(im)  # [B, C, H, W]
        except AssertionError as exc:
            if "doesn't match model" not in str(exc):
                raise

            target_size = self._get_encoder_input_size()
            if target_size is None:
                raise

            resized_im = F.interpolate(im, size=target_size, mode="bilinear", align_corners=False)
            self._sync_encoder_device(resized_im.device)
            x = self.dpt(resized_im)

        # Make sure output matches input spatial size
        if x.shape[-2:] != im.shape[-2:]:
            x = F.interpolate(x, size=im.shape[-2:], mode="bilinear", align_corners=False)

        return self.two_d_softmax(x)
