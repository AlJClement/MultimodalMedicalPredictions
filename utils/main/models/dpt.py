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
        self.encoder_name = str(cfg.MODEL.ENCODER_NAME)
        self.channel_type = str(getattr(cfg.MODEL, "CHANNEL_TYPE", "none")).strip().lower()
        self.meta_cols = list(getattr(cfg.INPUT_PATHS, "META_COLS", []))
        self.meta_feature_widths = list(getattr(cfg.MODEL, "META_FEATURES", []))
        self.meta_attention_indices = self._resolve_attention_feature_indices()
        self.meta_attention_labels = self._resolve_attention_labels()

        dpt_kwargs = dict(
            encoder_name=cfg.MODEL.ENCODER_NAME,
            encoder_weights=cfg.MODEL.ENCODER_WEIGHTS,
            in_channels=cfg.MODEL.IN_CHANNELS,
            classes=cfg.MODEL.OUT_CHANNELS,
            activation=None,
        )
        if self._supports_dynamic_img_size() and bool(getattr(cfg.MODEL, "DYNAMIC_IMG_SIZE", True)):
            dpt_kwargs["dynamic_img_size"] = True

        self.dpt = smp.DPT(**dpt_kwargs)
        self._configure_dynamic_input_size()
        self.latest_channel_attention = None
        self.latest_input_pre_multimodal = None
        self.latest_input_post_multimodal = None

        self.meta_channel_attention = None
        self.meta_channel_attention_scale = None
        self.meta_channel_attention_bias = None
        if self.channel_type == "multimodal":
            attention_dim = max(1, len(self.meta_attention_indices))
            if attention_dim != self.in_channels:
                raise ValueError(
                    f"multimodal channel gating expects one metadata summary per image channel, "
                    f"but got attention_dim={attention_dim} and in_channels={self.in_channels}"
                )
            self.meta_channel_attention_scale = nn.Parameter(torch.ones(attention_dim))
            self.meta_channel_attention_bias = nn.Parameter(torch.zeros(attention_dim))

    def _supports_dynamic_img_size(self):
        encoder_name = self.encoder_name.lower()
        dynamic_families = (
            "swin",
            "vit",
            "deit",
            "beit",
            "eva",
            "samvit",
        )
        return any(family in encoder_name for family in dynamic_families)

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

    def _resolve_attention_feature_indices(self):
        if not self.meta_cols:
            return []

        matched_indices = []
        used_indices = set()
        feature_names = [
            str(list(col_spec.items())[0][0]).strip().lower()
            for col_spec in self.meta_cols
        ]
        desired_groups = (
            ("age",),
            ("sex", "gender"),
            ("laterality", "side", "left", "right"),
        )

        for group in desired_groups:
            for idx, name in enumerate(feature_names):
                if idx in used_indices:
                    continue
                if any(token in name for token in group):
                    matched_indices.append(idx)
                    used_indices.add(idx)
                    break

        for idx in range(len(feature_names)):
            if len(matched_indices) >= 3:
                break
            if idx not in used_indices:
                matched_indices.append(idx)
                used_indices.add(idx)

        return matched_indices

    def _resolve_attention_labels(self):
        labels = []
        for idx in self.meta_attention_indices:
            if idx >= len(self.meta_cols):
                continue
            col_name, _encoding = list(self.meta_cols[idx].items())[0]
            labels.append(str(col_name))
        return labels

    def _reshape_meta(self, meta):
        if meta is None:
            return None

        if not torch.is_tensor(meta):
            meta = torch.as_tensor(meta, dtype=torch.float32, device=next(self.parameters()).device)

        meta = meta.float()
        if meta.ndim == 3 and meta.shape[1] == 1:
            meta = meta.squeeze(1)
        elif meta.ndim > 2:
            meta = meta.reshape(meta.shape[0], -1)

        return meta

    def _summarize_meta_features(self, meta):
        meta = self._reshape_meta(meta)
        if meta is None:
            return None

        if meta.ndim == 1:
            meta = meta.unsqueeze(0)

        if not self.meta_cols or not self.meta_feature_widths:
            return meta[:, : min(3, meta.shape[-1])]

        summaries = []
        start = 0
        for idx, _col_spec in enumerate(self.meta_cols):
            if idx >= len(self.meta_feature_widths):
                break
            width = int(self.meta_feature_widths[idx])
            if width <= 0:
                continue
            end = min(start + width, meta.shape[-1])
            if end <= start:
                break
            summaries.append(meta[:, start:end].mean(dim=-1, keepdim=True))
            start = end

        if not summaries:
            return meta[:, : min(3, meta.shape[-1])]

        return torch.cat(summaries, dim=-1)

    def _apply_multimodal_channel_attention(self, im, meta):
        if self.meta_channel_attention_scale is None or self.meta_channel_attention_bias is None:
            return im

        self.latest_input_pre_multimodal = im.detach().cpu()
        meta_summary = self._summarize_meta_features(meta)
        if meta_summary is None or meta_summary.numel() == 0:
            self.latest_input_post_multimodal = im.detach().cpu()
            return im

        if self.meta_attention_indices:
            available_indices = [idx for idx in self.meta_attention_indices if idx < meta_summary.shape[-1]]
            if available_indices:
                meta_summary = meta_summary[:, available_indices]

        if meta_summary.shape[-1] == 0:
            self.latest_input_post_multimodal = im.detach().cpu()
            return im

        expected_dim = self.meta_channel_attention_scale.numel()
        if meta_summary.shape[-1] < expected_dim:
            pad = meta_summary.new_zeros((meta_summary.shape[0], expected_dim - meta_summary.shape[-1]))
            meta_summary = torch.cat((meta_summary, pad), dim=-1)
        elif meta_summary.shape[-1] > expected_dim:
            meta_summary = meta_summary[:, :expected_dim]

        meta_summary = meta_summary.to(device=im.device, dtype=im.dtype)
        attention_logits = (meta_summary * self.meta_channel_attention_scale) + self.meta_channel_attention_bias
        attention = torch.softmax(attention_logits, dim=-1)
        self.latest_channel_attention = attention.detach().cpu()
        attention = attention.unsqueeze(-1).unsqueeze(-1)
        scaled_im = im * attention
        self.latest_input_post_multimodal = scaled_im.detach().cpu()
        return scaled_im

    def two_d_softmax(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1).float()
        x = x - x.amax(dim=-1, keepdim=True)
        x = torch.softmax(x, dim=-1)
        return x.reshape(b, c, h, w)

    def forward(self, im, meta=None):
        self.latest_channel_attention = None
        self.latest_input_pre_multimodal = None
        self.latest_input_post_multimodal = None
        if self.channel_type == "multimodal":
            im = self._apply_multimodal_channel_attention(im, meta)

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
