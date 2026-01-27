import imgaug.augmenters as iaa
from torchvision import transforms
from skimage.util import random_noise
import numpy as np
import cv2
import torch
import imgaug.augmenters.imgcorruptlike as ic
import matplotlib.pyplot as plt

class Augmentation():
    def __init__(self,cfg) -> None:
        self.datasetname = cfg.INPUT_PATHS.DATASET_NAME
        
        self.downsampled_image_width = cfg.DATASET.CACHED_IMAGE_SIZE[0]
        self.downsampled_image_height = cfg.DATASET.CACHED_IMAGE_SIZE[1]
        self.downsampled_aspect_ratio = self.downsampled_image_width / self.downsampled_image_height
        self.upsample_aspect_ratio = self.downsampled_image_height / self.downsampled_image_width

        self.data_aug_params = cfg.DATASET.AUGMENTATION
        self.data_aug_some_of = cfg.DATASET.AUGMENTATION.SOME_OF

        return

    def downsample_and_padd(self,):
        # Define how to downsample and pad images
        preprocessing_steps = [
            iaa.Crop(px=1),
            iaa.PadToAspectRatio(self.downsampled_aspect_ratio, position='right-bottom', pad_mode='edge'),
            iaa.Resize({"width": self.downsampled_image_width, "height": self.downsampled_image_height}),
        ]
        seq = iaa.Sequential(preprocessing_steps)
        return seq
    
    def upsample(self,orig_size,image):

        new_size = (int(orig_size[0][0]), int(orig_size[0][1]))

        # Resize the image
        if len(image.shape)<3:
            resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        else:
            img = np.reshape(image, (image.shape[1],image.shape[2],image.shape[0]))
            new_size = int(orig_size[0][1]), int(orig_size[0][0])
            resized_image = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
            resized_image = np.reshape(resized_image, (resized_image.shape[2], resized_image.shape[0],resized_image.shape[1]))
            for channel in range(resized_image.shape[0]):
                resized_image[channel] = (resized_image[channel] - resized_image[channel].min()) / (resized_image[channel].max()-resized_image[channel].min())
        return resized_image
    
    def reverse_downsample_and_pad(self, orig_shape, image):
        """
        Reverse downsample_and_pad: undo crop, padding, and resize to original shape.

        Args:
            orig_shape: Tuple (orig_h, orig_w) - desired output size (e.g. (558, 982))
            image: np.ndarray (C, H, W) - downsampled and padded image (e.g. (7, 512, 512))

        Returns:
            np.ndarray of shape (C, orig_h, orig_w) - restored image
        """
        orig_h, orig_w = orig_shape[0]
        channels = image.shape[0]

        # Undo crop: original crop removed 1 px from all sides (so -2 total in each direction)
        cropped_h = orig_h - 2
        cropped_w = orig_w - 2

        target_aspect = self.downsampled_aspect_ratio
        cropped_aspect = cropped_w / cropped_h

        # Compute padded size before resizing
        if cropped_aspect > target_aspect:
            padded_w = int(cropped_w)
            padded_h = int(round(padded_w / target_aspect))
        else:
            padded_h = int(cropped_h)
            padded_w = int(round(padded_h * target_aspect))

        # Step 1: Resize each channel to padded size
        resized_to_padded = np.zeros((channels, padded_h, padded_w), dtype=image.dtype)
        ## larger size (C, )

        for c in range(channels):
            resized_to_padded[c] = cv2.resize(image[c], (padded_w, padded_h), interpolation=cv2.INTER_LINEAR)
    
        # Step 2: Remove padding from right and bottom (it was added there during preprocessing)
        cropped = resized_to_padded[:, :int(cropped_h), :int(cropped_w)]

        # Step 3: Add back the 1px border crop removed earlier
        restored_cropped = np.pad(cropped, ((0, 0), (1, 1), (1, 1)), mode='edge')  # shape: (C, orig_h, orig_w)

        # Step 4: Finally, resize to original full shape (orig_h, orig_w)
        final_restored = np.zeros((channels, int(orig_h), int(orig_w)), dtype=image.dtype)
        for c in range(channels):
            final_restored[c] = cv2.resize(restored_cropped[c], (int(orig_w), int(orig_h)), interpolation=cv2.INTER_LINEAR)

        return torch.from_numpy(final_restored).unsqueeze(0)

    
    def augmentation_fn(self,):
        ## check if you want some of or multiple

        if self.data_aug_some_of == None:
            try:
                print('some of is NONE so all applied')
                aug =iaa.Sequential([
                iaa.Affine(translate_percent={"x": (-self.data_aug_params.TRANSLATION_X, self.data_aug_params.TRANSLATION_X),
                                            "y": (-self.data_aug_params.TRANSLATION_Y, self.data_aug_params.TRANSLATION_Y)},
                        scale=(1 - self.data_aug_params.SF, 1 + self.data_aug_params.SF),
                        rotate=(-self.data_aug_params.ROTATION_FACTOR, self.data_aug_params.ROTATION_FACTOR),
                        mode='edge'),
                iaa.Multiply(mul=(1 - self.data_aug_params.INTENSITY_FACTOR, 1 + self.data_aug_params.INTENSITY_FACTOR)),
                iaa.CoarseSaltAndPepper(self.data_aug_params.COARSE_SALTANDPEPPER, size_px=(4, 16)),
                iaa.ElasticTransformation(alpha=(0, self.data_aug_params.ELASTIC_STRENGTH),
                                        sigma=self.data_aug_params.ELASTIC_SMOOTHNESS, order=3,
                                        mode='nearest'),
                iaa.imgcorruptlike.SpeckleNoise(severity=np.random.randint(1,self.data_aug_params.SPECKLE_NOISE+1))
            ],
            random_order=True
            )
                seq = iaa.Sequential(aug)

            except:
                aug =iaa.Sequential([
                iaa.Affine(translate_percent={"x": (-self.data_aug_params.TRANSLATION_X, self.data_aug_params.TRANSLATION_X),
                                            "y": (-self.data_aug_params.TRANSLATION_Y, self.data_aug_params.TRANSLATION_Y)},
                        scale=(1 - self.data_aug_params.SF, 1 + self.data_aug_params.SF),
                        rotate=(-self.data_aug_params.ROTATION_FACTOR, self.data_aug_params.ROTATION_FACTOR),
                        mode='edge'),
                iaa.CoarseSaltAndPepper(self.data_aug_params.COARSE_SALTANDPEPPER, size_px=(4, 16)),
                iaa.Multiply(mul=(1 - self.data_aug_params.INTENSITY_FACTOR, 1 + self.data_aug_params.INTENSITY_FACTOR)),
                iaa.ElasticTransformation(alpha=(0, self.data_aug_params.ELASTIC_STRENGTH),
                                        sigma=self.data_aug_params.ELASTIC_SMOOTHNESS, order=3,
                                        mode='nearest')],
                random_order=True)
                seq = iaa.Sequential(aug)

        else:
            # print('some augmentation: ', self.data_aug_some_of)
            try:
                #try this for image but for keypoints do no speckle

                if self.datasetname == 'oai':
                    ###cant really move off screen images are too small so do not do negative translation
                    aug =iaa.SomeOf((1, self.data_aug_some_of),[
                    iaa.Affine(translate_percent={"x": (0, self.data_aug_params.TRANSLATION_X),
                                                "y": (-self.data_aug_params.TRANSLATION_Y,0)},
                            scale=(1 - self.data_aug_params.SF, 1 + self.data_aug_params.SF),
                            rotate=(0, self.data_aug_params.ROTATION_FACTOR),
                            mode='edge'),

                    # ic.SpeckleNoise(severity=np.random.randint(1,self.data_aug_params.SPECKLE_NOISE+1)),
                    iaa.CoarseSaltAndPepper(self.data_aug_params.COARSE_SALTANDPEPPER, size_px=(4, 16)),
                    iaa.Multiply(mul=(1 - self.data_aug_params.INTENSITY_FACTOR, 1 + self.data_aug_params.INTENSITY_FACTOR)),
                    iaa.ElasticTransformation(alpha=(0, self.data_aug_params.ELASTIC_STRENGTH),
                                            sigma=self.data_aug_params.ELASTIC_SMOOTHNESS, order=3,
                                            mode='nearest')],
                    random_order=True)
                    seq = iaa.Sequential(aug)
                else:
                    aug =iaa.SomeOf((1, self.data_aug_some_of),[
                    iaa.Affine(translate_percent={"x": (-self.data_aug_params.TRANSLATION_X, self.data_aug_params.TRANSLATION_X),
                                                "y": (-self.data_aug_params.TRANSLATION_Y, self.data_aug_params.TRANSLATION_Y)},
                            scale=(1 - self.data_aug_params.SF, 1 + self.data_aug_params.SF),
                            rotate=(-self.data_aug_params.ROTATION_FACTOR, self.data_aug_params.ROTATION_FACTOR),
                            mode='edge'),

                    # ic.SpeckleNoise(severity=np.random.randint(1,self.data_aug_params.SPECKLE_NOISE+1)),
                    iaa.CoarseSaltAndPepper(self.data_aug_params.COARSE_SALTANDPEPPER, size_px=(4, 16)),
                    iaa.Multiply(mul=(1 - self.data_aug_params.INTENSITY_FACTOR, 1 + self.data_aug_params.INTENSITY_FACTOR)),
                    iaa.ElasticTransformation(alpha=(0, self.data_aug_params.ELASTIC_STRENGTH),
                                            sigma=self.data_aug_params.ELASTIC_SMOOTHNESS, order=3,
                                            mode='nearest')],
                    random_order=True)
                    seq = iaa.Sequential(aug)

            except:
                raise ValueError

        return seq
    
    def augmentation_fn_testtime(self):
        """
        Simple, invertible augmentation pipeline for test-time augmentation (TTA).
        Re-uses original helper `downsample_and_padd()` and the existing
        self.data_aug_params values so no new globals/params are needed.

        IMPORTANT:
        - Keep only geometric transforms (Affine, flips, 90deg rotations).
        - Make deterministic with seq.to_deterministic() before applying:
                det = seq.to_deterministic()
                aug_img = det.augment_image(img_hwc)
        - Invert dense outputs with imgaug HeatmapsOnImage:
                heat = HeatmapsOnImage(pred_arr, shape=aug_img.shape)
                pred_back = det.inverse(heat).get_arr()
        """
        # re-use your downsample/pad + resize pipeline as the first step (if desired)
        # If you prefer to run TTA after downsampling externally, you can omit this line.
        preproc = self.downsample_and_padd()

        # Small geometric affine (translation, scale, rotation) — use your params
        affine = iaa.Affine(
            translate_percent={"x": (-self.data_aug_params.TRANSLATION_X, self.data_aug_params.TRANSLATION_X),
                            "y": (-self.data_aug_params.TRANSLATION_Y, self.data_aug_params.TRANSLATION_Y)},
            scale=(1 - self.data_aug_params.SF, 1 + self.data_aug_params.SF),
            rotate=(-self.data_aug_params.ROTATION_FACTOR, self.data_aug_params.ROTATION_FACTOR),
            mode='edge'
        )
        seq = iaa.Sequential([
            preproc,       # optional preprocessing (crop/pad/resize) — same as training pipeline
            affine,
        ], random_order=False)

        return seq

    # Put this inside your Augmentation class (or place in the same module and call via self)
    import numpy as np
    import cv2
    import torch

    # ---- helpers (use your existing methods if already present) ----
    def _build_affine_matrix(self, center, angle_deg, scale, tx_pixels, ty_pixels):
        M = cv2.getRotationMatrix2D(center, angle_deg, scale)  # 2x3
        M = M.astype(np.float64)
        M[0,2] += tx_pixels
        M[1,2] += ty_pixels
        return M.astype(np.float32)

    def apply_augmentations_cv2_local(self, img_hwc: np.ndarray,
        *,
        pad_to: tuple = None,
        resize_to: tuple = None,
        affine_params: dict = None,
        flip: str = None):
        """
        Minimal deterministic pipeline: pad (right/bottom), resize, affine, flip.
        Returns aug_img_hwc, record.
        """
        H0, W0 = img_hwc.shape[:2]
        record = {"orig_shape": (H0, W0)}
        aug = img_hwc.copy()

        # Pad to target (right/bottom)
        if pad_to is not None:
            Ht, Wt = pad_to
            pad_top = pad_left = 0
            pad_bottom = max(0, Ht - H0)
            pad_right = max(0, Wt - W0)
            if pad_bottom > 0 or pad_right > 0:
                aug = cv2.copyMakeBorder(aug, pad_top, pad_bottom, pad_left, pad_right,
                                        borderType=cv2.BORDER_REPLICATE)
            record["pad"] = {"applied": True, "pad_top": pad_top, "pad_bottom": pad_bottom,
                            "pad_left": pad_left, "pad_right": pad_right}
        else:
            record["pad"] = {"applied": False, "pad_top": 0, "pad_bottom": 0, "pad_left": 0, "pad_right": 0}

        # Resize to network input
        if resize_to is not None:
            Hr, Wr = resize_to
            aug = cv2.resize(aug, dsize=(Wr, Hr), interpolation=cv2.INTER_LINEAR)
            record["resize"] = {"applied": True, "to": (Hr, Wr)}
        else:
            record["resize"] = {"applied": False}

        # Affine (percent or px)
        if affine_params is not None:
            angle = float(affine_params.get("angle_deg", 0.0))
            scale = float(affine_params.get("scale", 1.0))
            H_aug, W_aug = aug.shape[:2]
            if "translate_px" in affine_params:
                tx, ty = affine_params["translate_px"]
            else:
                tx_frac, ty_frac = affine_params.get("translate_percent", (0.0, 0.0))
                tx = tx_frac * W_aug
                ty = ty_frac * H_aug
            center = (W_aug / 5.0, H_aug / 2.0)
            M = self._build_affine_matrix(center=center, angle_deg=angle, scale=scale, tx_pixels=tx, ty_pixels=ty)
            aug = cv2.warpAffine(aug, M, dsize=(W_aug, H_aug), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            record["affine"] = {"applied": True, "M": M, "center": center,
                                "angle_deg": angle, "scale": scale, "translate_px": (tx,ty)}
        else:
            record["affine"] = {"applied": False}

        # Flip
        if flip is not None:
            if flip == "h":
                aug = cv2.flip(aug, 1)
                record["flip"] = {"applied": True, "mode": "h"}
            elif flip == "v":
                aug = cv2.flip(aug, 0)
                record["flip"] = {"applied": True, "mode": "v"}
            else:
                raise ValueError("flip must be 'h' or 'v'")
        else:
            record["flip"] = {"applied": False, "mode": None}

        return aug, record

    def invert_prediction_cv2_local(self, pred_aug_torch: torch.Tensor, record: dict, *, output_size: tuple = None, interp=cv2.INTER_LINEAR, device=None):
        """
        Invert recorded augmentations on dense prediction (1,C,H_aug,W_aug) -> (1,C,H_out,W_out)
        """
        assert isinstance(pred_aug_torch, torch.Tensor)
        pred_np = pred_aug_torch.squeeze(0).permute(1,2,0).detach().cpu().numpy().astype(np.float32)

        # 1) inverse flip
        flip = record.get("flip", {})
        if flip.get("applied", False):
            if flip.get("mode") == "h":
                pred_np = cv2.flip(pred_np, 1)
            elif flip.get("mode") == "v":
                pred_np = cv2.flip(pred_np, 0)

        # 2) inverse affine
        affine = record.get("affine", {})
        if affine.get("applied", False):
            M = affine["M"]
            M3 = np.vstack([M.astype(np.float64), [0.0,0.0,1.0]])
            M3_inv = np.linalg.inv(M3)
            M_inv = M3_inv[:2,:].astype(np.float32)
            H_aug, W_aug = pred_np.shape[:2]
            pred_np = cv2.warpAffine(pred_np, M_inv, dsize=(W_aug, H_aug), flags=interp, borderMode=cv2.BORDER_REPLICATE)

        # 3) inverse resize -> map resized dims back to padded dims or original
        resize = record.get("resize", {})
        pad = record.get("pad", {})
        H0, W0 = record["orig_shape"]
        if resize.get("applied", False):
            if pad.get("applied", False):
                padded_h = int(H0 + pad.get("pad_bottom", 0))
                padded_w = int(W0 + pad.get("pad_right", 0))
                target_h, target_w = padded_h, padded_w
            else:
                target_h, target_w = H0, W0
            pred_np = cv2.resize(pred_np, dsize=(int(target_w), int(target_h)), interpolation=interp)

        # 4) inverse pad (crop right/bottom)
        if pad.get("applied", False):
            pred_np = pred_np[:H0, :W0, :]

        # final explicit output size
        if output_size is not None:
            H_out, W_out = output_size
            if (pred_np.shape[0] != H_out) or (pred_np.shape[1] != W_out):
                pred_np = cv2.resize(pred_np, dsize=(int(W_out), int(H_out)), interpolation=interp)
        else:
            H_out, W_out = H0, W0

        pred_back_t = torch.from_numpy(pred_np).permute(2,0,1).unsqueeze(0).float()
        if device is not None:
            pred_back_t = pred_back_t.to(device)
        return pred_back_t

    # ---- main TTA wrapper using your config values ----
    def tta_predict_with_config(self, model, img_hwc: np.ndarray, meta_data=None,
                                device=None,
                                include_identity=True,
                                include_center_scale=True,
                                include_translations=True,
                                include_rotations=True,
                                flips=(None,)):
        """
        Run deterministic TTA using translate_percent and SF from self.data_aug_params.
        - model: callable that maps (1, C, H, W) -> (1, C_out, H_pred, W_pred)
        - img_hwc: original image (H, W, C) numpy (uint8 or float32)
        - returns aggregated prediction (1, C_out, H_out, W_out) on device (or CPU if device None)
        """

        # infer target network size (your downsampled network input)
        net_H = int(self.downsampled_image_height)
        net_W = int(self.downsampled_image_width)

        # translation percents from config (range values are +/- TRANSLATION_X)
        tx_pct = float(self.data_aug_params.TRANSLATION_X)   # example: 0.05
        ty_pct = float(self.data_aug_params.TRANSLATION_Y)
        sf = float(self.data_aug_params.SF)                  # scale factor range (e.g., 0.05)

        rot = float(getattr(self.data_aug_params, "ROTATION_FACTOR", 0.0))

        # Build candidate lists
        scales = [1.0]
        if include_center_scale:
            scales = [1.1, 1.0 - sf, 1.0 + sf]

        tx_list = [0.0]
        ty_list = [0.0]
        if include_translations:
            tx_list = [0.0, tx_pct, 2*tx_pct]
            ty_list = [0.0, ty_pct/4, ty_pct/2]

        rot_list = [0.0]
        if include_rotations and rot != 0.0:
            rot_list = [0.0, -rot, rot]

        # prepare storage for inverted preds
        inverted_preds = []

        # iterate deterministic combinations (small grid)
        for scale in scales:
            for tx_frac in tx_list:
                for ty_frac in ty_list:
                    for angle in rot_list:
                            # build affine params using percent (these are relative to the augmented image after resize)
                            affine_params = {
                                "angle_deg": float(angle),
                                "scale": float(scale),
                                "translate_percent": (float(tx_frac), float(ty_frac))
                            }

                            # 1) Apply deterministic aug (pad->resize->affine->flip)
                            aug_img, record = self.apply_augmentations_cv2_local(
                                img_hwc,
                                pad_to=(net_H, net_W),      # pad to network input
                                resize_to=(net_H, net_W),
                                affine_params=affine_params,
                            )

                            fig, axes = plt.subplots(1, 2, figsize=(8, 5))

                            axes[0].imshow(aug_img[:, :, 1], cmap="gray")
                            axes[0].set_title("Augmented")
                            axes[0].axis("off")

                            axes[1].imshow(img_hwc[:, :, 1], cmap="gray")
                            axes[1].set_title("Original")
                            axes[1].axis("off")

                            fig.text(0.5, 0.05, str(affine_params), 
                                    ha="center", fontsize=11)                            
                            plt.tight_layout()
                            plt.show()


                            # 2) prepare model input
                            # convert aug_img to torch NCHW float32 (keep value range as model expects)
                            aug_image_t = torch.from_numpy(aug_img).permute(2,0,1).unsqueeze(0).float().to(device) if device is not None else torch.from_numpy(aug_img).permute(2,0,1).unsqueeze(0).float()
                            # If your model needs normalization do it here (same as training), e.g. aug_image_t = normalize(aug_image_t)

                            # 3) forward
                            with torch.no_grad():
                                pred = model(aug_image_t, meta_data) if meta_data is not None else model(aug_image_t)
                                # pred: (1, C_out, H_pred, W_pred)

                            # 4) make sure pred is resized to augmented image size if needed
                            H_aug, W_aug = aug_img.shape[:2]
                            H_pred = pred.shape[2]
                            W_pred = pred.shape[3]
                            if (H_pred != H_aug) or (W_pred != W_aug):
                                # resize each channel to H_aug,W_aug using cv2 (convert to numpy)
                                pred_np = pred.squeeze(0).permute(1,2,0).cpu().numpy()
                                pred_resized = np.zeros((H_aug, W_aug, pred_np.shape[2]), dtype=np.float32)
                                for ch in range(pred_np.shape[2]):
                                    pred_resized[:,:,ch] = cv2.resize(pred_np[:,:,ch], dsize=(W_aug, H_aug), interpolation=cv2.INTER_LINEAR)
                                pred_aug_t = torch.from_numpy(pred_resized).permute(2,0,1).unsqueeze(0).float()
                            else:
                                pred_aug_t = pred

                            # 5) invert using recorded transform
                            pred_back_t = self.invert_prediction_cv2_local(pred_aug_t, record, output_size=(img_hwc.shape[0], img_hwc.shape[1]), interp=cv2.INTER_LINEAR, device=device)
                            inverted_preds.append(pred_back_t.cpu())  # keep on cpu for aggregation

        # Aggregate (mean)
        stacked = torch.stack(inverted_preds, dim=0)  # (N,1,C,H,W)
        # collapse first two dims: N x 1 -> N
        stacked = stacked.squeeze(1)  # (N, C, H, W)
        agg = stacked.mean(dim=0, keepdim=True)  # (1, C, H, W)

        if device is not None:
            agg = agg.to(device)

        return agg
    