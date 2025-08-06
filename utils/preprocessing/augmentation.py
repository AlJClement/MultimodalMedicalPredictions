import imgaug.augmenters as iaa
from torchvision import transforms
from skimage.util import random_noise
import numpy as np
import cv2
import torch
class Augmentation():
    def __init__(self,cfg) -> None:
        
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
        if self.data_aug_some_of != None:
            try:
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
            if self.data_aug_params.SPECKLE_NOISE != 0:

                aug =iaa.SomeOf((0,self.data_aug_some_of),[
                iaa.Affine(translate_percent={"x": (-self.data_aug_params.TRANSLATION_X, self.data_aug_params.TRANSLATION_X),
                                            "y": (-self.data_aug_params.TRANSLATION_Y, self.data_aug_params.TRANSLATION_Y)},
                        scale=(1 - self.data_aug_params.SF, 1 + self.data_aug_params.SF),
                        rotate=(-self.data_aug_params.ROTATION_FACTOR, self.data_aug_params.ROTATION_FACTOR),
                        mode='edge'),
                iaa.Multiply(mul=(1 - self.data_aug_params.INTENSITY_FACTOR, 1 + self.data_aug_params.INTENSITY_FACTOR)),
                iaa.CoarseSaltAndPepper(self.data_aug_params.COARSE_SALTANDPEPPER, size_px=(4, 16)),
                iaa.imgcorruptlike.SpeckleNoise(severity=np.random.randint(1,self.data_aug_params.SPECKLE_NOISE))
                ],
                random_order=True
                )
                seq = iaa.Sequential(aug)
            else:

                aug =iaa.SomeOf((0,self.data_aug_some_of),[
                iaa.Affine(translate_percent={"x": (-self.data_aug_params.TRANSLATION_X, self.data_aug_params.TRANSLATION_X),
                                            "y": (-self.data_aug_params.TRANSLATION_Y, self.data_aug_params.TRANSLATION_Y)},
                        scale=(1 - self.data_aug_params.SF, 1 + self.data_aug_params.SF),
                        rotate=(-self.data_aug_params.ROTATION_FACTOR, self.data_aug_params.ROTATION_FACTOR),
                        mode='edge'),
                iaa.Multiply(mul=(1 - self.data_aug_params.INTENSITY_FACTOR, 1 + self.data_aug_params.INTENSITY_FACTOR)),
                iaa.CoarseSaltAndPepper(self.data_aug_params.COARSE_SALTANDPEPPER, size_px=(4, 16))
                ],
                random_order=True
                )
                seq = iaa.Sequential(aug)


        return seq
    