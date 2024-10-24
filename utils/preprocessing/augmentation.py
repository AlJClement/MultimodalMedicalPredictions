import imgaug.augmenters as iaa
from torchvision import transforms
from skimage.util import random_noise
import numpy as np

class Augmentation():
    def __init__(self,cfg) -> None:
        
        self.downsampled_image_width = cfg.DATASET.CACHED_IMAGE_SIZE[0]
        self.downsampled_image_height = cfg.DATASET.CACHED_IMAGE_SIZE[1]
        self.downsampled_aspect_ratio = self.downsampled_image_width / self.downsampled_image_height
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
    
    def augmentation_fn(self,):
        ## check if you want some of or multiple
        if self.data_aug_some_of != None:
            aug =iaa.Sequential([
                iaa.Affine(translate_percent={"x": (-self.data_aug_params.TRANSLATION_X, self.data_aug_params.TRANSLATION_X),
                                            "y": (-self.data_aug_params.TRANSLATION_Y, self.data_aug_params.TRANSLATION_Y)},
                        scale=(1 - self.data_aug_params.SF, 1 + self.data_aug_params.SF),
                        rotate=(-self.data_aug_params.ROTATION_FACTOR, self.data_aug_params.ROTATION_FACTOR),
                        mode='edge'),
                iaa.Multiply(mul=(1 - self.data_aug_params.INTENSITY_FACTOR, 1 + self.data_aug_params.INTENSITY_FACTOR)),
                iaa.ElasticTransformation(alpha=(0, self.data_aug_params.ELASTIC_STRENGTH),
                                        sigma=self.data_aug_params.ELASTIC_SMOOTHNESS, order=3,
                                        mode='nearest'),
                iaa.imgcorruptlike.SpeckleNoise(severity=np.random.randint(1,self.data_aug_params.SPECKLE_NOISE+1))
            ],
            random_order=True
            )
            seq = iaa.Sequential(aug)
        else:
            aug =iaa.SomeOf(self.data_aug_some_of,[
                iaa.Affine(translate_percent={"x": (-self.data_aug_params.TRANSLATION_X, self.data_aug_params.TRANSLATION_X),
                                            "y": (-self.data_aug_params.TRANSLATION_Y, self.data_aug_params.TRANSLATION_Y)},
                        scale=(1 - self.data_aug_params.SF, 1 + self.data_aug_params.SF),
                        rotate=(-self.data_aug_params.ROTATION_FACTOR, self.data_aug_params.ROTATION_FACTOR),
                        mode='edge'),
                iaa.Multiply(mul=(1 - self.data_aug_params.INTENSITY_FACTOR, 1 + self.data_aug_params.INTENSITY_FACTOR)),
                # iaa.ElasticTransformation(alpha=(0, self.data_aug_params.ELASTIC_STRENGTH),
                #                         sigma=self.data_aug_params.ELASTIC_SMOOTHNESS, order=3,
                #                         mode='nearest'),
                iaa.imgcorruptlike.SpeckleNoise(severity=np.random.randint(1,self.data_aug_params.SPECKLE_NOISE))
                ],
                random_order=True
                )
            seq = iaa.Sequential(aug)

        return seq
    