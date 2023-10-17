
import glob
import numpy as np
import os
import imgaug.augmenters as iaa
import json
import tqdm
from skimage import io
from skimage import img_as_ubyte
from PIL import Image
from imgaug.augmentables import KeypointsOnImage
from .metadata_import import MetadataImport
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import pandas as pd

class dataloader():
    def __init__(self, cfg) -> None:
        #paths
        self.dataset_name = cfg.INPUT_PATHS.DATASET_NAME
        self.partition_file_path = cfg.INPUT_PATHS.PARTITION
        self.img_dir = cfg.INPUT_PATHS.IMAGES
        self.label_dir = cfg.INPUT_PATHS.LABELS
        self.metadata = cfg.INPUT_PATHS.META_PATH
        self.cols = cfg.INPUT_PATHS.META_COLS
        self.output_path = cfg.OUTPUT_PATH
        self.cache_dir = self.output_path+'/cache'
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        #downsample size
        self.downsampled_image_width = cfg.DATASET.CACHED_IMAGE_SIZE[0]
        self.downsampled_image_height = cfg.DATASET.CACHED_IMAGE_SIZE[1]
        self.downsampled_aspect_ratio = self.downsampled_image_width / self.downsampled_image_height

        # Dataset parameters
        self.flip_axis = cfg.DATASET.FLIP_AXIS
        self.img_extension = cfg.DATASET.IMAGE_EXT
        self.annotation_type = cfg.DATASET.ANNOTATION_TYPE
        if self.annotation_type == 'LANDMARKS':
            self.ann_extension = '.txt'
        self.num_landmarks = cfg.DATASET.NUM_LANDMARKS
        self.sigma=cfg.DATASET.SIGMA
        
        self.metaimport = MetadataImport(cfg)
        self.metadata_csv = self.metaimport.load_csv()
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
    
    def get_partition(self):
        # open partition
        if self.partition_file_path:
            partition_file = open(self.partition_file_path)
            partition_dict = json.load(partition_file)

            # get the file names of all images in the directory
            img_file_dic = {}
            annotation_dic = {}

            for key in partition_dict.keys():
                img_file_dic[key]=[]
                annotation_dic[key]=[]

            for i in partition_dict:
                for id in partition_dict[i]:
                    #get image file
                    img_file_dic[i].append(os.path.join(self.img_dir, id + self.img_extension))
                    #add annotation folders
                    ann_list = []
                    for label_folder in os.listdir(self.label_dir):
                        ann_list.append(os.path.join(self.label_dir, label_folder,id + self.ann_extension))
                    
                    annotation_dic[i].append(ann_list)
            
        else:
            raise ValueError('Check partition dir')

        return img_file_dic, annotation_dic
    
    def get_img(self, seq, img_path:str):
        '''loads images as arr'''
        image = io.imread(img_path, as_gray=True)
        # Augment image
        image_resized = seq(image=image)
        image_rescaled = (image_resized - np.min(image_resized)) / np.ptp(image_resized)
        image_as_255 = img_as_ubyte(image_rescaled)
        return image_as_255, image.shape
    
    def get_landmarks(self, ann_path, seq, image_shape):
        # Get annotations
        kps_np_array = np.loadtxt(ann_path, usecols=(0, 1),delimiter=',', max_rows=self.num_landmarks)
        if self.flip_axis:
            kps_np_array = np.flip(kps_np_array, axis=1)
        # Augment landmark annotations
        kps = KeypointsOnImage.from_xy_array(kps_np_array, shape=image_shape)
        landmarks_arr = seq(keypoints=kps)
        return landmarks_arr

    def add_sigma_channels(self, ann_points, img_shape, plot=False):
        '''create an array with channel for each landmark, with sigma applied for each'''
        channels = np.zeros([self.num_landmarks, img_shape[0], img_shape[1]])
        for i in range(self.num_landmarks):
            x, y = (ann_points[i]).astype(int)
            if 0 <= y < channels.shape[1] and 0 <= x < channels.shape[2]:
                channels[i, y, x] = 1.0
                if self.sigma:
                    channels[i] = gaussian(channels[i], sigma=self.sigma)
                
        #debug: check plot is correct of guassian array
        # Plot the 2D image
        if plot==True:
            plt.imshow(channels[0], cmap='gray')
            plt.show()
                
        return channels

    def get_ann(self, ann_paths:str, seq, img_shape, orig_img_shape):
        '''loads annotations, loads array for each folder if there are subfolders in the label folder'''
        #Get sub-directories for annotations 
        ann_points, ann_array, folder_ls = [], [], []
        for ann_path in ann_paths:
            folder_name = ann_path.split("/")[-2]
            if self.annotation_type=="LANDMARKS":
                _ann_points = self.get_landmarks(ann_path, seq, orig_img_shape)
                _np_ann_points=_ann_points.to_xy_array().reshape(-1, self.num_landmarks, 2)[0]
                #get array of the points with guassian if applied
                _ann_array = self.add_sigma_channels(_np_ann_points, img_shape)
            else:
                ann_points = 'None'
                raise ValueError('only capable for landmarks right now')
            
            #add annotations points and arrays to a list if multiple annotators exist
            ann_points.append(_np_ann_points)
            ann_array.append(_ann_array)
            folder_ls.append(folder_name)

        return ann_array,ann_points,folder_ls
    
    def get_numpy_dataset(self, set, save_cache=True):
        '''this function inputs the cfg and gets out a numpy array of the desired input structure'''

        np_dataset, meta_arr = np.array([]), pd.DataFrame([])

        #initalize downsample/padd
        seq = self.downsample_and_padd()
        
        img_files, annotation_files = self.get_partition()
        
        print('loading:', set)

        for i in range(len(img_files[set])):
            pat_id = img_files[set][i].split('/')[-1].split('.')[0]

            ##LOAD IMAGES##
            _im_arr, orig_shape = self.get_img(seq,img_files[set][i])
            
            ##LOAD ANNOTATIONS##
            _annotation_arr, annotation_points, folder_ls = self.get_ann(annotation_files[set][i], seq, _im_arr.shape, orig_shape)

            ##LOAD META##
            _meta_arr = self.metaimport._get_array(self.metadata_csv, pat_id)

            if save_cache==True:
                #save input data files for debugging
                cache_data_dir = os.path.join(self.cache_dir, "{}_{}".format(self.downsampled_image_width, self.downsampled_image_height))
                if not os.path.exists(cache_data_dir):
                    os.makedirs(cache_data_dir)

                #save img
                _im = Image.fromarray(_im_arr)
                _im.save(os.path.join(cache_data_dir,pat_id+self.img_extension))
                #save annonation points
                for i in range(len(folder_ls)):
                    ann_folder = folder_ls[i]
                    annotation = annotation_points[i] 
                    np.savetxt(os.path.join(cache_data_dir,pat_id+ann_folder+'.txt'), annotation, fmt="%.14g", delimiter=" ")
            
            if meta_arr.empty:
                meta_arr = _meta_arr
                im_arr = np.expand_dims(_im_arr,axis=0)
                annotation_arr =np.expand_dims(_annotation_arr,axis=0)
            else:
                im_arr = np.concatenate((im_arr, np.expand_dims(_im_arr,axis=0)),0)
                annotation_arr = np.concatenate((annotation_arr,np.expand_dims(_annotation_arr,axis=0)),0)
                meta_arr=pd.concat([meta_arr,_meta_arr])

        if save_cache==True:                      
            #save meta dictionary
            meta_path = os.path.join(cache_data_dir,'all_meta.csv')
            meta_arr.to_csv(meta_path)

        return im_arr, annotation_arr, np.array(meta_arr)
