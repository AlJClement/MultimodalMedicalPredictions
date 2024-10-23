
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
import torch
from main import model_init
from torch.utils.data import Dataset
from visualisations import visuals
from tqdm import tqdm

class dataloader(Dataset):
    def __init__(self, cfg, set) -> None:
        #paths
        self.dataset_name = cfg.INPUT_PATHS.DATASET_NAME
        self.partition_file_path = cfg.INPUT_PATHS.PARTITION
        self.img_dir = cfg.INPUT_PATHS.IMAGES
        self.label_dir = cfg.INPUT_PATHS.LABELS
        self.metadata = cfg.INPUT_PATHS.META_PATH
        self.output_path = cfg.OUTPUT_PATH
        self.cache_dir = self.output_path+'/cache'
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.pixel_size = cfg.DATASET.PIXEL_SIZE
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
        self.cfg_combine_reviewers=cfg.DATASET.COMBINE_REVIEWERS
        
        self.pat_id_col = cfg.INPUT_PATHS.ID_COL
        self.metaimport = MetadataImport(cfg)
        self.metadata_csv = self.metaimport.load_csv()
        self.model_init = model_init(cfg)
        #get specific models/feature loaders
        self.meta_feat_structure = self.model_init.get_modelspecific_feature_structure()

        self.set = set
        self.data, self.target, self.meta, self.ids, self.orig_img_shape = self.get_numpy_dataset() 

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

                    if self.cfg_combine_reviewers==False:
                        #if you want to not combine the reviwers you need to get all possible paths
                        #get image file
                        #add annotation folders
                        ann_list = []
                        if len(os.listdir(self.label_dir))<10:
                            for label_folder in os.listdir(self.label_dir):
                                img_file_dic[i].append(os.path.join(self.img_dir, id + self.img_extension))                                
                                annotation_dic[i].append([os.path.join(self.label_dir, label_folder,id + label_folder + self.ann_extension)])
                        else:    
                            img_file_dic[i].append(os.path.join(self.img_dir, id + self.img_extension))       
                            label_folder = ''                 
                            annotation_dic[i].append([os.path.join(self.label_dir,id + self.ann_extension)])

                    else: 
                        #keeps each patient seperate
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
        try:
            #assumes file is jpg
            image = io.imread(img_path, as_gray=True)
        except:
            #if its saved as png
            try:
                image = io.imread(img_path[:-4]+'.png', as_gray=True)
            except:
                #add _ between L/R and series number
                new_name = img_path.split('/')[-1][:-5]+'_'+img_path.split('/')[-1][-5:-4]+'.png'
                image = io.imread(img_path.rsplit('/',1)[0]+'/'+new_name, as_gray=True)

        # Augment image
        image_resized = seq(image=image)
        image_rescaled = (image_resized - np.min(image_resized)) / np.ptp(image_resized)
        image_as_255 = img_as_ubyte(image_rescaled)
        
        return image_as_255, image.shape
    
    def get_landmarks(self, ann_path, seq, image_shape):
        # Get annotations
        print(ann_path)
        try:
            kps_np_array = np.loadtxt(ann_path, usecols=(0, 1),delimiter=',', max_rows=self.num_landmarks)
        except:
            ext = ann_path.split('/')[-2]+'.txt'
            ext = '.txt'
            kps_np_array = np.loadtxt(ann_path[:-4]+'_g'+ext, usecols=(0, 1),delimiter=',', max_rows=self.num_landmarks)
        
        print('truth:',kps_np_array)
        if self.flip_axis:
            if ann_path.split('/')[-1][0]=='A':
                kps_np_array = np.flip(kps_np_array, axis=1)
            if ann_path.split('/')[-1][0]=='0': #rnoh
                kps_np_array = np.flip(kps_np_array, axis=1)
            if ann_path.split('/')[-1][0]=='R': #rnoh
                kps_np_array = np.flip(kps_np_array, axis=1)
            if ann_path.split('/')[-1][0]=='P': #MKUH
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
    
    def combine_reviewers(self, array):
        '''takes a few annotations and combines them'''
        combined_arr = sum(array)
        return combined_arr

    def get_ann(self, ann_paths:str, seq, img_shape, orig_img_shape):
        '''loads annotations, loads array for each folder if there are subfolders in the label folder'''
        #Get sub-directories for annotations 
        ann_points, ann_array, folder_ls = [], [], []
        #print(ann_paths)
        for ann_path in ann_paths:
            folder_name = ann_path.split("/")[-2]
            if self.annotation_type=="LANDMARKS":
                _ann_points = self.get_landmarks(ann_path, seq, orig_img_shape)
                _np_ann_points=_ann_points.to_xy_array().reshape(-1, self.num_landmarks, 2)[0]
                #get array of the points with guassian if applied
                _ann_array = self.add_sigma_channels(_np_ann_points, img_shape)
                #print(_np_ann_points)
            else:
                ann_points = 'None'
                raise ValueError('only capable for landmarks right now')
            
            #add annotations points and arrays to a list if multiple annotators exist
            ann_points.append(_np_ann_points)
            ann_array.append(_ann_array)
            folder_ls.append(folder_name)
        
        if self.cfg_combine_reviewers==True:
            ann_arr_multireviewer = self.combine_reviewers(ann_array)
        else:
            ann_arr_multireviewer = ann_array

        return ann_arr_multireviewer,ann_points,folder_ls
    
    def get_numpy_dataset(self, save_cache=False):
        '''this function inputs the cfg and gets out a numpy array of the desired input structure'''
        # output is : 
        # [im_arr (numscans, size h, size w),
        # annotation_arr (numscans, numannotations, num_label_channels, size h, size w),
        # meta_arr (numscans,num_metafeatures)]

        #initalize downsample/padd
        seq = self.downsample_and_padd()
        
        img_files, annotation_files = self.get_partition()
        meta_arr = pd.DataFrame([])

        print('loading:', self.set)
        for i in tqdm(range(len(img_files[self.set]))):
            pat_id = img_files[self.set][i].split('/')[-1].split('.')[0]

            ##LOAD IMAGES##
            print(img_files[self.set][i])
            _im_arr, orig_shape = self.get_img(seq,img_files[self.set][i])
            print('orig_shape:',orig_shape)
            
            ##LOAD ANNOTATIONS##
            _annotation_arr, annotation_points, folder_ls = self.get_ann(annotation_files[self.set][i], seq, _im_arr.shape, orig_shape)

            # _annotation_arr = _annotation_arr[:5]
            # annotation_points= annotation_points[:5]
            print(len(_annotation_arr))
            ##LOAD META##
            _meta_arr = self.metaimport._get_array(self.metadata_csv, pat_id)
            if _meta_arr.empty:
                try:
                    pat_id_rnoh=pat_id.split('_')[1]
                    _meta_arr = self.metaimport._get_array(self.metadata_csv, pat_id_rnoh)
                    
                except:
                    raise ValueError('no meta data found for: ', pat_id)

            cache_data_dir = os.path.join(self.cache_dir, "{}_{}".format(self.downsampled_image_width, self.downsampled_image_height))

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

                
                #plot annotation array
                if len(annotation_points)>1:
                    #if its a list loop (*multiple annotators, kept seperate)
                    for aa in _annotation_arr:
                        i=0
                        _a = visuals('',self.pixel_size[0]).channels_thresholded(_annotation_arr)
                        plt.imshow(_a)
                        plt.imsave(os.path.join(cache_data_dir,pat_id+'_gt_map'+folder_ls[i]+self.img_extension),_a)
                        plt.close()
                        i=i+1
                else:
                    _a = visuals('',self.pixel_size[0]).channels_thresholded(_annotation_arr[0])
                    plt.imshow(_a)
                    plt.imsave(os.path.join(cache_data_dir,pat_id+'_gt_map'+self.img_extension),_a)
                    plt.close()

            if meta_arr.empty:
                accessionid_arr = np.array([pat_id])
                id_arr = np.array([pat_id])
                meta_arr = _meta_arr
                im_arr = np.expand_dims(_im_arr,axis=0)
                orig_shape_arr = np.expand_dims(orig_shape,axis=0)
                annotation_arr =np.expand_dims(_annotation_arr,axis=0)
            else:
                accessionid_arr = np.concatenate((accessionid_arr,np.array([pat_id])),0)
                id_arr = np.concatenate((id_arr,np.array([pat_id])),0)
                im_arr = np.concatenate((im_arr, np.expand_dims(_im_arr,axis=0)),0)
                annotation_arr = np.concatenate((annotation_arr,np.expand_dims(_annotation_arr,axis=0)),0)
                meta_arr=pd.concat([meta_arr,_meta_arr])
                orig_shape_arr=np.concatenate((orig_shape_arr,np.array([orig_shape])),0)

        if save_cache==True:                      
            #save meta dictionary, with ids
            meta_path = os.path.join(cache_data_dir,'all_meta.csv')
            meta_arr.to_csv(meta_path)
        
        #drop first col of ids
        accessionid_arr = meta_arr[self.pat_id_col]
        meta_arr=meta_arr.drop(self.pat_id_col,axis=1)

        #expand numpy arr and make values as torch
        im_arr = np.expand_dims(im_arr,axis=1)
        im_torch = torch.from_numpy(im_arr).float()

        #expand numpy arr and make values as torch
        orig_shape_arr = np.expand_dims(orig_shape_arr,axis=1)
        orig_shape_arr = torch.from_numpy(orig_shape_arr).float()

        annotation_torch = torch.from_numpy(annotation_arr).float()
        if len(annotation_torch.shape)==5:
            annotation_torch=torch.squeeze(annotation_torch,dim=1)

        #encoed meta/id data for network in cfg
        meta_data_restructured = self.meta_feat_structure(np.array(meta_arr))
        meta_data_restructured = np.expand_dims(meta_data_restructured,axis=1)
        meta_torch = torch.from_numpy(meta_data_restructured).float()

        id_arr = np.array(id_arr)
        id_arr = np.expand_dims(id_arr,axis=1)

        orig_shape_arr = np.array(orig_shape_arr)
        orig_shape_arr = np.expand_dims(orig_shape_arr,axis=1)

        accessionid_arr = np.array(accessionid_arr)
        accessionid_arr = np.expand_dims(accessionid_arr,axis=1)

        return im_torch, annotation_torch, meta_torch, id_arr, orig_shape_arr 

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        try:
            meta = self.meta[index]
        except:
            #means meta is empty
            meta = self.meta
        id = self.ids[index][0]
        orig_size = self.orig_img_shape[index][0]
        return x, y, meta, id, orig_size

    def __len__(self):
        return len(self.data)