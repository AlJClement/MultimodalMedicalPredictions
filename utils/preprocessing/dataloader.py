
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
from .augmentation import Augmentation
from pathlib import Path
import sys 
sys.path = [str(p) if isinstance(p, Path) else p for p in sys.path]

class dataloader(Dataset):
    def __init__(self, cfg, set, subset=None) -> None:
        #paths
        self.cfg=cfg
        self.partition_size_test = cfg.DATASET.PARTITION_SIZE_TEST
        self.debug = getattr(cfg.DATASET, 'DEBUG_DATALOADER', False)

        if subset != None:
            self.subset=subset
        else:
            self.subset=subset

        self.in_channels = int(cfg.MODEL.IN_CHANNELS)
        self.rgb = self.in_channels == 3
        
        self.dataset_name = cfg.INPUT_PATHS.DATASET_NAME
        self.partition_file_path = cfg.INPUT_PATHS.PARTITION
        self.img_dir = cfg.INPUT_PATHS.IMAGES
        self.label_dir = cfg.INPUT_PATHS.LABELS
        self.labels_dir_numbers = cfg.INPUT_PATHS.LABELS_NUMBERS
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
        self.num_out_channels = cfg.MODEL.OUT_CHANNELS

        self.sigma=cfg.DATASET.SIGMA
        self.cfg_combine_reviewers=cfg.DATASET.COMBINE_REVIEWERS
        
        self.pat_id_col = cfg.INPUT_PATHS.ID_COL
        self.metaimport = MetadataImport(cfg)
        self.metadata_csv, self.metadata_csv_classes = self.metaimport.load_csv()
        self.model_init = model_init(cfg)
        #get specific models/feature loaders
        self.meta_feat_structure = self.model_init.get_modelspecific_feature_structure()
        self.augmentation = Augmentation(self.cfg)
        self.preprocess_seq = self.augmentation.downsample_and_padd()
        self.label_number_frames = self._load_label_number_frames()
        self.testing_path_cache = self._load_testing_path_cache()

        self.set = set
        self.return_orig_img = self._should_return_orig_img()
        self.lazy_load = self.set != 'training'
        if self.lazy_load:
            self.data, self.target, self.landmarks, self.meta, self.ids, self.orig_img_shape, self.orig_im = self._build_lazy_dataset()
        else:
            self.data, self.target, self.landmarks, self.meta, self.ids, self.orig_img_shape,  self.orig_im= self.get_numpy_dataset() 

        if self.set != 'training':
            self.perform_aug = False
        else:
            self.perform_aug = cfg.DATASET.AUGMENTATION.APPLY
        self.train_aug_seq = self.augmentation.augmentation_fn() if self.perform_aug else None
    
        self.save_aug = cfg.DATASET.AUGMENTATION.SAVE
        if self.save_aug == True:
            self.aug_path = self.cache_dir+'/augmentations'
            if not os.path.exists(self.aug_path):
                os.makedirs(self.aug_path)
        return

    def _debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def _should_return_orig_img(self):
        if self.set == 'testing':
            return any([
                bool(getattr(self.cfg.TEST, "SAVE_TXT", False)),
                bool(getattr(self.cfg.TEST, "SAVE_HEATMAPS_LANDMARKS_IMG", False)),
                bool(getattr(self.cfg.TEST, "SAVE_IMG_LANG_PREDANDTRUE", False)),
                bool(getattr(self.cfg.TEST, "SAVE_HEATMAPS_ALONE", False)),
                bool(getattr(self.cfg.TEST, "SAVE_HEATMAPS_ASDCM", False)),
            ])

        if self.set == 'validation':
            return any([
                bool(getattr(self.cfg.TEST, "SAVE_TXT", False)),
                bool(getattr(self.cfg.TEST, "SAVE_HEATMAPS_ALONE", False)),
                bool(getattr(self.cfg.TRAIN, "SAVE_VAL_DCM", False)),
            ])

        return True

    def _load_label_number_frames(self):
        if self.labels_dir_numbers == [] or self.dataset_name != 'oai_nolandmarks':
            return {}

        frames = {}
        for csv_file in self.labels_dir_numbers:
            frames[csv_file] = pd.read_csv(csv_file, sep="|")
            if "ID" in frames[csv_file].columns:
                frames[csv_file]["ID"] = frames[csv_file]["ID"].astype(str)
        return frames

    def _load_testing_path_cache(self):
        cache = Path("./testing_path_cache.txt")
        if not cache.exists():
            return {}

        cache_map = {}
        for path in cache.read_text().splitlines():
            parts = Path(path).parts
            if len(parts) >= 3:
                cache_map[parts[-3]] = path
        return cache_map

    def _append_testing_path_cache(self, path):
        cache = Path("./testing_path_cache.txt")
        with cache.open("a") as f:
            f.write(path + "\n")

        parts = Path(path).parts
        if len(parts) >= 3:
            self.testing_path_cache[parts[-3]] = path

    def _get_oai_label_numbers(self, id):
        ann_list = []
        for csv_file, df in self.label_number_frames.items():
            df_pat = df.loc[df["ID"] == id]

            if 'cooke' in csv_file.lower():
                R_HKA = df_pat.loc[df_pat["SIDE"] == 1, "V01HKANGLE"].values
                L_HKA = df_pat.loc[df_pat["SIDE"] == 2, "V01HKANGLE"].values
            elif 'duryea' in csv_file.lower():
                R_HKA = df_pat.loc[df_pat["SIDE"] == '1: Right', "V01HKANGJD"].values
                L_HKA = df_pat.loc[df_pat["SIDE"] == '2: Left',  "V01HKANGJD"].values
            else:
                raise ValueError("dataset not implemented for label numbers")

            try:
                R_HKA = float(R_HKA[0])
            except (IndexError, ValueError, TypeError):
                R_HKA = np.nan

            try:
                L_HKA = float(L_HKA[0])
            except (IndexError, ValueError, TypeError):
                L_HKA = np.nan

            ann_list.append([R_HKA, L_HKA])

        return ann_list

    def _patient_id_from_path(self, img_path):
        if self.dataset_name == 'oai_nolandmarks':
            return img_path.strip("/").split("/")[-3]
        return img_path.split('/')[-1].split('.')[0]

    def _resolve_sample(self, img_path, ann_paths, pat_id):
        _im_arr, orig_shape, _im_orig = self.get_img(self.preprocess_seq, img_path)
        self._debug_print('orig_shape:', orig_shape)

        if self.dataset_name == 'oai_nolandmarks':
            _annotation_arr = ann_paths
            annotation_points = [0.0]
        else:
            _annotation_arr, annotation_points, _ = self.get_ann(ann_paths, self.preprocess_seq, _im_arr.shape, orig_shape)

        if self.num_out_channels != self.num_landmarks and self.dataset_name == 'oai':
            print('Num Landmarks does not match Num Channels')
            _annotation_arr = [_annotation_arr[0][[0,2,5,7,9,13],:,:]]
            annotation_points = [annotation_points[0][[0,2,5,7,9,13],:]]

        try:
            if _annotation_arr[0][0] == 0 and self.metadata_csv_classes is not None:
                _annotation_arr = self.metaimport._get_class_arr(self.metadata_csv_classes, pat_id)
        except Exception:
            pass

        return _im_arr, _annotation_arr, annotation_points, orig_shape, _im_orig

    def _build_lazy_dataset(self):
        img_files, annotation_files = self.get_partition()
        partition_key = self._resolve_partition_key(img_files)
        if self.subset != None:
            im_set = img_files[partition_key][:self.subset]
            ann_set = annotation_files[partition_key][:self.subset]
        else:
            im_set = img_files[partition_key]
            ann_set = annotation_files[partition_key]

        id_rows = [self._patient_id_from_path(img_path) for img_path in im_set]
        meta_rows = [self.metaimport._get_array(self.metadata_csv, pat_id) for pat_id in id_rows]
        meta_arr = pd.concat(meta_rows, ignore_index=True)
        meta_arr = meta_arr.drop(self.pat_id_col, axis=1)
        meta_data_restructured = self.meta_feat_structure(np.array(meta_arr))
        meta_data_restructured = np.expand_dims(meta_data_restructured, axis=1)
        meta_torch = torch.from_numpy(meta_data_restructured).float()

        placeholder = [None] * len(im_set)
        return list(im_set), list(ann_set), placeholder, meta_torch, np.array(id_rows), placeholder, placeholder

    def _resolve_partition_key(self, partition_dict):
        if self.set in partition_dict:
            return self.set

        if self.set == 'testing':
            fallback_keys = ['test', 'validation', 'val']
            for key in fallback_keys:
                if key in partition_dict:
                    self._debug_print(f"Falling back from 'testing' to '{key}' partition")
                    return key

        available = list(partition_dict.keys())
        raise KeyError(f"Partition '{self.set}' not found. Available partitions: {available}")

    def _find_oai_image_path(self, id):
        cached_path = self.testing_path_cache.get(id)
        if cached_path:
            return cached_path

        BASE = Path(self.img_dir)
        selected_path = None

        for d in BASE.rglob(id):
            if not d.is_dir():
                continue
            for img in d.rglob("*_1x1.jpg"):
                with Image.open(img) as im:
                    w, h = im.size
                    self._debug_print(im.size)
                    if w == 440 and h == 535:
                        self._debug_print(f"{img} {w}x{h}")
                        selected_path = img.as_posix()
                    if h > 2 * w:
                        self._debug_print(f"{img} {w}x{h}")
                        selected_path = img.as_posix()

        if selected_path is None:
            raise FileNotFoundError(f"Could not find cached testing image for {id}")

        self._append_testing_path_cache(selected_path)
        return selected_path

    
    def get_partition(self): ### make partition size not none if you want to sample
        # open partition
        if self.partition_file_path:
            partition_file = open(self.partition_file_path)
            partition_dict = json.load(partition_file)

            if self.labels_dir_numbers == []:
                pass
            else:
                if self.dataset_name != 'oai_nolandmarks':
                    raise ValueError('only oai implemented for label numbers')
                else:
                    ## remove anything in partition that is not in the label numbers files
                    # load label numbers file 1 and get all ids. drop anything that is not that id.
                    # load txt as dict
                    ids = []
                    with open(self.labels_dir_numbers[0]) as f:
                        header = f.readline().strip().split('|')
                        id_idx = next(i for i, h in enumerate(header) if h.lower() == "id")

                        for line in f:
                            if line.strip():
                                ids.append(line.strip().split('|')[id_idx])
                    ##get unique ids
                    unique_ids = [ids[0]] if ids else []
                    for num in ids[1:]:
                        if num != unique_ids[-1]:
                            unique_ids.append(num)

                    #partion dict items 
                    partition_ids = []

                    # loop through all partitions and extract first part before the dash
                    for partition_name, ids in partition_dict.items():
                        for full_id in ids:
                            first_number = full_id.split('-')[0]  # split by dash and take first part
                            partition_ids.append(first_number)
                    
                    filtered_unique_ids = [uid for uid in unique_ids if uid not in partition_ids]

                    filtered_unique_ids = filtered_unique_ids[:100]
                    
                    # filtered_unique_ids is your final list after removing IDs in id_list
                    partition_dict = {
                        "testing": filtered_unique_ids
                    }



            # get the file names of all images in the directory
            img_file_dic = {}
            annotation_dic = {}

            for key in partition_dict.keys():
                img_file_dic[key]=[]
                annotation_dic[key]=[]

            for i in partition_dict:
                if self.partition_size_test != None:
                    partition_dict[i] = partition_dict[i][:self.partition_size_test]
                else:
                    pass
                for id in partition_dict[i]:

                    if self.cfg_combine_reviewers==False:
                        #if you want to not combine the reviwers you need to get all possible paths
                        #get image file
                        #add annotation folders
                        ann_list = []
                        if self.label_dir=='':
                            ## checl if labels
                            if self.labels_dir_numbers == []:
                                #no labels so this will only be used for comparison of class
                                img_file_dic[i].append(os.path.join(self.img_dir, id + self.img_extension))       
                                label_folder = ''                 
                                annotation_dic[i].append([None])
                            else:
                                ## loop through labels files and get the values.
                                if self.dataset_name == 'oai_nolandmarks':
                                    ann_list = self._get_oai_label_numbers(id)
                                        
                                    if self.dataset_name == 'oai_nolandmarks':
                                        img = self._find_oai_image_path(id)
                                        self._debug_print(f"loaded already, {img}")
                                        img_file_dic[i].append(img)

                                    else:
                                        img_file_dic[i].append(os.path.join(self.img_dir, id + self.img_extension))       
                                    annotation_dic[i].append(ann_list)
                                else:   
                                    raise ValueError('only oai implemented for label numbers')


                        else:
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
        self._debug_print(img_path)
        '''loads images as arr'''
        image = None
        attempted_paths = [img_path, img_path[:-4] + '.png']

        if self.dataset_name == 'ddh':
            new_name = img_path.split('/')[-1][:-5] + img_path.split('/')[-1][-5:-4] + '.png'
            attempted_paths.append(img_path.rsplit('/', 1)[0] + '/' + new_name)

        if self.dataset_name == 'hand':
            parent = Path(img_path[:-4]).parent
            hand_matches = glob.glob(str(parent) + '/**/*/' + img_path.split('/')[-1])
            attempted_paths.extend(hand_matches)

        for candidate_path in attempted_paths:
            try:
                image = io.imread(candidate_path, as_gray=True)
                break
            except Exception:
                continue

        if image is None:
            raise FileNotFoundError(
                f"Could not load image for dataset '{self.dataset_name}'. "
                f"Original path: {img_path}. Tried: {attempted_paths}"
            )

        # Augment image
        image_resized = seq(image=image)
        image_rescaled = (image_resized - np.min(image_resized)) / np.ptp(image_resized)
        image_as_255 = img_as_ubyte(image_rescaled)
        
        return image_as_255, image.shape, image
    
    def get_landmarks(self, ann_path, seq, image_shape):
        # Get annotations
        self._debug_print(ann_path)
        try:
            kps_np_array = np.loadtxt(ann_path, usecols=(0, 1),delimiter=',', max_rows=self.num_landmarks)
            if self.dataset_name == 'oai':
                kps_np_array = np.loadtxt(ann_path, usecols=(0, 1),delimiter=',', max_rows=14)
                if self.num_out_channels == 6:
                    selected_indices = [0, 2, 5, 7, 9, 12]
                    kps_np_array = kps_np_array[selected_indices]



        except:
            ext = ann_path.split('/')[-2]+'.txt'
            ext = '.txt'
            kps_np_array = np.loadtxt(ann_path[:-4]+'_g'+ext, usecols=(0, 1),delimiter=',', max_rows=self.num_landmarks)
        
        self._debug_print('truth:',kps_np_array)
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
        channels = np.zeros([len(ann_points), img_shape[0], img_shape[1]])
        for i in range(len(ann_points)):
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
            if ann_path ==None:
                _np_ann_points = [0.0]
                _ann_array = [0.0]
                folder_name = None
                pass
            else:
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
        seq = self.preprocess_seq
        
        img_files, annotation_files = self.get_partition()
        meta_rows = []
        id_rows = []
        image_rows = []
        image_orig_rows = []
        orig_shape_rows = []
        annotation_rows = []
        landmark_rows = []


        self._debug_print('loading:', self.set)
        if self.subset != None:
            im_set=img_files[self.set][:self.subset]
        else:
            im_set=img_files[self.set]
        
        # if self.set == 'testing':
        #     im_set = sorted(im_set)

        for i in tqdm(range(len(im_set))):
            if self.dataset_name == 'oai_nolandmarks':
                pat_id = img_files[self.set][i].strip("/").split("/")[-3]
            else:
                pat_id = img_files[self.set][i].split('/')[-1].split('.')[0]
            
            ##LOAD IMAGES##
            # print(img_files[self.set][i])
            _im_arr, orig_shape, _im_orig = self.get_img(seq,img_files[self.set][i])
            self._debug_print('orig_shape:',orig_shape)
            
            ##LOAD ANNOTATIONS##
            if self.dataset_name == 'oai_nolandmarks':
                _annotation_arr = annotation_files[self.set][i]
                annotation_points = [0.0]
                folder_ls = [None]
            else:
                _annotation_arr, annotation_points, folder_ls = self.get_ann(annotation_files[self.set][i], seq, _im_arr.shape, orig_shape)
            

            if self.num_out_channels != self.num_landmarks:
                ### assume OAI drop only until 3
                if self.dataset_name == 'oai':
                    print('Num Landmarks does not match Num Channels')
                    _annotation_arr = [_annotation_arr[0][[0,2,5,7,9,13],:,:]]
                    annotation_points = [annotation_points[0][[0,2,5,7,9,13],:]]

            ##LOAD META##
            _meta_arr = self.metaimport._get_array(self.metadata_csv, pat_id)

            ##### if annotations == 0 store the class from meta file
            try:
                if _annotation_arr[0][0] == 0:
                    ## replace with class from config file metaclasses
                    _annotation_arr = self.metaimport._get_class_arr(self.metadata_csv_classes, pat_id)
            except:
                pass


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
                        _a = visuals('',self.pixel_size[0],self.cfg).channels_thresholded(_annotation_arr)
                        plt.imshow(_a)
                        plt.imsave(os.path.join(cache_data_dir,pat_id+'_gt_map'+folder_ls[i]+self.img_extension),_a)
                        plt.close()
                        i=i+1
                else:
                    _a = visuals('',self.pixel_size[0],self.cfg).channels_thresholded(_annotation_arr[0])
                    plt.imshow(_a)
                    plt.imsave(os.path.join(cache_data_dir,pat_id+'_gt_map'+self.img_extension),_a)
                    plt.close()

            
            id_rows.append(pat_id)
            meta_rows.append(_meta_arr)
            image_rows.append(_im_arr)
            image_orig_rows.append(_im_orig)
            orig_shape_rows.append(orig_shape)
            annotation_rows.append(_annotation_arr)
            landmark_rows.append(annotation_points)



        if save_cache==True:                      
            #save meta dictionary, with ids
            meta_path = os.path.join(cache_data_dir,'all_meta.csv')
            meta_arr = pd.concat(meta_rows, ignore_index=True)
            meta_arr.to_csv(meta_path)
        
        meta_arr = pd.concat(meta_rows, ignore_index=True)

        #drop first col of ids
        accessionid_arr = meta_arr[self.pat_id_col]
        meta_arr=meta_arr.drop(self.pat_id_col,axis=1)

        #expand numpy arr and make values as torch
        ### if landmarks dont == chanels its oai

        im_arr = np.expand_dims(np.stack(image_rows, axis=0),axis=1)
        im_torch = torch.from_numpy(im_arr).float()

        # im_orig_arrs =np.expand_dims(im_orig_arrs,axis=1)
        # im_orig_torch = torch.from_numpy(im_orig_arrs).float() 

        #expand numpy arr and make values as torch
        orig_shape_arr = np.expand_dims(np.stack(orig_shape_rows, axis=0),axis=1)
        orig_shape_arr = torch.from_numpy(orig_shape_arr).float()

        try:
            annotation_arr = np.stack(annotation_rows, axis=0)
            annotation_torch = torch.from_numpy(annotation_arr).float()
            if len(annotation_torch.shape)==5:
                annotation_torch=torch.squeeze(annotation_torch,dim=1)
        except:
            annotation_torch = annotation_rows


        #encoed meta/id data for network in cfg
        meta_data_restructured = self.meta_feat_structure(np.array(meta_arr))
        meta_data_restructured = np.expand_dims(meta_data_restructured,axis=1)
        meta_torch = torch.from_numpy(meta_data_restructured).float()


        orig_shape_arr = np.array(orig_shape_arr)
        orig_shape_arr = np.expand_dims(orig_shape_arr,axis=1)

        accessionid_arr = np.array(accessionid_arr)
        accessionid_arr = np.expand_dims(accessionid_arr,axis=1)

        #expand numpy arr and make values as torch
        landmark_arr = np.expand_dims(np.array(landmark_rows),axis=1)
        landmark_torch = torch.from_numpy(landmark_arr).float()

        return im_torch, annotation_torch, landmark_torch, meta_torch, np.array(id_rows), orig_shape_arr, image_orig_rows 

    def __getitem__(self, index):
        if self.lazy_load:
            img_path = self.data[index]
            ann_paths = self.target[index]
            pat_id = self.ids[index]
            _im_arr, _annotation_arr, annotation_points, orig_shape, _im_orig = self._resolve_sample(img_path, ann_paths, pat_id)

            x = torch.from_numpy(np.expand_dims(_im_arr, axis=0)).float()

            try:
                y = torch.from_numpy(np.asarray(_annotation_arr)).float()
                if y.ndim == 4 and y.shape[0] == 1:
                    y = y.squeeze(0)
            except Exception:
                y = _annotation_arr
                if isinstance(y, np.ndarray) and y.dtype == object:
                    y = y.flatten().tolist()

            landmarks = torch.from_numpy(np.expand_dims(np.array(annotation_points), axis=0)).float()
            meta = self.meta[index]
            orig_size = torch.tensor(np.expand_dims(np.asarray(orig_shape), axis=0), dtype=torch.float32)
            if self.return_orig_img:
                orig_img = torch.from_numpy(np.asarray(_im_orig))
            else:
                orig_img = torch.empty(0, dtype=torch.uint8)

            if self.in_channels > 1 and x.shape[0] == 1:
                x = x.repeat(self.in_channels, 1, 1)

            return x, y, landmarks, meta, pat_id, orig_size, orig_img

        x = self.data[index]
        y = self.target[index]
        
        #Check dtype if target is filled with classes
        if isinstance(y, np.ndarray) and y.dtype == object:
            # Convert to list of strings
            y = y.flatten().tolist()

        orig_img = self.orig_im[index]
        landmarks = self.landmarks[index]
        id = self.ids[index]

        if len(landmarks)==1: 
            pass
        else:
            raise ValueError('check num reviewers for augmentation')

        if self.perform_aug==True:
            aug_seq = self.train_aug_seq

            repeat = True
            counter = 0
            while repeat and counter < 1:
                counter += 1
                xx = x.cpu().detach().numpy().astype('uint8')
                _kps = landmarks.cpu().detach().numpy().squeeze(1)
                images = np.vstack((xx))
                
                kps = KeypointsOnImage.from_xy_array(_kps[0], shape=images.shape)
                aug_image, _aug_kps = aug_seq(images=images,keypoints=kps)
                
                aug_kps=_aug_kps.to_xy_array().reshape(self.num_out_channels, 2)
                aug_ann_array = self.add_sigma_channels(aug_kps, aug_image.shape)

            #convert back to torch friendly   
            x = torch.from_numpy(np.expand_dims(aug_image,axis=0)).float()
            y = torch.from_numpy(aug_ann_array).float()

            aug_landmarks = torch.from_numpy(np.expand_dims(aug_kps,axis=0)).float()

            landmarks = torch.from_numpy(np.expand_dims(aug_kps,axis=0)).float()
            if self.save_aug == True:
                visuals(self.aug_path+'/'+id,self.pixel_size[0], self.cfg).heatmaps(aug_image, aug_ann_array)

        try:
            meta = self.meta[index]
        except:
            #means meta is empty
            meta = self.meta

        orig_size = self.orig_img_shape[index][0]

        if self.in_channels > 1 and x.shape[0] == 1:
            x = x.repeat(self.in_channels, 1, 1)

        return x, y, landmarks, meta, id, orig_size, orig_img

    def __len__(self):
        return len(self.data)
