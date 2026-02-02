import matplotlib.pyplot as plt
import os
import datetime
import os
import tempfile
from PIL import Image
import numpy as np
# import pydicom
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage
import sys
import os
import pathlib
target_path = pathlib.Path(os.path.abspath(__file__)).parents[1]
sys.path.append(target_path)
from preprocessing.augmentation import Augmentation
import sys
sys.path.append("..")
from main.comparison_metrics import fhc, graf_angle_calc, protractor_hka

from scipy.ndimage import zoom

class visuals():
    def __init__(self, save_path=None, pixelsize=None, cfg=None, orig_size=None,img_ext='.jpg') -> None:
        self.cfg=cfg
        self.img_ext = img_ext
        self.save_path = save_path
        self.dataset_name = cfg.INPUT_PATHS.DATASET_NAME
    
        try:
            self.pixelsize = pixelsize.detach().cpu().numpy()
        except:
            self.pixelsize = pixelsize
        if orig_size is not None:
            self.orig_size = orig_size.detach().cpu().numpy()
        pass
    
    def channels_thresholded(self, output):
        #theshold then add all channels together
        for c in range(output.shape[0]):
            try:
                compressed_channels = compressed_channels+output[c]
            except:
                compressed_channels = output[c]
        return compressed_channels

    def save_dcm_heatmap(self, output_heatmap, dcm_loc):
        ds = '' #pydicom.dcmread(dcm_loc)
        read_arr = self.save_path.rsplit('/',2)[0]+'/'+self.save_path.split('/')[-1]+self.img_ext

        im_frame = Image.open(read_arr)
        np_frame = np.array(im_frame.getdata())
        ds.PixelData = np_frame
        save_dcm_path = self.save_path.rsplit('/',1)[0]+'/'+self.save_path.split('/')[-1]+'.dcm'
        ds.save_as(save_dcm_path)
        return
    
    # def upsample(self, w=1024, h=768):
    #     #default is ddh
    #             # Define how to downsample and pad images
    #     preprocessing_steps = [
    #         # iaa.Crop(px=1),
    #         # iaa.PadToAspectRatio(w/h, position='right-bottom', pad_mode='edge'),
    #         iaa.Resize({"width": w, "height": h}),
    #     ]

    #     return iaa.Sequential(preprocessing_steps)

    def save_np(self, prediction):
        try:
            prediction = prediction.detach().cpu().numpy()
        except:
            pass
        print(prediction.shape)
        np.save(self.save_path, prediction)

        # #orig size
        # zoom_factors = (1, self.orig_size[0][0] / prediction.shape[1], self.orig_size[0][1] / prediction.shape[2])

        # # Resize
        # orig_size_pred = zoom(prediction, zoom=zoom_factors, order=1)  # bilinear interpolation

        # np.save(self.save_path+'_origsize', orig_size_pred)

        return

    def save_astxt(self, img, predicted_points, img_size, orig_size):
        image = img.detach().cpu().numpy()
        predicted_points = predicted_points.detach().cpu().numpy()

        with open(self.save_path+'.txt', 'a') as output:
                for i in range(len(predicted_points)):
                    row = predicted_points[i]/self.pixelsize
                    # resize = image.shape[0]/image.shape
                    data_str = str(round(row[1],5))+","+str(round(row[0],5))
                    output.write(data_str+"\n")
                    print(data_str)

        print('saving', self.save_path)
        
        fig, ax = plt.subplots(1, 1)
        ax.imshow(image, cmap='Greys_r')
        ax.axis('off')

        plt.savefig(self.save_path+'.png',dpi=1200, bbox_inches='tight', pad_inches = 0)
        im = Image.open(self.save_path+'.png')
        rgb_im = im.convert('RGB')
        # rgb_im.save(self.save_path+'.jpg')
        plt.close()

        img_highres=rgb_im.resize((img_size[0],img_size[1]))
        img_highres.save(self.save_path+'.jpg')

        return
    
    def heatmaps(self, image, output, target_points=None, predicted_points=None, w_landmarks=True, all_landmarks=True, with_img = True, as_dcm=False, dcm_loc='', resize_to_orig=True,save_high_res=True):
        fig, ax = plt.subplots(1, 1)
        
        try:
            output = output.detach().cpu().numpy()
        except:
            pass
        try:
            image = image.detach().cpu().numpy()
        except:
            pass

        if target_points == None:
            pass
        else:
            predicted_points = predicted_points.detach().cpu().numpy()
            try:
                target_points = target_points.cpu().detach().numpy()
            except:
                target_points = target_points

    

        #print(self.pixelsize)
        #comment out if you want 5 landmarks plotted#
        if all_landmarks==True:
            pass
        else:
            output = output[:4]
            predicted_points = predicted_points[:4]
            target_points=target_points[:4]
        
        if len(output.shape)==4:
            output = output.squeeze(0)
        _output = self.channels_thresholded(output)
        ax.imshow(_output, cmap='inferno', alpha = 0.5)
        ax.axis('off')
        if image is None:
            pass
        else:
            ax.imshow(image, cmap='grey', alpha = 0.5)

        # ax.axis('off')
        # try:
        #     if target_points is None:
        #         tp_exist = None
        # except:
        #     tp_exist = True

        if target_points is None:
            pass
        else:
            if isinstance(target_points[0], tuple):
                ax.imshow(image, cmap='Greys_r',alpha=0.4)
                ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=5)
                                                        
                fhc_pred = fhc.fhc().get_fhc_pred(predicted_points,output, self.pixelsize)
                fhc_pred = fhc_pred[1]*100
                alpha_pred =round(graf_angle_calc().calculate_alpha(predicted_points),1)

                ##for setup from retuve
                fhc_true = target_points[3][0]
                alpha_true =target_points[2][0]

                ax.text(0.02, 0.98,f"FHC = {fhc_true}%\n α = {alpha_true}°", 
                            transform=ax.transAxes, fontsize=10, verticalalignment='top',bbox=dict(facecolor='green', alpha=0.6, edgecolor='none'))
                ax.text(0.02, 0.80,f"FHC = {fhc_pred:.1f}%\n α = {alpha_pred:.1f}°",
                            transform=ax.transAxes, fontsize=10, verticalalignment='top',bbox=dict(facecolor='red', alpha=0.6, edgecolor='none'))

            else:
                if with_img == True:
                    if w_landmarks == True:
                        ax.imshow(image, cmap='Greys_r')
                        #add landmarks
                        print('adding landmarks')
                        if self.dataset_name == 'oai_nolandmarks':
                            pass
                        else:
                            ax.scatter(target_points[:, 0], target_points[:, 1], color='lime', s=5)
                        ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=5)

                        ##
                        if self.cfg.INPUT_PATHS.DATASET_NAME == 'ddh':                  
                            fhc_pred, fhc_true = fhc.fhc().get_fhc(predicted_points,output,target_points,image,self.pixelsize)
                            fhc_pred, fhc_true = fhc_pred[1]*100, fhc_true[1]*100
                            alpha_true, alpha_pred = round(graf_angle_calc().calculate_alpha(target_points),1), round(graf_angle_calc().calculate_alpha(predicted_points),1)
                            print('pred alpha: ', alpha_pred)
                            print('true alpha: ', alpha_true)

                            ax.text(0.02, 0.98,f"FHC = {fhc_true:.1f}%\n α = {alpha_true:.1f}°", 
                                        transform=ax.transAxes, fontsize=10, verticalalignment='top',bbox=dict(facecolor='green', alpha=0.6, edgecolor='none'))
                            ax.text(0.02, 0.80,f"FHC = {fhc_pred:.1f}%\n α = {alpha_pred:.1f}°",
                                        transform=ax.transAxes, fontsize=10, verticalalignment='top',bbox=dict(facecolor='red', alpha=0.6, edgecolor='none'))
                        
                        elif self.cfg.INPUT_PATHS.DATASET_NAME == 'oai_nolandmarks':
                            L_hka_true_1, R_hka_true_1 = target_points[0][1], target_points[0][0]
                            L_hka_true_2, R_hka_true_2 = target_points[1][1], target_points[1][0]

                            L_hka_pred, R_hka_pred = protractor_hka.protractor_hka().hka_angles(predicted_points,output,target_points,image,self.pixelsize, HKA_only = True)

                            ax.text(-0.6, 0.98,f"L_hka_1 = {L_hka_true_1:.1f}°\n R_hka_1 = {R_hka_true_1:.1f}°", 
                                        transform=ax.transAxes, fontsize=5, verticalalignment='top',bbox=dict(facecolor='green', alpha=0.6, edgecolor='none'))
                            ax.text(-0.6, 0.9,f"L_hka_2 = {L_hka_true_2:.1f}°\n R_hka_2 = {R_hka_true_2:.1f}°", 
                                        transform=ax.transAxes, fontsize=5, verticalalignment='top',bbox=dict(facecolor='green', alpha=0.6, edgecolor='none'))
                           
                            ax.text(-0.6, 0.82,f"L_hka = {L_hka_pred:.1f}°\n R_hka = {R_hka_pred:.1f}°",
                                    transform=ax.transAxes, fontsize=5, verticalalignment='top',bbox=dict(facecolor='red', alpha=0.6, edgecolor='none'))
                
                        elif self.cfg.INPUT_PATHS.DATASET_NAME == 'oai':
                            L_hka_pred, R_hka_pred  = protractor_hka.protractor_hka().hka_angles(predicted_points,output,target_points,image,self.pixelsize, HKA_only = True)
                            L_hka_true, R_hka_true  = protractor_hka.protractor_hka().hka_angles(target_points,image,predicted_points,output,self.pixelsize, HKA_only = True)

                            ax.text(0.02, 0.98,f"L_hka = {L_hka_true:.1f}°\n R_hka = {R_hka_true:.1f}°", 
                                        transform=ax.transAxes, fontsize=5, verticalalignment='top',bbox=dict(facecolor='green', alpha=0.6, edgecolor='none'))
                            ax.text(0.02, 0.9,f"L_hka = {L_hka_pred:.1f}°\n R_hka = {R_hka_pred:.1f}°",
                                    transform=ax.transAxes, fontsize=5, verticalalignment='top',bbox=dict(facecolor='red', alpha=0.6, edgecolor='none'))


                        else:
                            pass
                    else:
                        ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=5)
                        ax.imshow(image, cmap='Greys_r',alpha=0.4)

        # with open(self.save_path+'.txt', 'a') as output:
        #     for i in range(len(predicted_points)):
        #         row = predicted_points[i]/self.pixelsize
        #         resize = image.shape[0]/image.shape
        #         data_str = str(round(row[1],5))+","+str(round(row[0],5))
        #         output.write(data_str+"\n")
        #         print(data_str)

        #     print('saving', self.save_path)

        if as_dcm == True:
            #put array into dcm version
            self.save_dcm_heatmap(_output, dcm_loc)
        else:
            ## make subfolders for saving easier debugging
            ##check if alpah or fhc do not match and make a copy in another folder
            if 'fhc_true' in locals():
                print(fhc_true, alpha_true)
                try:
                    fhc_true = 'n' if fhc_true >= 50 else 'a'
                    alpha_true = 'n' if alpha_true >= 60 else 'a'
                except:
                    fhc_true = 'n' if fhc_true == '>0.50' else 'a'
                    alpha_true = 'n' if alpha_true == '>60' else 'a'

                fhc_pred = 'n' if fhc_pred >= 50 else 'a'
                alpha_pred = 'n' if alpha_pred >= 60 else 'a'

                data_subset = self.save_path.split('/')[-2]

                if fhc_true == fhc_pred and alpha_pred==alpha_true:
                    ##save in main folder :)
                    save_img_path =self.save_path.split(data_subset)[-2]+ data_subset+'/correct'
                    pass
                elif fhc_true != fhc_pred and alpha_pred==alpha_true:
                    save_img_path =self.save_path.split(data_subset)[-2]+ data_subset+'/wrong_class_fhc'
                elif fhc_true == fhc_pred and alpha_pred!=alpha_true:
                    save_img_path = self.save_path.split(data_subset)[-2]+ data_subset+'/wrong_class_graf'
                else:
                    save_img_path = self.save_path.split(data_subset)[-2]+ data_subset+'/wrong_classes'
                
                os.makedirs(save_img_path, exist_ok=True)
                plt.savefig(save_img_path+self.save_path.split(data_subset)[-1]+'.png',dpi=1200, bbox_inches='tight', pad_inches = 0)
            
            else:
                save_img_path=self.save_path
                ## if oai check
                if self.cfg.INPUT_PATHS.DATASET_NAME == 'oai_nolandmarks':
                    # check rows are in increasing order of landmark
                    check_order = 0 
                    if not np.all(np.diff(predicted_points[:3, 1]) > 0):
                        check_order = 1
                    if not np.all(np.diff(predicted_points[3:, 1]) > 0):
                        check_order = 1
                    if check_order == 1:
                        save_img_path = save_img_path.replace('test', 'test/order_error')
                        os.makedirs(save_img_path.rsplit("/", 1)[0], exist_ok=True)
                    
                    if ('L_hka_pred' in locals()) == False:
                        L_hka_pred, R_hka_pred = protractor_hka.protractor_hka().hka_angles(predicted_points,output,target_points,image,self.pixelsize, HKA_only = True)

                    if abs(L_hka_pred) > 15 or abs(R_hka_pred) > 15:
                        save_img_path = save_img_path.replace('test', 'test/angle_outlier')
                        os.makedirs(save_img_path.rsplit("/", 1)[0], exist_ok=True)

                plt.savefig(save_img_path+'.png',dpi=1200, bbox_inches='tight', pad_inches = 0)

            #

            # im = Image.open(self.save_path+'.png')
            # rgb_im = im.convert('RGB')
            # rgb_im.save(self.save_path+'.jpg')


            plt.close()

        plt.close()
        return
