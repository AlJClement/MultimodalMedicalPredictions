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
from main.comparison_metrics import fhc, graf_angle_calc


class visuals():
    def __init__(self, save_path=None, pixelsize=None, cfg=None, orig_size=None,img_ext='.jpg') -> None:
        self.cfg=cfg
        self.img_ext = img_ext
        self.save_path = save_path
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
    
    def upsample(self, w=1024, h=768):
        #default is ddh
                # Define how to downsample and pad images
        preprocessing_steps = [
            # iaa.Crop(px=1),
            # iaa.PadToAspectRatio(w/h, position='right-bottom', pad_mode='edge'),
            iaa.Resize({"width": w, "height": h}),
        ]

        return iaa.Sequential(preprocessing_steps)

    def save_np(self, prediction):
        prediction = prediction.detach().cpu().numpy()
        print(prediction.shape)
        np.save(self.save_path, prediction)
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
            image = image.detach().cpu().numpy()
            output = output.detach().cpu().numpy()
        except:
            pass

        if target_points == None:
            pass
        else:
            predicted_points = predicted_points.detach().cpu().numpy()
            target_points = target_points.cpu().detach().numpy()
        
        if resize_to_orig == False:
            predicted_points[:, 1] *= (image.shape[0]/self.orig_size[0][1])
            predicted_points[:, 0] *= (image.shape[1]/self.orig_size[0][0])
            target_points[:, 1] *= (image.shape[0]/self.orig_size[0][1])
            target_points[:, 0] *= (image.shape[1]/self.orig_size[0][0])
            image = Augmentation(self.cfg).upsample(self.orig_size, image)
            output = Augmentation(self.cfg).upsample(self.orig_size, output)


        #print(self.pixelsize)
        #comment out if you want 5 landmarks plotted#
            if all_landmarks==True:
                pass
            else:
                output = output[:4]
                predicted_points = predicted_points[:4]
                target_points=target_points[:4]

        _output = self.channels_thresholded(output)
        ax.imshow(_output, cmap='inferno', alpha = 1)
        ax.axis('off')

        # ax.axis('off')
        # try:
        #     if target_points is None:
        #         tp_exist = None
        # except:
        #     tp_exist = True
            
        if target_points is None:
            ax.imshow(image, cmap='Greys_r',alpha=0.4)
        else:
            if with_img == True:
                if w_landmarks == True:
                    ax.imshow(image, cmap='Greys_r')
                    #add landmarks
                    print('adding landmarks')
                    ax.scatter(target_points[:, 0], target_points[:, 1], color='lime', s=5)
                    ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=5)
                                        
                    fhc_pred, fhc_true = fhc.fhc().get_fhc(predicted_points,output,target_points,image,self.pixelsize)
                    fhc_pred, fhc_true = fhc_pred[1]*100, fhc_true[1]*100
                    alpha_true, alpha_pred = round(graf_angle_calc().calculate_alpha(target_points),1), round(graf_angle_calc().calculate_alpha(predicted_points),1)

                    ax.text(0.02, 0.98,f"FHC = {fhc_true:.1f}%\n α = {alpha_true:.1f}°", 
                                transform=ax.transAxes, fontsize=10, verticalalignment='top',bbox=dict(facecolor='green', alpha=0.6, edgecolor='none'))
                    ax.text(0.02, 0.80,f"FHC = {fhc_pred:.1f}%\n α = {alpha_pred:.1f}°",
                                transform=ax.transAxes, fontsize=10, verticalalignment='top',bbox=dict(facecolor='red', alpha=0.6, edgecolor='none'))
                    
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
            plt.savefig(self.save_path,dpi=1200, bbox_inches='tight', pad_inches = 0)
            im = Image.open(self.save_path+'.png')
            rgb_im = im.convert('RGB')
            rgb_im.save(self.save_path+'.jpg')
            plt.close()

        plt.close()
        return
