import matplotlib.pyplot as plt
import os
import datetime
import os
import tempfile
from PIL import Image
import numpy as np
import pydicom
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage

class visuals():
    def __init__(self, save_path, pixelsize, cfg, img_ext='.jpg') -> None:
        self.img_ext = img_ext
        self.save_path = save_path
        try:
            self.pixelsize = pixelsize.detach().cpu().numpy()
        except:
            self.pixelsize = pixelsize
        
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
        ds = pydicom.dcmread(dcm_loc)
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
    
    def heatmaps(self, image, output, target_points, predicted_points, w_landmarks=True, all_landmarks=True, as_dcm=False, dcm_loc='', save_high_res=True):
        fig, ax = plt.subplots(1, 1)
        image = image.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        predicted_points = predicted_points.detach().cpu().numpy()
        target_points = target_points.cpu().detach().numpy()
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

        if w_landmarks == True:
            ax.imshow(image, cmap='Greys_r')
            #add landmarks
            ax.scatter(target_points[:, 0]/self.pixelsize, target_points[:, 1]/self.pixelsize, color='lime', s=5)
            ax.scatter(predicted_points[:, 0]/self.pixelsize, predicted_points[:, 1]/self.pixelsize, color='red', s=5)
        else:
            ax.scatter(predicted_points[:, 0]/self.pixelsize, predicted_points[:, 1]/self.pixelsize, color='red', s=5)
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
