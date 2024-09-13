import matplotlib.pyplot as plt
import os
import datetime
import os
import tempfile
from PIL import Image
import numpy as np
import pydicom
import imgaug.augmenters as iaa

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
        preprocessing_steps = [iaa.Resize({"width": w, "height": h}),]
        return iaa.Sequential(preprocessing_steps)


    def save_astxt(self, predicted_points, img_size):
        w=1024
        h=768
        print(img_size)
        upsample_ratio_x,upsample_ratio_y = img_size[0]*(img_size[0]/w), img_size[0]*(img_size[1]/h)
        print(upsample_ratio_x,upsample_ratio_y)
        # pred_points = self.upsample(pred_points)

        with open(self.save_path, 'a') as output:
            landmarks = predicted_points.detach().cpu().numpy()
            p = self.pixelsize#.detach().cpu().numpy()
            for i in range(len(landmarks)):
                row=landmarks[i]
                data_str = str(round(row[1]/p[1]*upsample_ratio_y,5))+","+str(round(row[0]/p[0]*upsample_ratio_x,5))
                output.write(data_str+"\n")
            print('saving', self.save_path)

        return
    
    def heatmaps(self, image, output, target_points, predicted_points, w_landmarks=True, all_landmarks=True, as_dcm=False, dcm_loc='', save_high_res=False):
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
            ax.scatter(target_points[:, 0], target_points[:, 1], color='lime', s=5)
            ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=5)
        else:
            ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=5)
            ax.imshow(image, cmap='Greys_r',alpha=0.4)

        if as_dcm == True:
            #put array into dcm version
            self.save_dcm_heatmap(_output, dcm_loc)
        else:
            plt.savefig(self.save_path,dpi=1200, bbox_inches='tight', pad_inches = 0)
            im = Image.open(self.save_path+'.png')
            rgb_im = im.convert('RGB')
            rgb_im.save(self.save_path+'.jpg')
            plt.close()

            #img_highres = self.downsample(rgb_im)
            if save_high_res == True:
                img_highres=rgb_im.resize((1024,768))
                img_highres.save(self.save_path+'_highres.jpg')

        plt.close()
        return
