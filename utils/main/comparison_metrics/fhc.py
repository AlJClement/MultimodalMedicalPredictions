import math
from .. import evaluation_helper
import numpy as np

class fhc():
    def __init__(self) -> None:
        pass
    def fhc(self,landmarks, flip = False):
        if flip == True:
            try:
                #handle for tensor
                labeltxts=landmarks.fliplr().tolist()
            except:
                labeltxts=np.flip(landmarks)
        else:
            labeltxts=landmarks.tolist()

        il_1 = labeltxts[0]
        il_2 = labeltxts[1]
        fh_1 = labeltxts[5]
        fh_2 = labeltxts[6]

        D = math.dist(fh_1, fh_2)

        x=1
        y=0

        if il_2[1]-il_1[1] == 0:
            il_1[1]=il_1[1]+0.0001
        m1 = (il_2[0]-il_1[0])/(il_2[1]-il_1[1])
        b1 = il_2[0]-m1*(il_2[1])

        if fh_2[1]-fh_1[1] == 0:
            fh_1[1]=fh_1[1]+0.0001
        m2 = (fh_2[0]-fh_1[0])/(fh_2[1]-fh_1[1])
        b2 = fh_2[0]-m2*(fh_2[1])


        ## Distance d 
        try:
            xi = (b1 - b2) / (m2 - m1)
        except: 
            xi =0
        yi = m1 * xi + b1

        d = math.dist(fh_2, [yi, xi])

        try:
            FHC = (d/D)
        except:
            FHC = 0

        if FHC<=0: FHC=0
        if FHC>=1: FHC=1

        return FHC
    
    
    def get_fhc(self, pred, pred_map, true, true_map, pixelsize):
        fhc_pred = self.fhc(pred)
        print('pred fhc:',fhc_pred)

        fhc_true = self.fhc(true)
        print('true fhc:',fhc_true)

        ls_values = [['fhc pred', fhc_pred],
                    ['fhc true', fhc_true]]
        return ls_values 
    
    def get_fhc_batches(self,pred,target,pixel_size):
        target_points, predicted_points = evaluation_helper.evaluation_helper().get_landmarks(pred, target, pixels_sizes=pixel_size)
        pred_fhc, target_fhc = [],[]

        for i in range(pred.shape[0]):
            #calculate alpha and class for each pred and target in the batch
            t_fhc= self.fhc(target_points[i])
            p_fhc= self.fhc(predicted_points[i])
            ##add to output arr
            pred_fhc.append(p_fhc)
            target_fhc.append(t_fhc)

        return pred_fhc,target_fhc
