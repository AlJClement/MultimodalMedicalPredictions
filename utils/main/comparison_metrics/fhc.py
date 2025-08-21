import math
from .. import evaluation_helper
import numpy as np

class fhc():
    def __init__(self) -> None:
        pass
    def points_same_side(self, A, B, P1, P2):
        x1, y1 = A
        x2, y2 = B
        a = y2 - y1
        b = -(x2 - x1)
        c = a * x1 + b * y1

        def side(P):
            x, y = P
            val = a * x + b * y - c
            return 0 if val == 0 else (1 if val > 0 else -1)

        s1 = side(P1)
        s2 = side(P2)

        if s1 == 0 or s2 == 0:
            return False  # One (or both) point is exactly on the line
        return s1 == s2  # True if same side, False otherwise

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
        
        if self.points_same_side(il_1, il_2, fh_1, fh_2) == True:
            FHC = 0
            return FHC
        
        else:
            fhc_type='Vertical Distance'
            if fhc_type=='EuclideanDistance':
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
            else:
                i = np.mean([il_1[0],il_2[0]])
                D = fh_2[0]-fh_1[0]
                d = fh_2[0]-i

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
    
    def get_fhc_pred(self, pred, pred_map, pixelsize):
        fhc_pred = self.fhc(pred)
        print('pred fhc:',fhc_pred)

        ls_values = ['fhc pred', fhc_pred]
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
