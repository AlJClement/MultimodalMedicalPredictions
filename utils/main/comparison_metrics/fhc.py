import math
from .. import evaluation_helper

class fhc():
    def __init__(self) -> None:
        pass
    def fhc(self,tensor_landmarks):
        labeltxts=tensor_landmarks.tolist()
        il_1 = labeltxts[0]
        il_2 = labeltxts[1]
        fh_1 = labeltxts[5]
        fh_2 = labeltxts[6]

        fhc_dist = math.dist(fh_1, fh_2)

        x=1
        y=0

        inter_x = (fh_1[x]+fh_2[x])/2
        inter_y = (il_1[y]+il_2[y])/2

        inter_dist = math.dist(fh_2, [inter_y, inter_x])

        try:
            FHC = inter_dist/fhc_dist
        except:
            FHC = 0

        return FHC
    
    def get_fhc(self, pred, pred_map, true, true_map, pixelsize):
        fhc_pred = self.fhc(pred)
        fhc_true = self.fhc(true)
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
