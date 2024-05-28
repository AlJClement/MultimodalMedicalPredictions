import math


class fhc():
    def __init__(self) -> None:
        pass
    def fhc(self,tensor_landmarks):
        labeltxts=tensor_landmarks.fliplr().tolist()
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

        FHC = inter_dist/fhc_dist

        return FHC
    
    def get_fhc(self, pred, pred_map, true, true_map, pixelsize):
        fhc_pred = self.fhc(pred)
        fhc_true = self.fhc(true)
        ls_values = [['fhc pred', fhc_pred],
                    ['fhc true', fhc_true]]
        return ls_values 
