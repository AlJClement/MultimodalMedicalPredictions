import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import visualisations
from visualisations import *

class class_agreement_metrics():
    def __init__(self, dataset_name, df_col, pred_col, true_col, loc='test'):
        #df col: column with ones and zeros defining equal or different classes
        self.gt_class_arr = df_col[true_col].to_numpy()
        self.pred_class_arr = df_col[pred_col].to_numpy()
        self.diagnosis_name = dataset_name
        self.loc = loc
        pass

    def _get_metrics(self):
        #check if multi class

        if np.unique(self.pred_class_arr).size < 2:
            if np.all(self.pred_class_arr == self.gt_class_arr) == True:
                tn, fp, fn, tp = 0,0,0,0
            elif np.unique(self.pred_class_arr).size == 1:
                tn, fp, fn, tp = 0,0,0,0
            else:
                #only one class for classification problem
                tn, fp, fn, tp = confusion_matrix(self.gt_class_arr, self.pred_class_arr).ravel()
            total = tn + fp + fn + tp
            if total == 0:
                accuracy = 0
            else:
                accuracy = 100 * float(tn + tp) / float(total)
            if tp+fp == 0:#
                precision = 0
            else:
                precision = 100 * float(tp) / float(tp + fp)
            if tp+fn == 0:
                recall = 0
                pass
            else:
                recall = 100 * float(tp) / float(tp + fn)
                recall = 100 * float(tp) / float(tp + fn)

        else:
            #multi class so outputs will be an array for tn, fp, fn, tp    
            classes = set(self.gt_class_arr)
            if self.diagnosis_name=='ddh':
                classes = ['i','ii','iii/iv']
            
            confusion_matrix_multiclasses = multilabel_confusion_matrix(self.gt_class_arr, self.pred_class_arr)#, labels=classes)

            ##plot confusion matrix
            visualisations.comparison(self.diagnosis_name).confusion_matrix_multiclass(classes, confusion_matrix_multiclasses, self.loc)

            #find values for all
            accuracy = np.array([])
            precision = np.array([])
            recall =  np.array([])
            total = np.array([])
            tn, fp, fn, tp = np.array([]),np.array([]),np.array([]),np.array([])

            for _class in confusion_matrix_multiclasses:
                _tn, _fp, _fn, _tp = _class.ravel()
                _total = _tn + _fp + _fn + _tp
                if _total == 0:
                    _accuracy = 0
                else:
                    _accuracy = 100 * float(_tn + _tp) / float(_total)

                if _tp+_fp == 0:#
                    _precision = 0
                else:
                    _precision = 100 * float(_tp) / float(_tp + _fp)
                if _tp+_fn == 0:
                    _recall = 0
                    pass
                else:
                    _recall = 100 * float(_tp) / float(_tp + _fn)
                
                precision = np.append(precision, _precision)
                accuracy = np.append(accuracy, _accuracy)
                recall = np.append(recall, _recall)
                total =  np.append(total, _total)
                
                tn, fp, fn, tp = np.append(tn,_tn),np.append(fp,_fp), np.append(fn,_fn),np.append(tn,_tp)

        ls = [['TN:', tn],
              ['FP:', fp], 
              ['FN:', fn],
              ['TP:', tp],
              ['percision: ',precision],
              ['recall: ', recall],
              ['accuracy', accuracy]]
        
        metric_str = ''
        for i in ls:
            try:
                metric_str = metric_str+i[0]+' '+np.array2string(i[1])+', '
            except:
                #if theres only one output
                metric_str = metric_str+i[0]+' '+str(i[1])+', '

        metric_str=metric_str[:-2]

        return metric_str
