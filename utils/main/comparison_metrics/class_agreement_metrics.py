import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import visualisations
from visualisations import *
import re

class class_agreement_metrics():
    def __init__(self, dataset_name, df_col, pred_col, true_col, outpath, loc='test'):
        #df col: column with ones and zeros defining equal or different classes
        self.gt_class_arr = df_col[true_col].to_numpy()
        self.pred_class_arr = df_col[pred_col].to_numpy()
        self.diagnosis_name = dataset_name
        self.loc = loc
        self.output_path = outpath
        pass

    def _get_metrics(self,group = False, groups=[('i'),('ii','iii/iv')]):
        #check if group == true, if it does then the defined 'groups will group classes'
        # for example in ddh we group 1 vs 2/3/4 and 1/2 vd 3/4. 
        pred_class_arr = self.pred_class_arr 
        gt_class_arr = self.gt_class_arr
        if group == True:
            #loop in the groups and combine the classes in botht he pred cols and true cols as the new name combined.
            for g in groups:
                if len(g)==1:
                    pass
                else:
                    #print('combing classes')
                    i=0
                    for c in g:
                        if i == 0:
                            new_class_str = c
                            i = i +1
                        else:
                            new_class_str = new_class_str + '&'+c

                    for i in range(len(g)):
                        #replace with ints so then you can replace with string and overlaping strings wont cause doubles#
                        c=g[i]
                        pred_class_arr = ['new_class' if x==c else x for x in pred_class_arr]
                        gt_class_arr =  ['new_class' if x==c else x for x in gt_class_arr]

                        
                    for i in range(len(g)):
                        c=g[i]
                        pred_class_arr=[new_class_str if x=='new_class' else x for x in pred_class_arr]
                        gt_class_arr=[new_class_str if x=='new_class' else x for x in gt_class_arr]
        else:
            #use given pred and true cols with as many classes as exist                   pred_class_arr=list(map(lambda x: x.replace(list(g), new_class_str), _pred_class_arr))


            pass 

        #check if multi class
        if np.unique(gt_class_arr).size <= 2:
            if np.all(pred_class_arr == gt_class_arr) == True:
                #then all classes are the same, confusion matrix will give back one value
                #if only one value figure out if its true neg or true positive
                x = confusion_matrix(gt_class_arr, pred_class_arr).ravel()

                #get one value from gt_class_arr and figure out if its the first or second class. if its the first in the list its tp if second tn
                if gt_class_arr[0]==groups[0]:
                    tn = float(x[0])
                    fp, fn, tp = 0.0,0.0,0.0
                else:
                    tp = float(x[0])
                    tn, fp, fn = 0.0,0.0,0.0
            else:
                #only one class for classification problem
                tn, fp, fn, tp = confusion_matrix(gt_class_arr, pred_class_arr).ravel()

            total = tn + fp + fn + tp
            if total == 0:
                accuracy = 0.0
            else:
                accuracy = 100 * float(tn + tp) / float(total)
            if tp+fp == 0:#
                precision = 0.0
            else:
                precision = 100 * float(tp) / float(tp + fp)
            if tp+fn == 0:
                recall = 0.0
                pass
            else:
                recall = 100 * float(tp) / float(tp + fn)

            if tn+fp == 0:
                specificity = 0.0
                pass
            else:
                specificity = 100 * float(tn) / float(tn + fp)
            
            if tp +fn == 0:
                sensitivity = 0.0
                pass
            else:
                sensitivity = 100 * float(tp) / float(tp + fn)
        else:
            #multi class so outputs will be an array for tn, fp, fn, tp    
            classes = set(gt_class_arr)
            if self.diagnosis_name=='ddh':
                classes = ['i','ii','iii/iv']
            
            confusion_matrix_multiclasses = multilabel_confusion_matrix(gt_class_arr, pred_class_arr)#, labels=classes)

            ##plot confusion matrix
            visualisations.comparison(self.diagnosis_name,self.output_path).confusion_matrix_multiclass(classes, confusion_matrix_multiclasses, self.loc, name =str(groups[0]) +' vs '+str(groups[1]))

            #find values for all
            accuracy = np.array([])
            precision = np.array([])
            recall =  np.array([])
            total = np.array([])
            sensitivity = np.array([])
            specificity = np.array([])
            tn, fp, fn, tp = np.array([]),np.array([]),np.array([]),np.array([])

            for _class in confusion_matrix_multiclasses:
                _tn, _fp, _fn, _tp = _class.ravel()
                _total = _tn + _fp + _fn + _tp
                if _total == 0:
                    _accuracy = 0.0
                else:
                    _accuracy = 100 * float(_tn + _tp) / float(_total)

                if _tp+_fp == 0:#
                    _precision = 0.0
                else:
                    _precision = 100 * float(_tp) / float(_tp + _fp)

                if _tp+_fn == 0:
                    _recall = 0.0
                    pass
                else:
                    _recall = 100 * float(_tp) / float(_tp + _fn)
                
                if _tn+_fp == 0:
                    _specificity = 0.0
                    pass
                else:
                    _specificity = 100 * float(_tn) / float(_tn + _fp)
                
                if _tp + _fn == 0:
                    _sensitivity = 0.0
                    pass
                else:
                    _sensitivity = 100 * float(_tp) / float(_tp + _fn)
                    
                precision = np.append(precision, _precision)
                accuracy = np.append(accuracy, _accuracy)
                recall = np.append(recall, _recall)
                total =  np.append(total, _total)
                specificity = np.append(specificity, _specificity)
                sensitivity =  np.append(sensitivity, _sensitivity)
                
                tn, fp, fn, tp = np.append(tn,_tn),np.append(fp,_fp), np.append(fn,_fn),np.append(tn,_tp)


        ls = [['TN:', tn],
              ['FP:', fp], 
              ['FN:', fn],
              ['TP:', tp],
              ['percision: ',precision],
              ['recall: ', recall],
              ['accuracy', accuracy],
              ['sensitivity', sensitivity],
              ['specificity', specificity]]
        print(ls)
        
        metric_str = ''
        for i in ls:
            try:
                metric_str = metric_str+i[0]+' '+np.array2string(i[1])+', '
            except:
                #if theres only one output
                metric_str = metric_str+i[0]+' '+str(i[1])+', '

        metric_str=metric_str[:-2]

        # ls = [['percision: ',round(precision[0],2),round(precision[1],2)],
        #       ['recall: ',round(recall[0],2),round(recall[1],2)],
        #       ['accuracy: ',round(accuracy[0],2),round(accuracy[1],2)],
        #       ['sensitivity: ',round(sensitivity[0],2),round(sensitivity[1],2)],
        #       ['specificity: ',round(specificity[0],2),round(specificity[1],2)]]
                
        return ls
