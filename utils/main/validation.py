import torch.nn as nn
import torch
from torch.autograd import Variable
import datetime
from .model_init import model_init
from .loss import *
from .evaluation_helper import *
import sys
import os
import pathlib
target_path = pathlib.Path(os.path.abspath(__file__)).parents[1]
sys.path.append(target_path)
import visualisations
from visualisations import visuals
from .evaluation_helper import evaluation_helper
import pandas as pd
from .comparison_metrics import *

class validation():
    def __init__(self, cfg, logger, net, l2_reg=True, save_img=True):
        self.dataset_name = cfg.INPUT_PATHS.DATASET_NAME
        self.dataset_type = cfg.DATASET.ANNOTATION_TYPE
        self.num_landmarks = cfg.DATASET.NUM_LANDMARKS
        self.max_epochs = cfg.TRAIN.EPOCHS

        self.net = net
        self.logger = logger
        self.l2_reg=l2_reg
        if l2_reg == True:
            self.loss_func = eval('L2RegLoss(cfg.TRAIN.LOSS)')
        else:
            self.loss_func = eval(cfg.TRAIN.LOSS)
        
        if (cfg.TRAIN.LOSS).split('_')[-1]=='wclass':
            self.add_class_loss = True
        else:
            self.add_class_loss = False

        if (cfg.TRAIN.LOSS).split('_')[-1]=='walpha':
            self.add_alpha_loss = True
        else:
            self.add_alpha_loss = False

        self.class_calculation = graf_angle_calc()

        self.bs = cfg.TRAIN.BATCH_SIZE
        self.lr = cfg.TRAIN.LR
        self.momentum = 0.99
        self.optimizer = self._get_optimizer(self.net)
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.evaluation = evaluation_helper()
        self.save_img = save_img

        self.outputpath=cfg.OUTPUT_PATH +'/validation'
        if os.path.exists(self.outputpath)==False:
            os.mkdir(self.outputpath)
        self.pixel_size = torch.tensor(cfg.DATASET.PIXEL_SIZE).to(cfg.MODEL.DEVICE)

        self.comparison_metrics=cfg.TEST.COMPARISON_METRICS
        self.sdr_thresholds = cfg.TEST.SDR_THRESHOLD
        self.sdr_units = cfg.TEST.SDR_UNITS
    
    def _get_optimizer(self,net):
        optim = torch.optim.SGD(net.parameters(), lr = self.lr, momentum=self.momentum)
        return optim
    
    
    def compare_metrics(self, id, pred, pred_map, true, true_map, pixelsize):
        df = pd.DataFrame({"ID": [id]})
        for metric in self.comparison_metrics:
            func = eval(metric)
            output = func(pred, pred_map, true, true_map, pixelsize)
            for i in output:
                df[i[0]]=[i[1]]

        return df
    
    def comparison_summary(self, df):
        summary_ls = []
        arr_mre = np.array([])
        for key in df.keys():
            try:
                mean_val=df[key].mean().round(2)
                summary_ls.append([key, mean_val])
                if 'landmark radial error' in key: 
                    arr_mre = np.append(arr_mre,mean_val)
            except:
                pass

        MRE = np.mean(arr_mre).round(2)
        MRE_std = np.std(arr_mre).round(2)

        return summary_ls, [MRE, MRE_std]
    
    def alpha_thresholds(self, df, thresholds=[1,2,5,10]):
        #this function calculates the different percentages of angle difference that lays in these thresholds
        df_alpha_diff = df['difference alpha']
        df_pred_class = df['class pred']
        df_true_class = df['class true']

        if type(df_alpha_diff) == pd.Series:
            alpha_diff = df_alpha_diff.to_numpy()
            pred_class = df_pred_class.to_numpy()
            true_class = df_true_class.to_numpy()
        

        np_agree= np.where(true_class == pred_class, 1.0, 0.0)
        np_disagree= np.where(true_class != pred_class, 1.0, 0.0)


        alpha_thresh = []
        for threshold in thresholds:            
            filter = np.where(alpha_diff < threshold, 1.0, 0.0)
            percent = 100 * np.sum(filter) / np.size(alpha_diff)

            percent_agree = 100 *np.sum(np_agree*filter)/np.size(np_agree)
            percent_disagree = 100*np.sum(np_disagree*filter)/np.size(np_disagree)

            alpha_thresh.append([percent,percent_agree.round(2),percent_disagree.round(2)])
            
            
        txt = ""
        for val in alpha_thresh:
            txt += "{:.2f}%, (agreeance {:.2f}%, disagreeance {:.2f}%)\t".format(val[0],val[1],val[2])
    
        return txt 
    
    def val_meta(self, dataloader, epoch):
        self.net.eval()
        total_loss = 0
        batches = 0

        with torch.no_grad():  # So no gradients accumulate#
            comparison_df = pd.DataFrame([])

            for batch_idx, (data, target, meta_data, id) in enumerate(dataloader):
                predicted_points = []
                
                batches += 1
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                meta_data = Variable(meta_data).to(self.device)
                
                # get prediction
                pred = self.net(data,meta_data)            

                if self.add_class_loss==True or self.add_alpha_loss == True:
                    pred_alphas, pred_classes,target_alphas, target_classes = self.class_calculation.get_class_from_output(pred,target,self.pixel_size)

                if self.l2_reg==True:
                    if self.add_class_loss==True:
                        loss = self.loss_func(pred.to(self.device), target.to(self.device), self.net, pred_classes, target_classes )
                    elif self.add_alpha_loss== True:
                        loss = self.loss_func(pred.to(self.device), target.to(self.device), self.net, pred_alphas, target_alphas)
                    else:
                        loss = self.loss_func(pred.to(self.device), target.to(self.device), self.net)
                else:
                    if self.add_class_loss==True:
                        loss = self.loss_func(pred.to(self.device), target.to(self.device), pred_classes, self.net, target_classes)
                    elif self.add_alpha_loss== True:
                        loss = self.loss_func(pred.to(self.device), target.to(self.device), pred_alphas, self.net, target_alphas)
                    else:
                        loss = self.loss_func(pred.to(self.device), target.to(self.device))

                total_loss += loss
                
                target_points, predicted_points = evaluation_helper().get_landmarks(pred, target, pixels_sizes=self.pixel_size)

                # save figures only on last epoch
                if epoch == self.max_epochs:
                    pass
                    ##only save on max epoch
                if self.save_img == True:
                    for i in range(self.bs):
                        if self.save_img  == True:
                            #
                            #print('saving validation img:', id[i])
                            #print(self.pixel_size[0])
                            visuals(self.outputpath+'/'+id[i], self.pixel_size[0]).heatmaps(data[i][0], pred[i], target_points[i], predicted_points[i])
                            visuals(self.outputpath+'/heatmap_'+id[i], self.pixel_size[0]).heatmaps(data[i][0], pred[i], target_points[i], predicted_points[i], w_landmarks=False)
            
                for i in range(self.bs):
                    #add to comparison df
                    print('Alpha for', id[i])
                    id_metric_df = self.compare_metrics(id[i], predicted_points[i], pred[i], target_points[i], target[i], self.pixel_size)

                    if comparison_df.empty == True:
                        comparison_df = id_metric_df
                    else:
                        comparison_df = comparison_df._append(id_metric_df, ignore_index=True)

            
        comparison_df.to_csv(self.outputpath+'/comparison_metrics.csv')
        print('Saving Results to comparison_metrics.csv')
        av_loss = total_loss / batches
        av_loss = av_loss.cpu().detach().numpy()
        print('Validation Set Average Loss: {:.4f}'.format(av_loss,  flush=True))

        #Get mean values from comparison summary ls, landmark metrics
        comparsion_summary_ls, MRE = self.comparison_summary(comparison_df)
        self.logger.info("MEAN VALUES: {}".format(comparsion_summary_ls))
        self.logger.info("MRE: {} +/- {} %".format(MRE[0], MRE[1]))

        alpha_thresh_percentages=self.alpha_thresholds(comparison_df)
        self.logger.info("Alpha Thresholds: {}".format(alpha_thresh_percentages))

        #from df get classification metrics TP, TN, FN, FP
        class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'class pred', 'class true', loc='validation')._get_metrics(group=True,groups=[('i'),('ii','iii/iv')])
        self.logger.info("Class Agreement: {}".format(class_agreement))

        class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'class pred', 'class true', loc='validation')._get_metrics(group=True,groups=[('i','ii'),('iii/iv')])
        self.logger.info("Class Agreement: {}".format(class_agreement))
        
        if self.dataset_type == 'LANDMARKS':
            #calculate SDR
            try:
                for i in range(self.num_landmarks):
                    col= 'landmark radial error p'+str(i+1)
                    sdr_stats, txt = landmark_overall_metrics(self.pixel_size, self.sdr_units).get_sdr_statistics(comparison_df[col], self.sdr_thresholds)
                    self.logger.info("{} for {}".format(txt, col))

            except:
                raise ValueError('Check Landmark radial errors are calcuated')

        #plot angles pred vs angles 
        visualisations.comparison(self.dataset_name).true_vs_pred_scatter(comparison_df['alpha pred'].to_numpy(),comparison_df['alpha true'].to_numpy(),loc='validation')


        return av_loss
