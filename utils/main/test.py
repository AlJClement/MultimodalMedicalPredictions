import torch.nn as nn
import torch
from torch.autograd import Variable
import datetime
from .model_init import model_init
from .loss import *
import sys
import pathlib
import os
from .evaluation_helper import evaluation_helper
target_path = pathlib.Path(os.path.abspath(__file__)).parents[1]
sys.path.append(target_path)
from .model_init import model_init
import visualisations
from visualisations import visuals
import os
import pandas as pd
from .comparison_metrics import *
import torch
torch.cuda.empty_cache() 
class test():
    def __init__(self, cfg, logger):
        self.cfg=cfg
        self.pixel_size = torch.tensor(cfg.DATASET.PIXEL_SIZE).to(cfg.MODEL.DEVICE)
        self.plot_predictions = True
        self.logger = logger
        self.net = self.load_network(cfg.TEST.NETWORK)
        self.dataset_type = cfg.DATASET.ANNOTATION_TYPE

    
        self.loss_func = eval(cfg.TRAIN.LOSS)
        self.bs = cfg.TRAIN.BATCH_SIZE
        self.lr = cfg.TRAIN.LR
        self.momentum = 0.9
        self.device = torch.device(cfg.MODEL.DEVICE)

        if cfg.MODEL.DEVICE == 'cuda':
            torch.cuda.empty_cache()

        self.save_img_path = cfg.OUTPUT_PATH +'/test'

        if not os.path.exists(self.save_img_path):
            os.mkdir(self.save_img_path)
        
        self.pixel_size = torch.tensor(cfg.DATASET.PIXEL_SIZE).to(cfg.MODEL.DEVICE)

        self.comparison_metrics=cfg.TEST.COMPARISON_METRICS
        self.dataset_name = cfg.INPUT_PATHS.DATASET_NAME

    def load_network(self, model_path):
        model = model_init(self.cfg).get_net_from_conf(get_net_info=False)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
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
        for key in df.keys():
            try:
                mean_val=df[key].mean().round(2)
                summary_ls.append([key, mean_val])
            except:
                pass
        return summary_ls

    def alpha_thresholds(self, df_col, thresholds=[1,2,5,10]):
        #this function calculates the different percentages of angle difference that lays in these thresholds
        if type(df_col) == pd.Series:
            alpha_diff = df_col.to_numpy()
            
        alpha_thresh = []
        for threshold in thresholds:
            filter = np.where(alpha_diff < threshold, 1.0, 0.0)
            percent = 100 * np.sum(filter) / np.size(alpha_diff)
            alpha_thresh.append(percent)
            
        txt = ""
        for val in alpha_thresh:
            txt += "{:.2f}%\t".format(val)
    
        return txt 
    
    def run(self, dataloader):
        comparison_df = pd.DataFrame([])
        for batch_idx, (data, target, meta, id) in enumerate(dataloader):
            print(batch_idx)

            data, target = Variable(data).to(self.device), Variable(target).to(self.device)
            meta_data = Variable(meta).to(self.device)
            
            pred = self.net(data, meta_data)
            target_points,predicted_points=evaluation_helper().get_landmarks(pred, target, pixels_sizes=self.pixel_size)

            #plot and caluclate values for each subject in the batch
            for i in range(self.bs):

                if self.plot_predictions == True:
                    print('saving test img:', id[i])
                    visuals(self.outputpath+'/'+id[i], self.pixel_size[0]).heatmaps(data[i][0], pred[i], target_points[i], predicted_points[i])
                    visuals(self.outputpath+'/heatmap_'+id[i], self.pixel_size[0]).heatmaps(data[i][0], pred[i], target_points[i], predicted_points[i], w_landmarks=False)
            
                #add to comparison df
                id_metric_df = self.compare_metrics(id[i], predicted_points[i], pred[i], target_points[i], target[i],self.pixel_size)

                for i in range(self.bs):
                    #add to comparison df
                    print('Alpha for', id[i])
                    id_metric_df = self.compare_metrics(id[i], predicted_points[i], pred[i], target_points[i], target[i], self.pixel_size)

                    if comparison_df.empty == True:
                        comparison_df = id_metric_df
                    else:
                        comparison_df = comparison_df._append(id_metric_df, ignore_index=True)

            t_s=datetime.datetime.now()

        t_e=datetime.datetime.now()
        total_time=t_e-t_s
        print('Time taken for epoch = ', total_time)
        comparison_df.to_csv('./output/test/comparison_metrics.csv')
        print('Saving Results to comparison_metrics.csv')
        
        comparsion_summary_ls = self.comparison_summary(comparison_df)
        self.logger.info("SUMMARY: {}".format(comparsion_summary_ls))

        #from df get class agreement metrics TP, TN, FN, FP
        class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'class pred', 'class true')._get_metrics(group=True,groups=[('i'),('ii','iii/iv')])
        self.logger.info("Class Agreement: {}".format(class_agreement))

        class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'class pred', 'class true')._get_metrics(group=True,groups=[('i','ii'),('iii/iv')])
        self.logger.info("Class Agreement: {}".format(class_agreement))
        
        if self.dataset_type == 'LANDMARKS':
            #calculate SDR
            try:
                for i in range(self.num_landmarks):
                    col= 'landmark radial error p'+str(i+1)
                    sdr_stats, txt = landmark_overall_metrics(self.pixel_size).get_sdr_statistics(comparison_df[col], self.sdr_thresholds)
                    self.logger.info("{} for {}".format(txt, col))

            except:
                raise ValueError('Check Landmark radial errors are calcuated')

        #plot angles pred vs angles 
        visualisations.comparison(self.dataset_name).true_vs_pred_scatter(comparison_df['alpha pred'].to_numpy(),comparison_df['alpha true'].to_numpy())

        return 
