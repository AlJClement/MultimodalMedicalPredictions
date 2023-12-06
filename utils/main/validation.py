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
from visualisations import visuals
from .evaluation_helper import evaluation_helper
import pandas as pd
from .comparison_metrics import *
class validation():
    def __init__(self, cfg, logger, net, l2_reg=True, save_img=True):
        self.net = net
        self.logger = logger
        self.l2_reg=l2_reg
        if l2_reg == True:
            self.loss_func = eval('L2RegLoss(cfg.TRAIN.LOSS)')
        else:
            self.loss_func = eval(cfg.TRAIN.LOSS)

        self.bs = cfg.TRAIN.BATCH_SIZE
        self.lr = cfg.TRAIN.LR
        self.momentum = 0.99
        self.optimizer = self._get_optimizer(self.net)
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.evaluation = evaluation_helper()
        self.save_img = save_img
        self.pixelsize = torch.tensor(cfg.DATASET.PIXEL_SIZE).to(cfg.MODEL.DEVICE)

        self.outputpath=cfg.OUTPUT_PATH +'/validation_imgs'
        self.pixel_size = torch.tensor(cfg.DATASET.PIXEL_SIZE).to(cfg.MODEL.DEVICE)

        self.comparison_metrics=cfg.TEST.COMPARISON_METRICS

        pass
    
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
        for key in df.keys():
            try:
                mean_val=df[key].mean().round(2)
                summary_ls.append([key, mean_val])
            except:
                pass
        return summary_ls
    
    
    def val_meta(self, dataloader, epoch):
        self.net.eval()
        total_loss = 0
        batches = 0

        with torch.no_grad():  # So no gradients accumulate
            for batch_idx, (data, target, meta_data, id) in enumerate(dataloader):
                predicted_points = []
                scaled_predicted_points = []
                
                batches += 1
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                meta_data = Variable(meta_data).to(self.device)
                
                # get prediction
                pred = self.net(data,meta_data)            
                if self.l2_reg==True:
                    loss = self.loss_func(pred.to(self.device), target.to(self.device), self.net)
                else:
                    loss = self.loss_func(pred.to(self.device), target.to(self.device))
                total_loss += loss

                # evaluation
                comparison_df = pd.DataFrame([])

                for i in range(len(pred)):

                    predicted_points, target_points=evaluation_helper().get_landmarks(pred, target, pixels_sizes=self.pixel_size)

                    # save figures
                    if self.save_img == True:
                        for i in range(self.bs):
                            if self.save_img  == True:
                                #print('saving validation img:', id[i])
                                visuals(self.outputpath+'/'+id[i]).heatmaps(data[i][0], pred[i], target_points[i], predicted_points[i])

                            #add to comparison df
                            id_metric_df = self.compare_metrics(id[i], predicted_points[i], pred[i], target_points[i], target[i],self.pixel_size)

                            if comparison_df.empty == True:
                                comparison_df = id_metric_df
                            else:
                                comparison_df = comparison_df._append(id_metric_df, ignore_index=True)

            
                comparison_df.to_csv(self.outputpath+'/comparison_metrics.csv')
                print('Saving Results to comparison_metrics.csv')
                av_loss = total_loss / batches
                av_loss = av_loss.cpu().detach().numpy()
                print('Validation Set Average Loss: {:.4f}'.format(av_loss,  flush=True))

        return av_loss
