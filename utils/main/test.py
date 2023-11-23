import torch.nn as nn
import torch
from torch.autograd import Variable
import datetime
from .model_init import model_init
from .loss import *
torch.cuda.empty_cache() 
from .model_init import model_init
from ..visualisations.visuals import visuals
from .evaluation_helper import evaluation
import os
import pandas as pd

class test():
    def __init__(self, cfg, logger):
        self.plot_predictions = True
        self.logger = logger
        self.net = self.load_network(cfg)

        self.loss_func = eval(cfg.TRAIN.LOSS)
        self.bs = cfg.TRAIN.BATCH_SIZE
        self.lr = cfg.TRAIN.LR
        self.momentum = 0.9
        self.optimizer = self._get_optimizer(self.net)
        self.device = torch.device(cfg.MODEL.DEVICE)
        if cfg.MODEL.DEVICE == 'cuda':
            torch.cuda.empty_cache()

        self.save_img_path = cfg.OUTPUT_PATH +'/test'

        if not self.save_img_path.exists():
            os.mkdir(self.save_img_path)

        self.compare_metrics=cfg.TEST.COMPARISON_METRICS

    def load_network(self, model_path):
        model = model_init(self.cfg).get_net_from_conf(self, get_net_info=False)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    def compare_metrics(self, id, pred, true):
        df = pd.DataFrame({"ID": id})
        for metric in self.comparison_metrics:
            func = eval("main.comparison_metrics(landmarks)." + metric)
            output_pred, output_true=func(pred,true)
            df[metric+'_pred']=output_pred
            df[metric+'_true']=output_true

        return df

    
    def test_meta(self, dataloader):
        comparison_df = pd.Dataframe([])
        for batch_idx, (data, target, meta, id) in enumerate(dataloader):
            print(batch_idx)

            data, target = Variable(data).to(self.device), Variable(target).to(self.device)
            meta_data = Variable(meta).to(self.device)
            
            #target shape: (B, C, W, H) - where C is #landmarks
            pred = self.net(data, meta_data)
            pred=pred.detach().cpu().numpy()
            data=data.detach().cpu().numpy()[0][0]

            predicted_points, target_points=evaluation().get_landmarks(self, pred, target_points)

            if self.plot_prediction == True:
                visuals(self.save_img_path).heatmaps(data, pred, target_points, predicted_points)

            #add to comparison df
            id_metric_df = self.compare_metrics(id, predicted_points, target_points)

            if comparison_df.empty == True:
                comparison_df = id_metric_df
            else:
                comparison_df = comparison_df.append(id_metric_df, ignore_index=True)
            
            batches += 1
            t_s= datetime.datetime.now()

        
        t_e= datetime.datetime.now()
        total_time =t_e-t_s
        print('Time taken for epoch = ', total_time)
        
        return 
