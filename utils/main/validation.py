import torch.nn as nn
import torch
from torch.autograd import Variable
import datetime
from .model_init import model_init
from .loss import *
from .evaluation import *
import sys
import os
import pathlib
target_path = pathlib.Path(os.path.abspath(__file__)).parents[1]
sys.path.append(target_path)
from visualisations import visuals

class validation():
    def __init__(self, cfg, logger, save_img=True):
        self.model_init = model_init(cfg)
        self.logger = logger
        #get specific models/feature loaders
        self.net, self.net_param = self.model_init.get_net_from_conf()

        self.loss_func = eval(cfg.TRAIN.LOSS)
        self.bs = cfg.TRAIN.BATCH_SIZE
        self.lr = cfg.TRAIN.LR
        self.momentum = 0.9
        self.optimizer = self._get_optimizer(self.net)
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.evaluation = evaluation()
        self.save_img = save_img
        self.pixelsize = cfg.DATASET.PIXEL_SIZE

        self.outputpath=cfg.OUTPUT_PATH +'/validation_imgs'
        pass

    def _get_network(self):
        return self.net
    
    def _get_optimizer(self,net):
        optim = torch.optim.SGD(net.parameters(), lr = self.lr, momentum=self.momentum)
        return optim
    
    def val_meta(self, dataloader, epoch):
        self.net.eval()
        total_loss = 0
        batches = 0

        with torch.no_grad():  # So no gradients accumulate
            for batch_idx, (data, target, meta_data, id) in enumerate(dataloader):
                predicted_points = []
                scaled_predicted_points = []
                image_eres = []

                batches += 1
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                meta_data = Variable(meta_data).to(self.device)
                
                # get prediction
                pred = self.net(data.to(self.device), meta_data.to(self.device))
                loss = self.loss_func(pred.to(self.device), target.to(self.device))
                total_loss += loss

                # evaluation
                scaled_predicted_points, predicted_points, target_points, eres = self.evaluation.get_landmarks(pred, target, self.pixelsize)
                scaled_predicted_points.append(scaled_predicted_points)
                predicted_points.append(predicted_points)
                image_eres.append(eres)

                # save figures
                if self.save_img == True:
                    if not os.path.isdir(self.outputpath+'/'+pat_id):
                        os.mkdir(self.outputpath+'/'+pat_id)

                    img_path=self.outputpath+'/'+pat_id
                    visuals.val_figure(data, pred.numpy(), predicted_points,
                                        target_points, eres, batch_idx, save=True, save_path=img_path)
                    
            av_loss = total_loss / batches

        av_loss = av_loss.cpu().detach().numpy()
        print('Validation set: Average loss: {:.4f}'.format(av_loss,  flush=True))

        return av_loss
