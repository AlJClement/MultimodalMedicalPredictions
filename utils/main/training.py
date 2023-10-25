import torch.nn as nn
import torch
from torch.autograd import Variable
import datetime
from .model_init import model_init
from .loss import *

class training():
    def __init__(self, cfg):
        self.model_init = model_init(cfg)
        #get specific models/feature loaders
        self.net = self.model_init.get_net_from_conf()

        self.loss_func = eval(cfg.TRAIN.LOSS)
        self.bs = cfg.TRAIN.BATCH_SIZE
        self.lr = cfg.TRAIN.LR
        self.momentum = 0.9
        self.optimizer = self._get_optimizer(self.net)
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        pass

    # def _get_lossfunc(self,loss_str):
    #     if loss_str == 'MSE':
    #         #nn.MSELoss()
    #         return mse_across_batch()
    #     elif loss_str == 'BSE':
    #         return bce_across_batch()
    #     elif loss_str == 'NLL':
    #         returnnll_across_batch()
    #     else:
    #         raise ValueError('loss fn not defined')
        
    def _get_optimizer(self,net):
        optim = torch.optim.SGD(net.parameters(), lr = self.lr, momentum=self.momentum)
        return optim
    
    def train_meta(self, dataloader, epoch):
        self.net.train()  #Put the network in train mode

        total_loss = 0
        batches = 0

        for batch_idx, (data, target, meta) in enumerate(dataloader):
            print(batch_idx)
            self.optimizer.zero_grad()
            data, target = Variable(data).to(self.device), Variable(target).to(self.device)
            meta_data = Variable(meta).to(self.device)

            batches += 1
            t_s= datetime.datetime.now()

            pred = self.net(data, meta_data)
            loss = self.loss_func(pred.to(self.device), target.to(self.device))
                
            loss.backward()
            self.optimizer.step()
            total_loss += loss
            
            if batch_idx % 100 == 0: #Report stats every x batches
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx+1) * len(data), len(dataloader.dataset),
                            100. * (batch_idx+1) / len(dataloader), loss.item()), flush=True)
            del loss, target, data, pred

        av_loss = total_loss / batches
        av_loss = av_loss.detach().cpu().numpy()
        print('\nTraining set: Average loss: {:.4f}'.format(av_loss,  flush=True))
        
        t_e= datetime.datetime.now()
        total_time =t_e-t_s
        print('Time taken for epoch = ', total_time)
        
        return av_loss
