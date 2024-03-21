import torch.nn as nn
import torch
from torch.autograd import Variable
import datetime
from .model_init import model_init
from .loss import *
torch.cuda.empty_cache() 
from .comparison_metrics import graf_angle_calc
from .evaluation_helper import evaluation_helper

class training():
    def __init__(self, cfg, logger, l2_reg=True):
        self.plot_target = False
    
        self.model_init = model_init(cfg)
        self.logger = logger
        #get specific models/feature loaders
        self.net, self.net_param = self.model_init.get_net_from_conf()
        self.l2_reg=l2_reg
        if l2_reg == True:
            self.loss_func = eval('L2RegLoss(cfg.TRAIN.LOSS)')
        else:
            self.loss_func = eval(cfg.TRAIN.LOSS)
        
        if (cfg.TRAIN.LOSS).split('_')[-1]=='wclass':
            self.add_class_loss = True
            self.gamma = cfg.TRAIN.GAMMA
        else:
            self.add_class_loss = False

        if (cfg.TRAIN.LOSS).split('_')[-1]=='walpha':
            self.add_alpha_loss = True
            self.gamma = cfg.TRAIN.GAMMA

        else:
            self.add_alpha_loss = False

        self.class_calculation = graf_angle_calc()

        self.bs = cfg.TRAIN.BATCH_SIZE
        self.lr = cfg.TRAIN.LR
        self.momentum = 0.99
        self.optimizer = self._get_optimizer(self.net)
        self.device = torch.device(cfg.MODEL.DEVICE)
        if cfg.MODEL.DEVICE == 'cuda':
            torch.cuda.empty_cache()
        self.pixel_size = torch.tensor(cfg.DATASET.PIXEL_SIZE).to(cfg.MODEL.DEVICE)
        
        pass

    def _get_network(self):
        return self.net
    
    def _get_optimizer(self,net):
        optim = torch.optim.SGD(net.parameters(), lr = self.lr, momentum=self.momentum)
        return optim
    
    def train_meta(self, dataloader, epoch):
        self.net.train()  #Put the network in train mode
        total_loss = 0
        batches = 0

        for batch_idx, (data, target, meta, id) in enumerate(dataloader):
            print(batch_idx)
            self.optimizer.zero_grad()

            #data shape: (B, 1, W, H)
            #target shape: (B, C, W, H) - where C is #landmarks
            #meta_data shape: (B, 1, NUM_metafeatures)
            data, target = Variable(data).to(self.device), Variable(target).to(self.device)
            meta_data = Variable(meta).to(self.device)
            
            if self.plot_target == True:
                tar=target.detach().cpu().numpy()
                d=data.detach().cpu().numpy()[0][0]
                for c in range(tar[0].shape[0]):
                    try:
                        tar_im = tar_im+tar[0][c]
                    except:
                        tar_im = tar[0][c]
    
                plt.imshow(tar_im)
                plt.imshow(d)

            batches += 1
            t_s= datetime.datetime.now()

            #target shape: (B, C, W, H) - where C is #landmarks
            pred = self.net(data, meta_data)

            if self.add_class_loss==True or self.add_alpha_loss == True:
                pred_alphas, pred_classes,target_alphas, target_classes = self.class_calculation.get_class_from_output(pred,target,self.pixel_size)

            
            if self.l2_reg==True:
                if self.add_class_loss==True:
                    loss = self.loss_func(pred.to(self.device), target.to(self.device), self.net, pred_alphas, target_alphas,pred_classes, target_classes,self.gamma)
                elif self.add_alpha_loss== True:
                    loss = self.loss_func(pred.to(self.device), target.to(self.device), self.net, pred_alphas, target_alphas,self.gamma)
                else:
                    loss = self.loss_func(pred.to(self.device), target.to(self.device), self.net)
            else:
                if self.add_class_loss==True:
                    loss = self.loss_func(pred.to(self.device), target.to(self.device), self.net,pred_alphas, target_alphas, pred_classes, target_classes,self.gamma)
                elif self.add_alpha_loss== True:
                    loss = self.loss_func(pred.to(self.device), target.to(self.device), self.net, pred_alphas, target_alphas,self.gamma)
                else:
                    loss = self.loss_func(pred.to(self.device), target.to(self.device))

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0: #Report stats every x batches
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx+1) * len(data), len(dataloader.dataset),
                            100. * (batch_idx+1) / len(dataloader), loss.item()), flush=True)
                    
            del loss, target, data, pred
            torch.cuda.empty_cache()

        av_loss = total_loss / batches
      #av_loss = av_loss.detach().cpu().numpy()
        print('\nTraining Set Average loss: {:.4f}'.format(av_loss,  flush=True))
        
        t_e= datetime.datetime.now()
        total_time =t_e-t_s
        print('Time taken for epoch = ', total_time)
        
        return av_loss
