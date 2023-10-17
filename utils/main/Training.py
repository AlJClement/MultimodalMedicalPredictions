import torch.nn as nn
import tensorflow as tf
from torch.autograd import Variable
import datetime

class Training():
    def __init__(self, cfg) -> None:
        self.net = eval("models." + cfg.MODEL.NAME)(cfg.MODEL)

        pass

    def train_meta(self, dataloader, optim, loss_func, epoch, device='cpu', type=None, dsc_loss = None, run_meta_data_btl = False):
        self.net.train()  #Put the network in train mode
        total_loss = 0
        batches = 0
        count = 1
        if epoch>1:
            count =1

        for batch_idx, (data, target, meta_data, id) in enumerate(dataloader):
            optim.zero_grad()
            data, target = Variable(data).to(device), Variable(target).to(device)
            meta_data = Variable(meta_data).to(device)
            batches += 1
            t_s= datetime.datetime.now()

            # Training loop
            if run_meta_data_btl == False:
                pred = self.net(data.to(device))
            else:
                # get prediction 
                # print(meta_data.is_cuda)
                # print(data.is_cuda)
                pred = self.net(data, meta_data)

            if type == 'moi':
                if count == 0:
                    loss = dsc_loss(pred.to(device), target.to(device))
                    count = count+1

                else:
                    # pred = torch.multiply(pred,data.to(torch.float32))
                    # target = torch.multiply(target,data.to(torch.float32))
                    loss = loss_func(pred.to(device),target.to(device))

            else:
                loss = loss_func(pred.to(device), target.to(device))
                
            loss.backward()
            optim.step()
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
    
    def val_meta(net, val_dataloader, optim, loss_func, epoch, type=None, dsc_loss=None, run_meta_data_btl=False):
        net.eval()
        total_loss = 0    
        batches = 0
        count = 0
        if epoch>1:
            count =1
        
        with torch.no_grad():  # So no gradients accumulate
            for batch_idx, (data, target, meta_data, id) in enumerate(val_dataloader):
                batches += 1
                data, target, meta_data = Variable(data).to(device), Variable(target).to(device), Variable(meta_data).to(device)
                print(meta_data)
                
                # get prediction
            if run_meta_data_btl == False:
                pred = net(data.to(device))
            else:
                pred = net(data.to(device), meta_data.to(device))

                if type == 'moi':
                    if count == 0:
                        loss = dsc_loss(pred.to(device), target.to(device))
                        count = count+1

                    else:
                        loss = loss_func(pred.to(device),target.to(device))

                else:
                    loss = loss_func(pred.to(device), target.to(device))

                total_loss += loss
            del loss, target, data, pred

            av_loss = total_loss / batches
            
        av_loss = av_loss.cpu().detach().numpy()
        print('Validation set: Average loss: {:.4f}'.format(av_loss,  flush=True))
        print('\n')
        return av_loss`