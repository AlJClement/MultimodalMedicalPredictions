import torch 
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
# dimensions are [B, C, W, H]
# the log is done within this loss function whereas normally it would be a log softmax
# why? - i think because think about log graph would be really steep
def nll_across_batch_wclass(output, target, class_output, class_target):
    nll = target * torch.log(output.double())
    return -torch.mean(torch.sum(nll, dim=(2, 3)))
    
def nll_across_batch(output, target):
    nll = target * torch.log(output.double())
    return -torch.mean(torch.sum(nll, dim=(2, 3)))

def bce_across_batch(output, target):
    bce = target * torch.log(output.double()) + (1 - target) * torch.log(1 - output.double())
    return -torch.mean(torch.sum(bce, dim=(2, 3)))

def mse_across_batch(output, target):
    mse = torch.pow(target - output.double(), 2)
    return -torch.mean(torch.sum(mse, dim=(2, 3)))

def plot_all_loss(losses, max_epochs, save_path):
    its = np.linspace(1, max_epochs, max_epochs)
    plt.figure()
    plt.plot(its, losses[0,:])
    plt.plot(its, losses[1,:])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])

    plt.savefig(save_path+'/loss_fig.png')

class L2RegLoss(nn.Module):
    def __init__(self, main_loss_str, lam=0.01, mu=1):
        super(L2RegLoss, self).__init__()
        self.eps = 1e-7
        self.mu = mu
        self.lam = lam
        self.main_loss=eval(main_loss_str)

    def forward(self, x, target, model):
        #abs(p) for l1
        l2 = [p.pow(2).sum() for p in model.parameters()]
        l2 = sum(l2)
        loss = self.main_loss(x, target) + self.lam*l2
        
        return loss
    