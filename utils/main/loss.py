import torch 
import numpy as np
import matplotlib.pyplot as plt
# dimensions are [B, C, W, H]
# the log is done within this loss function whereas normally it would be a log softmax
# why? - i think because think about log graph would be really steep

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
