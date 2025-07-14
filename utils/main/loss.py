import torch 
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
# dimensions are [B, C, W, H]
# the log is done within this loss function whereas normally it would be a log softmax
# why? - i think because think about log graph would be really steep
def nll_across_batch_mse_wclass(output, target, pred_alphas, target_alphas, class_output, class_target,gamma=0.1,add_weights=True):
    nll = target * torch.log(output.double())
    nll_img = -torch.mean(torch.sum(nll, dim=(2, 3)))
    classes = {'i':0,'ii':1,'iii/iv':2}

    ###get pred as torch
    pred_alpha_torch=torch.FloatTensor(np.array(pred_alphas))
    target_alpha_torch=torch.FloatTensor(np.array(target_alphas))
    
    #maybe try making this conservative so minus the 
    #replace strings wthi values for classes
    class_output_, class_target_=np.array(class_output), np.array(class_target)
    for c,val in classes.items():
        for i in range(len(class_output_)):
            if class_output_[i] == c: 
                class_output_[i]= int(val)
            if class_target_[i] == c: 
               class_target_[i]= int(val)

    one_output=class_output_.astype(int)
    one_target=class_target_.astype(int)
    _one_o=torch.LongTensor(one_output)
    _one_t=torch.LongTensor(one_target)

    ##hot encode
    nb_classes = len(classes)

    one_hot_outputs=torch.nn.functional.one_hot(_one_o,nb_classes)
    one_hot_targets=torch.nn.functional.one_hot(_one_t,nb_classes)

    ##if add weights:
    weights = torch.LongTensor([[1],[2],[4]])
    if add_weights == True:
        one_hot_outputs= torch.transpose((torch.transpose(one_hot_outputs,0,1)*weights),1,0)
        one_hot_targets= torch.transpose((torch.transpose(one_hot_targets,0,1)*weights),1,0)


    #mutlply by weights and divide by weight of predicted class
    one_hot_outputs=torch.sum(one_hot_outputs,1)
    one_hot_targets=torch.sum(one_hot_targets,1)
    weighted_pred = pred_alpha_torch.to(float)*one_hot_outputs.to(float)
    weighted_target = target_alpha_torch.to(float)*one_hot_targets.to(float)
    

    diff = torch.divide(torch.abs(torch.subtract(weighted_target,weighted_pred)),one_hot_outputs)

    mse = torch.pow(diff.double(), 2)
    mse_class=torch.mean(torch.sum(mse))

    return nll_img*(1-gamma)+ mse_class*gamma #-torch.mean(torch.sum(nll, dim=(2, 3)))

def nll_across_batch_nll_walpha(output, target, alpha_output, alpha_target, gamma):
    nll = target * torch.log(output.double())
    nll_img = -torch.mean(torch.sum(nll, dim=(2, 3)))

    nll = torch.FloatTensor(alpha_target)*torch.log(torch.FloatTensor(alpha_output).double())
    nll_alpha = -torch.mean(torch.sum(nll))

    return nll_img*(1-gamma)+ nll_alpha*gamma #-torch.mean(torch.sum(nll, dim=(2, 3)))


def nll_across_batch_mse_walpha(output, target, alpha_output, alpha_target, gamma):
    nll = target * torch.log(output.double())
    nll_img = -torch.mean(torch.sum(nll, dim=(2, 3)))
        
    mse = torch.pow(torch.FloatTensor(alpha_target )- torch.FloatTensor(alpha_output).double(), 2)
    mse_alpha=torch.mean(torch.sum(mse))

    return nll_img*(1-gamma)+mse_alpha*gamma

def nll_across_batch_mse_walphafhc(output, target, alpha_output, alpha_target, fhc_out, fhc_target, gamma):
    nll = target * torch.log(output.double())
    nll_img = -torch.mean(torch.sum(nll, dim=(2, 3)))
        
    mse_alpha = torch.pow(torch.FloatTensor(alpha_target )- torch.FloatTensor(alpha_output).double(), 2)
    mse_alpha=torch.mean(torch.sum(mse_alpha))
    mse_fhc = torch.pow(torch.FloatTensor(fhc_target )- torch.FloatTensor(fhc_out).double(), 2)
    mse_fhc=torch.mean(torch.sum(mse_fhc))

    g= gamma/2

    return nll_img*(1-gamma)+mse_alpha*g+mse_fhc*g


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

        if main_loss_str.split('_')[-1]=='walpha':
            self.addclass='walpha'
        elif main_loss_str.split('_')[-1]=='walphafhc':
            self.addclass='walphafhc'
        elif main_loss_str.split('_')[-1]== 'wclass':
            self.addclass='wclass'
        else:
            self.addclass=False

    def forward(self, x, target, model,  pred_alphas=None, target_alphas=None,class_output=None,class_target=None,pred_fhc=None,  target_fhc=None,gamma=0.0):
        #abs(p) for l1
        l2 = [p.pow(2).sum() for p in model.parameters()]
        l2 = sum(l2)

        if self.addclass=='walpha':
            print('pred:',pred_alphas)
            print('target:',target_alphas)
            loss = self.main_loss(x, target,pred_alphas,target_alphas,gamma) + self.lam*l2
        elif self.addclass=='wclass':
            print('pred:',class_output)
            print('target:',class_target)
            loss = self.main_loss(x, target,pred_alphas,target_alphas,class_output,class_target,gamma) + self.lam*l2
        elif self.addclass=='walphafhc':
            print('pred a:',pred_alphas)
            print('target a:',target_alphas)
            print('pred fhc:',pred_fhc)
            print('target fhc:',target_fhc)
            loss = self.main_loss(x, target,pred_alphas,target_alphas,pred_fhc, target_fhc,gamma) + self.lam*l2
        else:
            loss = self.main_loss(x, target) + self.lam*l2

            
        return loss
    