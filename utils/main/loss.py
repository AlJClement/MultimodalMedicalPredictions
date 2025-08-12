import torch 
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from . import evaluation_helper

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
    
    mse_alpha = torch.pow(torch.FloatTensor(alpha_target)- torch.FloatTensor(alpha_output).double(), 2)
    mse_alpha=torch.mean(torch.sum(mse_alpha))
    mse_fhc = torch.pow(torch.FloatTensor(fhc_target)*100- torch.FloatTensor(fhc_out).double()*100, 2)
    mse_fhc=torch.mean(torch.sum(mse_fhc))

    g= gamma/2

    # ## first normalise so they all have equal input to decision
    nll_img_norm = nll_img/(mse_alpha+mse_fhc+nll_img)
    mse_alpha_norm = mse_alpha/(mse_alpha+mse_fhc+nll_img)
    mse_fhc_norm = mse_fhc/(mse_alpha+mse_fhc+nll_img)

    return nll_img*(1-gamma)+mse_alpha*g+mse_fhc*g

def get_normalized_vectors(P1, P2):
    vectors = []
    for p1, p2 in zip(P1, P2):
        v = p2 - p1  # torch subtraction
        if torch.all(v == 0):
            v = torch.tensor([0.0, 0.1], device=p1.device, dtype=p1.dtype)
        norm = torch.linalg.norm(v)
        v_normalized = v / (norm + 1e-8)
        vectors.append(v_normalized)
    return torch.stack(vectors)

def nll_across_batch_cosinelandmarkvector(output, target, gamma):
    nll = target * torch.log(output.double())
    nll_img = -torch.mean(torch.sum(nll, dim=(2, 3)))

    pixelsize=1
    target_points,predicted_points=evaluation_helper.evaluation_helper().get_landmarks(output, target, pixelsize)            
    # Normalize vectors
    ## vectors between ilium 1, bony ridge 2, fhc 3
    vec1 = get_normalized_vectors(predicted_points[:,0,:].float(), predicted_points[:,1,:].float())
    vec2 = get_normalized_vectors(predicted_points[:,2,:].float(), predicted_points[:,3,:].float())
    vec3 = get_normalized_vectors(predicted_points[:,5,:].float(), predicted_points[:,6,:].float())
    vec1_t = get_normalized_vectors(target_points[:,0,:].float(), target_points[:,1,:].float())
    vec2_t = get_normalized_vectors(target_points[:,2,:].float(), target_points[:,3,:].float())
    vec3_t = get_normalized_vectors(target_points[:,5,:].float(), target_points[:,6,:].float())

    # Cosine similarity
    cosine_sim_il = torch.mean(torch.sum(vec1 * vec1_t, dim=1))
    cosine_sim_br = torch.mean(torch.sum(vec2 * vec2_t, dim=1))
    cosine_sim_fhc = torch.mean(torch.sum(vec3 * vec3_t, dim=1))

    # cosine_sim_il = torch.tensor(cosine_sim_il, requires_grad=True)
    # cosine_sim_br = torch.tensor(cosine_sim_br, requires_grad=True)
    # cosine_sim_fhc = torch.tensor(cosine_sim_fhc, requires_grad=True)
    
    g= gamma/2
    g_a = g/2

    return nll_img*(1-gamma)+(cosine_sim_il*g_a)+(cosine_sim_br*g_a)+cosine_sim_fhc*g


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
        elif main_loss_str.split('_')[-1]== 'cosinelandmarkvector':
            self.addclass='cosinelandmarkvector'
        else:
            self.addclass=False
        
        print('loss', self.addclass)

    def forward(self, x, target, model,gamma, pred_alphas=None, target_alphas=None,class_output=None,class_target=None,pred_fhc=None,  target_fhc=None):
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
        elif self.addclass=='cosinelandmarkvector':
            print('pred a:',pred_alphas)
            print('target a:',target_alphas)
            print('pred fhc:',pred_fhc)
            print('target fhc:',target_fhc)
            loss = self.main_loss(x, target,gamma) + self.lam*l2
        else:
            loss = self.main_loss(x, target) + self.lam*l2

            
        return loss
    