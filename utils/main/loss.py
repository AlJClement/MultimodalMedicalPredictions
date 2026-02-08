import torch 
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from . import evaluation_helper
import math
from . import comparison_metrics

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
    target_points,predicted_points=evaluation_helper.evaluation_helper().get_landmarks(output, target, pixelsize, gumbel=False)            
    # Normalize vectors
    ## vectors between ilium 1, bony ridge 2, fhc 3
    vec1 = get_normalized_vectors(predicted_points[:,0,:].float(), predicted_points[:,1,:].float())
    vec2 = get_normalized_vectors(predicted_points[:,2,:].float(), predicted_points[:,3,:].float())
    vec3 = get_normalized_vectors(predicted_points[:,5,:].float(), predicted_points[:,6,:].float())
    vec1_t = get_normalized_vectors(target_points[:,0,:].float(), target_points[:,1,:].float())
    vec2_t = get_normalized_vectors(target_points[:,2,:].float(), target_points[:,3,:].float())
    vec3_t = get_normalized_vectors(target_points[:,5,:].float(), target_points[:,6,:].float())

    # Cosine similarity, mean across batch, normalised so just the multiplcation of vector 1 and 2
    cosine_sim_il = torch.mean(torch.sum(vec1 * vec1_t, dim=1))
    cosine_sim_br = torch.mean(torch.sum(vec2 * vec2_t, dim=1))
    cosine_sim_fhc = torch.mean(torch.sum(vec3 * vec3_t, dim=1))

    # cosine_sim_il = torch.tensor(cosine_sim_il, requires_grad=True)
    # cosine_sim_br = torch.tensor(cosine_sim_br, requires_grad=True)
    # cosine_sim_fhc = torch.tensor(cosine_sim_fhc, requires_grad=True)
    
    g= gamma/2
    g_a = g/2

    return nll_img*(1-gamma)+(cosine_sim_il*g_a)+(cosine_sim_br*g_a)+cosine_sim_fhc*g

def nll_across_batch_cosinelandmarkvectorOAI(output, target, gamma):
    nll = target * torch.log(output.double())
    nll_img = -torch.mean(torch.sum(nll, dim=(2, 3)))

    pixelsize=1
    target_points,predicted_points=evaluation_helper.evaluation_helper().get_landmarks(output, target, pixelsize, gumbel=True)            
    # Normalize vectors
    ## vectors between v1 knee and hip and v2 knee and foot
    vec1 = get_normalized_vectors(predicted_points[:,0,:].float(), predicted_points[:,1,:].float())
    vec2 = get_normalized_vectors(predicted_points[:,1,:].float(), predicted_points[:,2,:].float())
    vec3 = get_normalized_vectors(predicted_points[:,3,:].float(), predicted_points[:,4,:].float())
    vec4 = get_normalized_vectors(predicted_points[:,4,:].float(), predicted_points[:,5,:].float())

    eps = 1e-7  # numerical stability

    # dot products per sample
    dot_side1 = torch.sum(vec1 * vec2, dim=1)
    dot_side2 = torch.sum(vec3 * vec4, dim=1)

    # clamp to avoid NaNs
    dot_side1 = dot_side1.clamp(-1.0 + eps, 1.0 - eps)
    dot_side2 = dot_side2.clamp(-1.0 + eps, 1.0 - eps)

    # angles in radians
    angle_1 = torch.acos(dot_side1)
    angle_2 = torch.acos(dot_side2)

    # optional: convert to degrees
    angle_1_deg = angle_1 * (180.0 / math.pi)
    angle_2_deg = angle_2 * (180.0 / math.pi)

    # mean angle across batch
    angle_1_deg = angle_1_deg.mean()
    angle_2_deg = angle_2_deg.mean()

       # Normalize vectors
    ## vectors between v1 knee and hip and v2 knee and foot
    vec1_t = get_normalized_vectors(target_points[:,0,:].float(), target_points[:,1,:].float())
    vec2_t = get_normalized_vectors(target_points[:,1,:].float(), target_points[:,2,:].float())
    vec3_t = get_normalized_vectors(target_points[:,3,:].float(), target_points[:,4,:].float())
    vec4_t = get_normalized_vectors(target_points[:,4,:].float(), target_points[:,5,:].float())

    eps = 1e-7  # numerical stability

    # dot products per sample
    dot_side1_t = torch.sum(vec1_t * vec2_t, dim=1)
    dot_side2_t = torch.sum(vec3_t * vec4_t, dim=1)

    # clamp to avoid NaNs
    dot_side1_t = dot_side1_t.clamp(-1.0 + eps, 1.0 - eps)
    dot_side2_t = dot_side2_t.clamp(-1.0 + eps, 1.0 - eps)

    # angles in radians
    angle_1_t = torch.acos(dot_side1_t)
    angle_2_t = torch.acos(dot_side2_t)

    # optional: convert to degrees
    angle_1_deg_t = angle_1_t * (180.0 / math.pi)
    angle_2_deg_t = angle_2_t * (180.0 / math.pi)

    # mean angle across batch
    angle_1_deg_t = angle_1_deg_t.mean()
    angle_2_deg_t = angle_2_deg_t.mean() 
    
    g = gamma/2
    a1 = angle_1_deg-angle_1_deg_t
    a2 = angle_2_deg-angle_2_deg_t

    return nll_img*(1-gamma)+(a1*g)+(a2*g)

def nll_across_batch_OAIanglediff(output, target, gamma, cfg, gumbel=True):
    nll_across_batch_OAImrediff.calls += 1

    add_loss_after_iter = cfg.TRAIN.DELAY_GUMBEL_LOSS

    nll = target * torch.log(output.double())
    nll_img = -torch.mean(torch.sum(nll, dim=(2, 3)))

    if int(nll_across_batch_OAImrediff.calls/2) >= add_loss_after_iter:
        pixelsize=1
        target_points,predicted_points=evaluation_helper.evaluation_helper().get_landmarks(output, target, pixelsize, gumbel, nll_across_batch_OAImrediff.calls, cfg)            

        pred_L_arr = []
        pred_R_arr = []
        targ_L_arr = []
        targ_R_arr = []
        
        for i in range(predicted_points.shape[0]):
            p_swapped = predicted_points[i]
            pred = comparison_metrics.protractor_hka().hka_angles(p_swapped,[],[],[],[])
            pred_L, pred_R = pred[0][1], pred[3][1]
            t_swapped = target_points[i]
            targ = comparison_metrics.protractor_hka().hka_angles(t_swapped,[],[],[],[])
            targ_L, targ_R = targ[0][1], targ[3][1]
            pred_L_arr.append(pred_L)
            pred_R_arr.append(pred_R)
            targ_L_arr.append(targ_L)
            targ_R_arr.append(targ_R)

        pred_L_arr = np.array(pred_L_arr)
        pred_R_arr = np.array(pred_R_arr)
        targ_L_arr = np.array(targ_L_arr)
        targ_R_arr = np.array(targ_R_arr)
            
        g = gamma/2
        a1 = np.mean(abs(pred_L_arr-targ_L_arr))
        a2 = np.mean(abs(pred_R_arr-targ_R_arr))

        return nll_img*(1-gamma)+(a1*g)+(a2*g)
    else:
        return nll_img

def nll_across_batch_OAImrediff(output, target, gamma, cfg=None,gumbel=True):
    nll_across_batch_OAImrediff.calls += 1
    #print('CALLS: ', nll_across_batch_OAImrediff.calls)
    add_loss_after_iter = cfg.TRAIN.DELAY_GUMBEL_LOSS

    nll = target * torch.log(output.double())
    nll_img = -torch.mean(torch.sum(nll, dim=(2, 3)))

    if int(nll_across_batch_OAImrediff.calls/2) > add_loss_after_iter:
        pixelsize=1
        target_points,predicted_points=evaluation_helper.evaluation_helper().get_landmarks(output, target, pixelsize, gumbel, nll_across_batch_OAImrediff.calls, cfg)            

        diff = predicted_points - target_points 
        # euc_per_point = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)  
        # point_loss = torch.mean(euc_per_point)  # scalar

        ###L2 Loss - peanalizing the large loss
        sq_dist = (diff ** 2).sum(dim=-1)   # [B, L]
        point_loss = sq_dist.mean()         # scalar
        print('added mre loss') ##normalise, detach helps it not normlaize through the denominator so this will remain able to learn from
        nll_norm   = nll_img / (nll_img.detach() + 1e-8)
        point_norm = point_loss / (point_loss.detach() + 1e-8)
        return nll_norm*(1-gamma)+(point_norm*gamma)
    else:
        return nll_img / (nll_img.detach() + 1e-8)

nll_across_batch_OAImrediff.calls = 0

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
        elif main_loss_str.split('_')[-1]== 'cosinelandmarkvectorOAI':
            self.addclass='cosinelandmarkvectorOAI'
        elif main_loss_str.split('_')[-1]== 'OAIanglediff':
            self.addclass='anglediff'
        elif main_loss_str.split('_')[-1]== 'OAImrediff':
            self.addclass='mrediff'
        else:
            self.addclass=False
        
        print('loss', self.addclass)

    def forward(self, x, target, model,gamma, cfg=None, pred_alphas=None, target_alphas=None,class_output=None,class_target=None,pred_fhc=None,  target_fhc=None):
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
        elif self.addclass=='cosinelandmarkvectorOAI':
            loss = self.main_loss(x, target,gamma) + self.lam*l2
        elif self.addclass==False:
            loss = self.main_loss(x, target) + self.lam*l2
        elif 'diff' in self.addclass:
            loss = self.main_loss(x, target,gamma, cfg) + self.lam*l2
        else:
            loss = self.main_loss(x, target) + self.lam*l2

        return loss
    