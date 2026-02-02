import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.transform import resize
import argparse
from support import helper
from torch.utils.data import DataLoader
from preprocessing import dataloader
from main import test
import numpy as np
import os
import torch
torch.cuda.empty_cache() 
from torch.autograd import Variable
from main.model_init import model_init
from main.comparison_metrics import fhc, graf_angle_calc
from main.evaluation_helper import evaluation_helper

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import math

def channels_thresholded(output):
    #theshold then add all channels together
    for c in range(output.shape[0]):
        try:
            compressed_channels = compressed_channels+output[c]
        except:
            compressed_channels = output[c]
    return compressed_channels


class ModelWrapper(torch.nn.Module):
    ##to allow for gradcam
    def __init__(self, model, dummy_meta):
        super().__init__()
        self.model = model
        self.dummy_meta = dummy_meta

    def forward(self, x):
        return self.model(x, self.dummy_meta)

# Custom target for one landmark's heatmap
class LandmarkHeatmapTarget:
    def __init__(self, landmark_index):
        self.landmark_index = landmark_index

    def __call__(self, output):
        # print("Model output shape:", output.shape)
        # adjust indexing based on output shape
        if output.dim() == 4:
            # Expected shape: [B, C, H, W]
            return output[:, self.landmark_index, :, :].sum()
        elif output.dim() == 3:
            # Could be [C, H, W] or [B, H, W]
            # Try to index channels dimension accordingly
            return output[self.landmark_index, :, :].sum()
        else:
            raise ValueError(f"Unexpected output tensor shape: {output.shape}")

def load_network(cfg,model_path):
    model = model_init(cfg).get_net_from_conf(get_net_info=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_features_hook(name, feature_maps):
    def hook(model, input, output):
        feature_maps[name] = output.detach()
    return hook

def run_plot_model_feats(cfg,dataloader, modeltype,folder):
    #get weights 
    plt_gradcam=True
    device = 'cuda:0'

    if modeltype=='unet':
        modelpath= '/home/scratch/allent/MultimodalMedicalPredictions/'+folder+'/model:1/_model_run:1_idx.pth'
    elif modeltype== 'hrnet':
        modelpath = '/home/scratch/allent/MultimodalMedicalPredictions/'+folder+'/model:1/_model_run:1_idx.pth'
    else:
        raise ValueError('model type incorrect')
    # state_dict = torch.load(modelpath, map_location=torch.device(device))

    net = load_network(cfg,modelpath)
    
    # place to store/capture feature maps
    feature_maps = {}

    # find layers print to see what the names are
    for name, module in net.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            print(f"{name:30} → {type(module).__name__:20} | Param: {param_name:10} | Shape: {tuple(param.shape)}")

    #hook to first layer
    # layer_name='layer1'
    if modeltype=='unet':
        net.unet.encoder.layer1.register_forward_hook(get_features_hook('layer1',feature_maps))
        net.unet.encoder.layer2.register_forward_hook(get_features_hook('layer2',feature_maps))
        net.unet.encoder.layer3.register_forward_hook(get_features_hook('layer3',feature_maps))
        net.unet.encoder.layer4.register_forward_hook(get_features_hook('layer4',feature_maps))
    else:
        net.backbone.stage2[0].branches[1][3].conv2.register_forward_hook(get_features_hook('stage2',feature_maps))   #64x64
        net.backbone.stage3[2].branches[2][3].conv2.register_forward_hook(get_features_hook('stage3',feature_maps))   #128 by 128
        net.backbone.stage4[2].branches[3][3].conv2.register_forward_hook(get_features_hook('stage4_deepestbranch',feature_maps))         # 256x256 Deepest branch, final conv
        net.backbone.stage4[2].fuse_layers[0][3][0].register_forward_hook(get_features_hook('stage4_muliscalefusion', feature_maps))  # Fusion of multiscale features
        net.get_pose_net.branches[3][3].conv2.register_forward_hook(get_features_hook('outputhighresblock', feature_maps))       # Output of the final high-res block
        net.get_pose_net.fuse_layers[3][0][2][0].register_forward_hook(get_features_hook('finalstagefusion', feature_maps))    # Final stage fusion

        ###


    # Plot features from each data loader pred
    for batch_idx, (data, target, landmarks, meta, id, orig_size, orig_img)in enumerate(dataloader):
        
        data, target = Variable(data).to(device), Variable(target).to(device)
        meta_data = Variable(meta).to(device)
        orig_size = Variable(orig_size).to(device)

        if modeltype == 'unet':
            with torch.no_grad():
                #get prediction
                pred = net(data, meta)
        else:
            pred = net(data, meta)

        for layer_name, activation in feature_maps.items():
            act = activation[0]  # First sample in batch
            print(act.shape)

            # Compute average activation per channel
            channel_means = act.mean(dim=(1, 2))  # shape [C]
            num_feats = 16
            subplot_feats = 4
            # ** Get top channels based on means **
            # print(len(act)) 
            # SORTED IN DECENDING ORDER ALREADT
            topk = torch.topk(channel_means, num_feats)
            fig, axes = plt.subplots(subplot_feats, subplot_feats, figsize=(12, 12))  # 4 rows, 4 columns

            for i in range(num_feats):
                ch_idx = topk.indices[i]
                row, col = divmod(i, 4)
                axes[row, col].imshow(act[ch_idx].cpu(), cmap='cividis')#,alpha=0.5)
                axes[row, col].set_title(f'Ch {ch_idx.item()}')
                axes[row, col].axis('off')
                            
                # if modeltype == 'hrnet':
                #     #only get 1 channel
                #     axes[row, col].imshow(data.squeeze(0).squeeze(0).cpu()[0], cmap='gray', alpha=0.5)
                # else:
                #     #plot heatmap prediction
                #     axes[row, col].imshow(data.squeeze(0).squeeze(0).cpu(), cmap='gray', alpha=0.5)
            

            # plt.show()
            output_folder = 'feats_QC'+ f'/{folder}'
            os.makedirs(f'./{output_folder}/', exist_ok=True)
            plt.savefig(f'./{output_folder}/'+f'{layer_name}_{id[0]}.png')
            plt.title('shape: '+ str(act.shape[0])+'_'+str(act.shape[0]))
            plt.close()

            #from torchvis import make_dot
            #make_dot(pred, params=dict(net.named_parameters())).render("unetpp", format="png")

                    
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # 3 rows, 3 cols grid



        ##### gracam
        #loop through layers
        if modeltype=='unet':
        #target_layer = net.unet.decoder.blocks.x_0_1.conv2
            _target_layers = [net.unet.encoder.layer3, net.unet.encoder.layer2, net.unet.encoder.layer1 , net.unet.encoder.layer4, net.unet.decoder.blocks.x_0_0.conv2, net.unet.decoder.blocks.x_0_4.conv2]
            names = ['layer3','layer2','layer1','layer4', 'decode_layer1','decode_layer4']
        # x_i_j  where:
        # - i = depth (usually corresponds to encoder level)
        # - j = stage in the decoder (number of decoding steps)
        else:
            _target_layers =  [net.backbone.stage2[0].branches[1][3].conv2, net.backbone.stage3[2].branches[2][3].conv2, net.backbone.stage4[2].branches[3][3].conv2] #net.backbone.stage4[2].fuse_layers[0][3][0]]#, net.get_pose_net.fuse_layers[3][0][2][0]] #4th branch and 4th block
            names = ['stage2','stage3','stage4_deepestbranch']#,'stage4_multiscalefusion']#,'finalstagefusion']

        cam_norm_all = None
        for cam_TYPE in ['gradcam', 'gradcampluspls']:

            for c in range(len(_target_layers)):
                fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # 3 rows, 3 cols grid

                print(c)
                target_layer = _target_layers[c]
                name = names[c]
                print(name)
                for i in range(9):
                    #get prediction        
                    ax = axes[i // 3, i % 3]

                    if i == 0:
                        if modeltype == 'hrnet':
                            #only get 1 channel
                            ax.imshow(data.squeeze(0).squeeze(0).cpu()[0], cmap='gray')
                        else:
                            #plot heatmap prediction
                            ax.imshow(data.squeeze(0).squeeze(0).cpu(), cmap='gray')

                        ax.set_title(f'Heatmap Prediciton')
                        _output = channels_thresholded(pred.detach().cpu().squeeze(0).numpy())
                        ax.imshow(_output, cmap='inferno', alpha = 0.7)
                        ax.axis('off')
                    
                    elif i == 1:
                        predicted_points=evaluation_helper.get_hottest_points(data.squeeze(0).squeeze(0).cpu(),pred)
                        predicted_points = predicted_points.detach().cpu().squeeze(0).numpy()
                        # target_points = target_points.cpu().detach().numpy().cpu()
                        #plot landmarks and ground truths
                        ax.set_title(f'Landmark Prediciton')
                        if modeltype == 'hrnet':
                            #only get 1 channel
                            ax.imshow(data.squeeze(0).squeeze(0).cpu()[0], cmap='gray')
                        else:
                            #plot heatmap prediction
                            ax.imshow(data.squeeze(0).squeeze(0).cpu(), cmap='gray')
                        #add landmarks
                        # ax.scatter(target_points[:, 0], target_points[:, 1], color='lime', s=5)
                        ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=7)
                        # fhc_pred, fhc_true = fhc.fhc().get_fhc(predicted_points,_output,target_points,target,pixelsize)
                        # fhc_pred, fhc_true = fhc_pred[1]*100, fhc_true[1]*100
                        # alpha_true, alpha_pred = round(graf_angle_calc().calculate_alpha(target_points),1), round(graf_angle_calc().calculate_alpha(predicted_points),1)

                        # ax.text(0.02, 0.98,f"FHC = {fhc_true:.1f}%\n α = {alpha_true:.1f}°", 
                        #             transform=ax.transAxes, fontsize=10, verticalalignment='top',bbox=dict(facecolor='green', alpha=0.6, edgecolor='none'))
                        # ax.text(0.02, 0.80,f"FHC = {fhc_pred:.1f}%\n α = {alpha_pred:.1f}°",
                        #             transform=ax.transAxes, fontsize=10, verticalalignment='top',bbox=dict(facecolor='red', alpha=0.6, edgecolor='none'))
                    else:
                        landmark_index = i-2
                        target = [LandmarkHeatmapTarget(landmark_index)]

                        wrapped_model = ModelWrapper(net, meta_data).to(device)
                        wrapped_model.eval()

                    
                        if cam_TYPE == 'gradcamplusplus':
                            cam = GradCAMPlusPlus(model=wrapped_model, target_layers=[target_layer])
                        else:
                            cam = GradCAM(model=wrapped_model, target_layers=[target_layer])

                        # input_tensor = torch.tensor(data).clone().float().to(device).requires_grad_() 
                        input_tensor = data.clone().detach().to(device).requires_grad_()

                        cam_out = cam(input_tensor=input_tensor, targets=target)[0]
                        cam_norm = (cam_out - cam_out.min()) / (cam_out.max() - cam_out.min())
                        

                        if any(math.isnan(x) for row in cam_norm for x in row):
                            print('nan')
                            pass
                        else:
                            try:    
                                if cam_norm_all == None:
                                    cam_norm_all = cam_norm
                                else:
                                    cam_norm_all = cam_norm_all+cam_norm
                            except:
                                cam_norm_all = cam_norm_all+cam_norm


                        if modeltype == 'hrnet':
                            #only get 1 channel
                            ax.imshow(data.squeeze(0).squeeze(0).cpu()[0], cmap='gray', alpha=0.5)
                        else:
                            #plot heatmap prediction
                            ax.imshow(data.squeeze(0).squeeze(0).cpu(), cmap='gray', alpha=0.5)
                        
                        ax.imshow(cam_norm, cmap='jet', alpha=0.5)
                        ax.axis('off')
                        ax.set_title(f'GradCAM Landmark {landmark_index}')
                        ax.scatter(predicted_points[i-2, 0], predicted_points[i-2, 1], color='red', s=7)
                        

                plt.tight_layout()
                os.makedirs(f'./{output_folder}/', exist_ok=True)
                plt.savefig(f'./{output_folder}/'+cam_TYPE+f'_all_landmarks_{id[0]}_{name}.png', bbox_inches='tight', pad_inches=0)
                plt.close()
                

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 rows, 3 cols grid
                if modeltype == 'hrnet':
                    #only get 1 channel
                    axes[0].imshow(data.squeeze(0).squeeze(0).cpu()[0], cmap='gray')
                    axes[1].imshow(data.squeeze(0).squeeze(0).cpu()[0], cmap='gray')
                    axes[2].imshow(data.squeeze(0).squeeze(0).cpu()[0], cmap='gray', alpha=0.5)
                else:
                    #plot heatmap prediction
                    axes[0].imshow(data.squeeze(0).squeeze(0).cpu(), cmap='gray')
                    axes[1].imshow(data.squeeze(0).squeeze(0).cpu(), cmap='gray')
                    axes[2].imshow(data.squeeze(0).squeeze(0).cpu(), cmap='gray', alpha=0.5)
                
                axes[0].set_title(f'Heatmap Prediciton')
                axes[0].imshow(_output, cmap='inferno', alpha = 0.7)
                axes[0].axis('off')

                axes[1].set_title(f'Landmark Prediciton')
                axes[1].scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=7)    
                axes[1].axis('off')

                #add landmarks
                if isinstance(cam_norm_all, np.ndarray):
                    axes[2].imshow(cam_norm_all, cmap='jet', alpha=0.5)
                    axes[2].axis('off')
                    axes[2].set_title(f'GradCAM Landmark')

                plt.tight_layout()
                os.makedirs(f'./{output_folder}/', exist_ok=True)
                plt.savefig(f'./{output_folder}/'+cam_TYPE+f'_gradcamALL_{id[0]}_{name}.png', bbox_inches='tight', pad_inches=0)
                plt.close()

                cam_norm_all = None
        


            

def parse_args():
    parser = argparse.ArgumentParser(description='Train a network to detect landmarks')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)
    
    args = parser.parse_args()
    return args

def main():
    cfg_name = 'ddh_denoise_journalpaper'
    folder = 'output_sigma1_10epochLR005_7landmarks_freeze'
    model_type = 'unet'

    cfg_name = 'ddh_denoise_hrnet'
    folder = 'output_HRNET_sigma1_10epochLR005_7landmarks_freeze'
    model_type = 'hrnet'

    # print the arguments into the log
    help = helper(cfg_name, 'test')
    cfg = help._get_cfg()

    #preprocess data (put into a numpy array)
    test_dataset=dataloader(cfg,'testing', subset=5)
    help._dataset_shape(test_dataset)

    #load data into data loader (imports all data into a dataloader)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, drop_last = True)

    #run_plot_model_feats(cfg,test_dataloader,'unet')
    run_plot_model_feats(cfg,test_dataloader,model_type, folder)


if __name__ == '__main__':
    main()


# for name, param in state_dict.items():
#     print(f"{name:60} {tuple(param.shape)}")
#     try:
#         weights = state_dict[name]  # adjust key
#         w = weights.squeeze(-1).squeeze(-1)  
#         resampled = resize(w, (512, 512), anti_aliasing=True)

#         print(resampled.shape)
#         if 'weight' in name:
#             plt.imshow(resampled, aspect='auto', cmap='viridis')
#             plt.savefig('./weight_QC/'+name+'.png')

#     except:
#         pass
