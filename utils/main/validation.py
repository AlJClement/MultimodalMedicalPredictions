import torch.nn as nn
import torch
from torch.autograd import Variable
import datetime
from .model_init import model_init
from .loss import *
from .evaluation_helper import evaluation_helper
from .evaluation_helper import *
import sys
import os
import pathlib
target_path = pathlib.Path(os.path.abspath(__file__)).parents[1]
sys.path.append(target_path)
import visualisations
from visualisations import visuals
from .evaluation_helper import evaluation_helper
import pandas as pd
from .comparison_metrics import *
from .comparison_metrics import fhc
import torch.nn.functional as F
from preprocessing.augmentation import Augmentation
class validation():
    def __init__(self, cfg, logger, net, l2_reg=True, save_img=True):
        self.dcm_dir = cfg.INPUT_PATHS.DCMS
        self.dataset_name = cfg.INPUT_PATHS.DATASET_NAME
        self.dataset_type = cfg.DATASET.ANNOTATION_TYPE
        self.num_landmarks = cfg.DATASET.NUM_LANDMARKS
        self.max_epochs = cfg.TRAIN.EPOCHS
        self.img_extension = cfg.DATASET.IMAGE_EXT
        self.cfg=cfg
        self.img_size =cfg.DATASET.CACHED_IMAGE_SIZE

        self.save_heatmap_asdcms=cfg.TRAIN.SAVE_VAL_DCM

        self.save_all_landmarks = cfg.TEST.SHOW_ALL_LANDMARKS
        self.save_txt = cfg.TEST.SAVE_TXT
        self.save_heatmap = cfg.TEST.SAVE_HEATMAPS_ALONE
        self.save_heatmap_as_np = cfg.TEST.SAVE_HEATMAPS_NP

        self.save_img_path = cfg.OUTPUT_PATH +'/validation'
        if self.save_txt == True:
            if not os.path.isdir(self.save_img_path+'/'+'txt/'):
                os.makedirs(self.save_img_path+'/'+'txt/')
    
        if self.save_heatmap_as_np == True:
            if not os.path.isdir(self.save_img_path+'/np/'):
                os.makedirs(self.save_img_path+'/np/')
        self.net = net
        self.logger = logger
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

        if (cfg.TRAIN.LOSS).split('_')[-1]=='walphafhc':
            self.add_alphafhc_loss = True
            self.gamma = cfg.TRAIN.GAMMA
        else:
            self.add_alphafhc_loss = False

        if (cfg.TRAIN.LOSS).split('_')[-1]=='walpha':
            self.add_alpha_loss = True
            self.gamma = cfg.TRAIN.GAMMA
        else:
            self.add_alpha_loss = False

        if (cfg.TRAIN.LOSS).split('_')[-1]=='cosinelandmarkvector':
            self.add_landmark_loss = True
            self.gamma = cfg.TRAIN.GAMMA
        else:
            self.add_landmark_loss = False        

        self.class_calculation = graf_angle_calc()
        self.gamma = cfg.TRAIN.GAMMA
        self.bs = cfg.TRAIN.BATCH_SIZE
        self.lr = cfg.TRAIN.LR
        self.momentum = 0.99
        if net==None:
            self.optimizer=None
        else:
            self.optimizer = self._get_optimizer(self.net)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.combine_graf_fhc=cfg.TRAIN.COMBINE_GRAF_FHC

        self.evaluation = evaluation_helper
        self.save_img = save_img

        self.outputpath=cfg.OUTPUT_PATH
        if os.path.exists(self.outputpath)==False:
            os.mkdir(self.outputpath)
        self.pixel_size = torch.tensor(cfg.DATASET.PIXEL_SIZE).to(cfg.MODEL.DEVICE)

        self.comparison_metrics=cfg.TEST.COMPARISON_METRICS
        self.sdr_thresholds = cfg.TEST.SDR_THRESHOLD
        self.sdr_units = cfg.TEST.SDR_UNITS
        self.num_output_channels = cfg.MODEL.OUT_CHANNELS

        self.fhc_calc = fhc()

    
    def _get_optimizer(self,net):
        optim = torch.optim.SGD(net.parameters(), lr = self.lr, momentum=self.momentum)
        return optim
    
    
    def compare_metrics(self, id, pred, pred_map, true, true_map, pixelsize, orig_size):
        df = pd.DataFrame({"ID": [id]})
        for metric in self.comparison_metrics:
            func = eval(metric)
            output = func(pred, pred_map, true, true_map, pixelsize)
            for i in output:
                df[i[0]]=[i[1]]

        return df
    
    def comparison_summary(self, df):
        summary_ls = []
        arr_mre = np.array([])
        for key in df.keys():
            try:
                mean_val=df[key].mean().round(2)
                summary_ls.append([key, mean_val])
                if 'landmark radial error' in key: 
                    arr_mre = np.append(arr_mre,mean_val)
            except:
                pass
        
        index_to_remove = 4  # remove element at index 2 (value 30)
        arr_mre_nolab = np.delete(arr_mre,index_to_remove)

        MRE = np.mean(arr_mre).round(2)
        MRE_std = np.std(arr_mre).round(2)
        MRE_nolabrum= np.mean(arr_mre_nolab).round(2)
        MRE_nolabrum_std = np.std(arr_mre_nolab).round(2)

        return summary_ls, [MRE, MRE_std, MRE_nolabrum, MRE_nolabrum_std]
    
    
    def alpha_thresholds(self, df, thresholds=[1,2,5,10]):
        #this function calculates the different percentages of angle difference that lays in these thresholds
        df_alpha_diff = df['difference alpha']
        df_pred_class = df['class pred']
        df_true_class = df['class true']

        if type(df_alpha_diff) == pd.Series:
            alpha_diff = df_alpha_diff.to_numpy()
            pred_class = df_pred_class.to_numpy()
            true_class = df_true_class.to_numpy()
        

        np_agree= np.where(true_class == pred_class, 1.0, 0.0)
        np_disagree= np.where(true_class != pred_class, 1.0, 0.0)


        alpha_thresh = []
        for threshold in thresholds:            
            filter = np.where(alpha_diff < threshold, 1.0, 0.0)
            percent = 100 * np.sum(filter) / np.size(alpha_diff)

            percent_agree = 100 *np.sum(np_agree*filter)/np.size(np_agree)
            percent_disagree = 100*np.sum(np_disagree*filter)/np.size(np_disagree)

            alpha_thresh.append([percent,percent_agree.round(2),percent_disagree.round(2)])
            
            
        txt = ""
        for val in alpha_thresh:
            txt += "{:.2f}%, (agreeance {:.2f}%, disagreeance {:.2f}%)\t".format(val[0],val[1],val[2])
    
        return txt 
    
    def resize_backto_original(self, pred_map, target_map, orig_size):
        #resize images
        orig_size = orig_size.to('cpu').numpy()
        pred = pred_map.detach().cpu().numpy()
        target = target_map.detach().cpu().numpy()
        augmenter = Augmentation(self.cfg)
        target = augmenter.reverse_downsample_and_pad(orig_size, target)
        pred = augmenter.reverse_downsample_and_pad(orig_size, pred)

        return pred, target
    
    def get_combined_agreement(self, df, name_pred, name_true, groups):
        #if they agree yes, if they dont make it the more conservative of the two
        # get classes as binary 
        i=0
        for group in groups:
            #convert class to graf class as abnormal/normal based on grouping    
            new_class =['n','a']
            if len(group)==2:
                if i == 0:
                    a,b=0,1
                else:
                    a,b=1,0
                df['graf class pred']=df['class pred'].apply(lambda x: new_class[a] if x in group else new_class[b])
                df['graf class true']=df['class true'].apply(lambda x: new_class[a] if x in group else new_class[b])
            else:
                pass

            i=i+1

        #check fhc == graf
        df['true agree']=(df['graf class true'] == df['fhc class true'])
        df[name_true] = np.where(df['true agree'] == True, df['graf class true'], 'a')
        df['pred agree']=(df['graf class pred'] == df['fhc class pred'])
        df[name_pred] = np.where(df['pred agree'] == True, df['graf class pred'], 'a')

        return df
    
    def val_meta(self, dataloader, epoch):
        self.net.eval()
        total_loss = 0
        batches = 0

        with torch.no_grad():  # So no gradients accumulate#
            comparison_df = pd.DataFrame([])
            EH=evaluation_helper.evaluation_helper()


            for batch_idx, (data, target, landmarks, meta, id, orig_size, orig_img)in enumerate(dataloader):                
                batches += 1
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                meta_data = Variable(meta).to(self.device)
                
                # get prediction
                pred = self.net(data,meta_data)      

                ##get predicted values from prediction heatmap for loss if needed
                if self.add_class_loss==True or self.add_alpha_loss == True or self.add_alphafhc_loss==True:
                        pred_alphas, pred_classes,target_alphas, target_classes = self.class_calculation.get_class_from_output(pred,target,self.pixel_size)
                if self.add_alphafhc_loss==True:
                    pred_fhc, target_fhc = self.fhc_calc.get_fhc_batches(pred,target,self.pixel_size)

                ## calculate loss
                if self.l2_reg==True:
                    if self.add_class_loss==True:
                        loss = self.loss_func(pred.to(self.device), target.to(self.device), self.net,self.gamma, pred_alphas, target_alphas,pred_classes, target_classes)
                    elif self.add_alphafhc_loss== True:
                        loss = self.loss_func(pred.to(self.device), target.to(self.device), self.net, self.gamma, pred_alphas, target_alphas, pred_classes, target_classes, pred_fhc, target_fhc)
                    elif self.add_landmark_loss== True:
                        loss = self.loss_func(pred.to(self.device), target.to(self.device), self.net, self.gamma)
                    elif self.add_alpha_loss== True:
                        loss = self.loss_func(pred.to(self.device), target.to(self.device), self.net, self.gamma, pred_alphas, target_alphas)
                    else:
                        loss = self.loss_func(pred.to(self.device), target.to(self.device), self.net, self.gamma)
                else:
                    raise ValueError('only implemented for l2 reg right now')
                
                ## Update loss
                total_loss += loss

                ## Save images for validation
                if self.save_img == True:
                    for i in range(self.bs):
                        #get original image and resize predictions to image size
                        _data = orig_img[i].numpy()
                        _pred, _target =self.resize_backto_original(pred[i], target[i], orig_size[i])

                        _target_points,_predicted_points=EH.get_landmarks(_pred, _target, pixels_sizes=self.pixel_size.to('cpu'))
                        _target_points,_predicted_points= _target_points.squeeze(0),_predicted_points.squeeze(0)

                        if self.save_img  == True:
                            print('saving validation img:', id[i])
                            #print(self.pixel_size[0])
                            validation_dir=self.outputpath+'/validation'
                            if os.path.exists(validation_dir)==False:
                                os.mkdir(validation_dir)


                            visuals(validation_dir+'/'+id[i], self.pixel_size[0]).heatmaps(_data ,_pred,_target_points, _predicted_points)
                            visuals(validation_dir+'/heatmap_'+id[i], self.pixel_size[0]).heatmaps(_data ,_pred,_target_points, _predicted_points, w_landmarks=False)

                            if self.save_heatmap_asdcms == True:
                                out_dcm_dir = self.outputpath+'/as_dcms' 
                                if os.path.exists(out_dcm_dir)==False:
                                    os.mkdir(out_dcm_dir)

                                dcm_loc = self.dcm_dir +'/'+ id[i][:-1]+'_'+id[i][-1]+'.dcm'
                                visuals(out_dcm_dir+'/'+id[i], self.pixel_size[0]).heatmaps(_data ,_pred,_target_points, _predicted_points, as_dcm=True, dcm_loc=dcm_loc)
                                visuals(out_dcm_dir+'/heatmap_'+id[i], self.pixel_size[0]).heatmaps(_data ,_pred,_target_points, _predicted_points,w_landmarks=False, as_dcm=True, dcm_loc=dcm_loc)

                        if self.save_txt == True:
                            visuals(validation_dir+'/'+'txt/'+id[i],self.pixel_size, self.cfg).save_astxt(_data ,_pred,_target_points, _predicted_points,orig_size[i])
                    
                        if self.save_heatmap == True:
                            visuals(validation_dir+'/heatmap_only_'+id[i], self.pixel_size[0], self.cfg).heatmaps(_data ,_pred,_target_points, _predicted_points, w_landmarks=False, all_landmarks=self.save_all_landmarks, with_img = False)

                        if self.save_heatmap_as_np == True:
                            visuals(validation_dir+'/np/numpy_heatmaps_'+id[i],self.pixel_size, self.cfg).save_np(pred[i])
                            
                        #add to comparison df
                        print('Comparison metrics for', id[i])
                        id_metric_df = self.compare_metrics(id[i], _predicted_points, _pred, _target_points, _target,self.pixel_size, orig_size[i])

                        if comparison_df.empty == True:
                            comparison_df = id_metric_df
                        else:
                            comparison_df = comparison_df._append(id_metric_df, ignore_index=True)

            
        comparison_df.to_csv(self.outputpath+'/comparison_metrics.csv')
        print('Saving Results to comparison_metrics.csv')
        av_loss = total_loss / batches
        av_loss = av_loss.cpu().detach().numpy()
        print('Validation Set Average Loss: {:.4f}'.format(av_loss,  flush=True))

        #Get mean values from comparison summary ls, landmark metrics
        comparsion_summary_ls, MRE = self.comparison_summary(comparison_df)
        self.logger.info("MEAN VALUES (pix): {}".format(comparsion_summary_ls))
        self.logger.info("MRE: {} +/- {} pix".format(MRE[0], MRE[1]))
        self.logger.info("MRE no labrum: {} +/- {} pix".format(MRE[2], MRE[3]))

        if 'graf_angle_calc().graf_class_comparison' in self.cfg.TEST.COMPARISON_METRICS:
            alpha_thresh_percentages=self.alpha_thresholds(comparison_df)
            self.logger.info("Alpha Thresholds: {}".format(alpha_thresh_percentages))

            #from df get classification metrics TP, TN, FN, FP for graf
            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'class pred', 'class true', self.outputpath,loc='validation')._get_metrics(group=True,groups=[('i'),('ii','iii/iv')])
            self.logger.info("Class Agreement GRAF: {}".format(class_agreement))

            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'class pred', 'class true', self.outputpath, loc='validation')._get_metrics(group=True,groups=[('i','ii'),('iii/iv')])
            self.logger.info("Class Agreement GRAF: {}".format(class_agreement))

        if self.combine_graf_fhc==True:
        # #add fhc cols for normal and abnormal (n and a)
            comparison_df['fhc class pred']=comparison_df['fhc pred'].apply(lambda x: 'n' if x > .50 else 'a')
            comparison_df['fhc class true']=comparison_df['fhc true'].apply(lambda x: 'n' if x > .50 else 'a')

            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'fhc class pred', 'fhc class true',  self.outputpath, loc='validation')._get_metrics(group=True,groups=[('n'),('a')])
            self.logger.info("Class Agreement FHC: {}".format(class_agreement))


            ## Concensus of FHC and Graf
            comparison_df = self.get_combined_agreement(comparison_df,'graf&fhc pred i_ii&iii&iv', 'graf&fhc true i_ii&iii&iv', groups=[('i'),('ii','iii/iv')])
            comparison_df = self.get_combined_agreement(comparison_df,'graf&fhc pred i&ii_iii&iv', 'graf&fhc true i&ii_iii&iv', groups=[('i','ii'),('iii/iv')])
            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'graf&fhc pred i_ii&iii&iv', 'graf&fhc true i_ii&iii&iv',  self.outputpath, loc='validation')._get_metrics(group=True,groups=[('i'),('ii','iii/iv')])
            self.logger.info("Class Agreement i vs ii/iii/iv GRAF&FHC: {}".format(class_agreement))
            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'graf&fhc pred i&ii_iii&iv', 'graf&fhc true i&ii_iii&iv', self.outputpath,  loc='validation')._get_metrics(group=True,groups=[('i','ii'),('iii/iv')])
            self.logger.info("Class Agreement i/ii vs iii/iv GRAF&FHC: {}".format(class_agreement))

        sdr_summary = ""
        if self.dataset_type == 'LANDMARKS':
            #calculate SDR
            try:
                for i in range(self.num_landmarks):
                    col= 'landmark radial error p'+str(i+1)
                    sdr_stats, txt = landmark_overall_metrics(self.pixel_size, self.sdr_units).get_sdr_statistics(comparison_df[col], self.sdr_thresholds)
                    self.logger.info("{} for {}".format(txt, col))

                    try:
                        #print(sdr_summary)
                        sdr_summary=np.concatenate((sdr_summary,np.array([sdr_stats])), axis=0)
                    except:
                        sdr_summary = np.array([sdr_stats])
                
                sdr_summary_nolab = np.delete(sdr_summary, 4, axis=0)
                sdr_summary_nolab = sdr_summary_nolab.T.mean(axis=1)

                sdr_summary = sdr_summary.T.mean(axis=1)

                #get mean
                self.logger.info("SDR all landmarks: {},{},{},{}".format(round(sdr_summary[0],2),round(sdr_summary[1],2),round(sdr_summary[2],2),round(sdr_summary[3],2)))
                self.logger.info("SDR all landmarks (no labrum): {},{},{},{}".format(round(sdr_summary_nolab[0],2),round(sdr_summary_nolab[1],2),round(sdr_summary_nolab[2],2),round(sdr_summary_nolab[3],2)))


            except:
                raise ValueError('Check Landmark radial errors are calcuated')

        #plot angles pred vs angles 
        if 'graf_angle_calc().graf_class_comparison' in self.cfg.TEST.COMPARISON_METRICS:
            visualisations.comparison(self.dataset_name,self.outputpath,'graf').true_vs_pred_scatter(comparison_df['alpha pred'].to_numpy(),comparison_df['alpha true'].to_numpy(),loc='validation')
            visualisations.comparison(self.dataset_name,self.outputpath,'fhc').true_vs_pred_scatter(comparison_df['fhc pred'].to_numpy(),comparison_df['fhc true'].to_numpy(),loc='validation')


        return av_loss
