import torch.nn as nn
import torch
from torch.autograd import Variable
import datetime
from .model_init import model_init
from .loss import *
import sys
import pathlib
import os
from .evaluation_helper import evaluation_helper
target_path = pathlib.Path(os.path.abspath(__file__)).parents[1]
sys.path.append(target_path)
from .model_init import model_init
import visualisations
from visualisations import visuals
import os
import pandas as pd
from .comparison_metrics import *
import torch
torch.cuda.empty_cache() 
from .validation import validation

class test():
    def __init__(self, cfg, logger):
        self.combine_graf_fhc=True
        self.dcm_dir = cfg.INPUT_PATHS.DCMS
        self.validation = validation(cfg,logger,net=None)
        self.cfg=cfg
        self.img_size = cfg.DATASET.CACHED_IMAGE_SIZE
        self.logger = logger
        self.dataset_type = cfg.DATASET.ANNOTATION_TYPE
        self.num_landmarks = cfg.DATASET.NUM_LANDMARKS

        self.save_asdcms = cfg.TEST.SAVE_HEATMAPS_ASDCM
        self.save_txt=cfg.TEST.SAVE_TXT
        self.save_heatmap_land_img= cfg.TEST.SAVE_HEATMAPS_LANDMARKS_IMG
        self.save_img_landmarks_predandtrue = cfg.TEST.SAVE_IMG_LANG_PREDANDTRUE
        self.save_heatmap = cfg.TEST.SAVE_HEATMAPS_ALONE
        self.save_heatmap_as_np = cfg.TEST.SAVE_HEATMAPS_NP
        self.save_all_landmarks = cfg.TEST.SHOW_ALL_LANDMARKS

        self.loss_func = eval(cfg.TRAIN.LOSS)
        self.bs = cfg.TRAIN.BATCH_SIZE
        self.lr = cfg.TRAIN.LR
        self.momentum = cfg.TRAIN.MOMENTUM
        self.device = torch.device(cfg.MODEL.DEVICE)

        if cfg.MODEL.DEVICE == 'cuda':
            torch.cuda.empty_cache()

        self.output_path = cfg.OUTPUT_PATH
        self.net = self.load_network(self.output_path+cfg.TEST.NETWORK)

        self.save_img_path = cfg.OUTPUT_PATH +'/test'
        if self.save_txt == True:
            if not os.path.isdir(self.output_path+'/'+'txt/'):
                os.makedirs(self.output_path+'/'+'txt/')
    
        if self.save_heatmap_as_np == True:
            if not os.path.isdir(self.output_path+'/np/'):
                os.makedirs(self.output_path+'/np/')
    
        if not os.path.exists(self.save_img_path):
            os.mkdir(self.save_img_path)
        
        self.pixel_size = torch.tensor(cfg.DATASET.PIXEL_SIZE).to(cfg.MODEL.DEVICE)

        self.comparison_metrics=cfg.TEST.COMPARISON_METRICS
        self.dataset_name = cfg.INPUT_PATHS.DATASET_NAME
        self.sdr_thresholds = cfg.TEST.SDR_THRESHOLD
        self.sdr_units = cfg.TEST.SDR_UNITS

    def load_network(self, model_path):
        model = model_init(self.cfg).get_net_from_conf(get_net_info=False)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    def compare_metrics(self, id, pred, pred_map, true, true_map, pixelsize):
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

        MRE = np.mean(arr_mre).round(2)
        MRE_std = np.std(arr_mre).round(2)

        return summary_ls, [MRE, MRE_std]
    
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
        txt_norm=""
        for val in alpha_thresh:
            txt += "{:.2f}%, (agreeance {:.2f}%, disagreeance {:.2f}%)\t".format(val[0],val[1],val[2])
            txt_norm += "{:.2f}%, (agreeance {:.2f}%, disagreeance {:.2f}%)\t".format(val[0],100*val[1]/val[0],100*val[2]/val[0])
    
        return txt, txt_norm 
    
    def run(self, dataloader):
        comparison_df = pd.DataFrame([])
        for batch_idx, (data, target, meta, id, orig_size) in enumerate(dataloader):
            print(batch_idx)

            data, target = Variable(data).to(self.device), Variable(target).to(self.device)
            meta_data = Variable(meta).to(self.device)
            orig_size = Variable(orig_size).to(self.device)
            
            pred = self.net(data, meta_data)
            target_points,predicted_points=evaluation_helper().get_landmarks(pred, target, pixels_sizes=self.pixel_size)

            #plot and caluclate values for each subject in the batch
            for i in range(self.bs):
                print('Test Image:', id[i])
                
                if self.save_img_landmarks_predandtrue == True:
                    visuals(self.save_img_path+'/'+id[i], self.pixel_size[0], self.cfg).heatmaps(data[i][0], pred[i], target_points[i], predicted_points[i])

                if self.save_heatmap_land_img == True:
                    visuals(self.save_img_path+'/heatmap_'+id[i], self.pixel_size[0], self.cfg).heatmaps(data[i][0], pred[i], target_points[i], predicted_points[i], w_landmarks=False, all_landmarks=self.save_all_landmarks)

                if self.save_asdcms == True:

                    out_dcm_dir = self.save_img_path+'/as_dcms' 
                    if os.path.exists(out_dcm_dir)==False:
                        os.mkdir(out_dcm_dir)

                    dcm_loc = self.dcm_dir +'/'+ id[i][:-1]+'_'+id[i][-1]+'.dcm'

                    if self.save_img_landmarks_predandtrue == True:
                        visuals(out_dcm_dir+'/'+id[i], self.pixel_size[0], self.cfg).heatmaps(data[i][0], pred[i], target_points[i], predicted_points[i], all_landmarks=self.save_all_landmarks, with_img = True, as_dcm=True, dcm_loc=dcm_loc)
                    if self.save_heatmap_land_img == True:
                        visuals(out_dcm_dir+'/heatmap_'+id[i], self.pixel_size[0], self.cfg).heatmaps(data[i][0], pred[i], target_points[i], predicted_points[i],w_landmarks=False,all_landmarks=self.save_all_landmarks, with_img = True, as_dcm=True, dcm_loc=dcm_loc)
                
                if self.save_txt == True:
                    visuals(self.output_path+'/'+'txt/'+id[i],self.pixel_size, self.cfg).save_astxt(data[i][0],predicted_points[i],self.img_size,orig_size[i])
                
                if self.save_heatmap == True:
                    visuals(self.save_img_path+'/heatmap_only_'+id[i], self.pixel_size[0], self.cfg).heatmaps(data[i][0], pred[i], target_points[i], predicted_points[i], w_landmarks=False, all_landmarks=self.save_all_landmarks, with_img = False)

                if self.save_heatmap_as_np == True:
                    visuals(self.output_path+'/np/numpy_heatmaps_'+id[i],self.pixel_size, self.cfg).save_np(pred[i])

                #add to comparison df
                id_metric_df = self.compare_metrics(id[i], predicted_points[i], pred[i], target_points[i], target[i],self.pixel_size)

                print('Alpha for', id[i])
                id_metric_df = self.compare_metrics(id[i], predicted_points[i], pred[i], target_points[i], target[i], self.pixel_size)

                if comparison_df.empty == True:
                    comparison_df = id_metric_df
                else:
                    comparison_df = comparison_df._append(id_metric_df, ignore_index=True)

            t_s=datetime.datetime.now()

        t_e=datetime.datetime.now()
        total_time=t_e-t_s
        print('Time taken for epoch = ', total_time)
        comparison_df.to_csv(self.output_path+'/test/comparison_metrics.csv')
        print('Saving Results to comparison_metrics.csv')

        self.logger.info("---------TEST RESULTS--------")

        #Get mean values from comparison summary ls, landmark metrics
        comparsion_summary_ls, MRE = self.comparison_summary(comparison_df)
        self.logger.info("MEAN VALUES: {}".format(comparsion_summary_ls))
        self.logger.info("MRE: {} +/- {} %".format(MRE[0], MRE[1]))

        alpha_thresh_percentages,alpha_thresh_percentages_normalized=self.alpha_thresholds(comparison_df)
        self.logger.info("Alpha Thresholds: {}".format(alpha_thresh_percentages))
        self.logger.info("Alpha Thresholds Normalized: {}".format(alpha_thresh_percentages_normalized))

        #from df get class agreement metrics TP, TN, FN, FP
        class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'class pred', 'class true',self.output_path)._get_metrics(group=True,groups=[('i'),('ii','iii/iv')])
        self.logger.info("Class Agreement - i vs ii/iii/iv : {}".format(class_agreement[4]))
        self.logger.info("Class Agreement - i vs ii/iii/iv : {}".format(class_agreement[5]))
        self.logger.info("Class Agreement - i vs ii/iii/iv : {}".format(class_agreement[6]))

        class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'class pred', 'class true', self.output_path)._get_metrics(group=True,groups=[('i','ii'),('iii/iv')])

        self.logger.info("Class Agreement - i/ii vs iii/iv : {}".format(class_agreement[4]))
        self.logger.info("Class Agreement - i/ii vs iii/iv : {}".format(class_agreement[5]))
        self.logger.info("Class Agreement - i/ii vs iii/iv : {}".format(class_agreement[6]))


        if self.combine_graf_fhc==True:
        # #add fhc cols for normal and abnormal (n and a)
            comparison_df['fhc class pred']=comparison_df['fhc pred'].apply(lambda x: 'n' if x > .50 else 'a')
            comparison_df['fhc class true']=comparison_df['fhc true'].apply(lambda x: 'n' if x > .50 else 'a')

            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'fhc class pred', 'fhc class true',self.output_path, loc='test')._get_metrics(group=True,groups=[('n'),('a')])
            self.logger.info("Class Agreement FHC: {}".format(class_agreement))


            ## Concensus of FHC and Graf
            comparison_df = self.validation.get_combined_agreement(comparison_df,'graf&fhc pred i_ii&iii&iv', 'graf&fhc true i_ii&iii&iv', groups=[('i'),('ii','iii/iv')])
            comparison_df = self.validation.get_combined_agreement(comparison_df,'graf&fhc pred i&ii_iii&iv', 'graf&fhc true i&ii_iii&iv', groups=[('i','ii'),('iii/iv')])
            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'graf&fhc pred i_ii&iii&iv', 'graf&fhc true i_ii&iii&iv',self.output_path, loc='test')._get_metrics(group=True,groups=[('i'),('ii','iii/iv')])
            self.logger.info("Class Agreement i vs ii/iii/iv GRAF&FHC: {}".format(class_agreement))
            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'graf&fhc pred i&ii_iii&iv', 'graf&fhc true i&ii_iii&iv',self.output_path, loc='test')._get_metrics(group=True,groups=[('i','ii'),('iii/iv')])
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
                
                sdr_summary = sdr_summary.T.mean(axis=1)

                #get mean
                self.logger.info("SDR all landmarks: {},{},{},{}".format(round(sdr_summary[0],2),round(sdr_summary[1],2),round(sdr_summary[2],2),round(sdr_summary[3],2)))

            except:
                raise ValueError('Check Landmark radial errors are calcuated')
        #get mean alpha difference
        self.logger.info('ALPHA MEAN DIFF:{}'.format(round(comparison_df['difference alpha'].mean(),3)))
        self.logger.info('ALPHA ABSOLUTE MEAN DIFF:{}'.format(round(comparison_df['difference alpha'].apply(abs).mean(),3)))

        #plot angles pred vs angles 
        visualisations.comparison(self.dataset_name, self.output_path).true_vs_pred_scatter(comparison_df['alpha pred'].to_numpy(),comparison_df['alpha true'].to_numpy())

        return 
