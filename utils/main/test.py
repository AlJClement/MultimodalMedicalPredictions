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
import torch.nn.functional as F
from preprocessing.augmentation import Augmentation


class test():
    def __init__(self, cfg, logger):
        self.dataset_name = cfg.INPUT_PATHS.DATASET_NAME
        self.combine_graf_fhc=cfg.TRAIN.COMBINE_GRAF_FHC
        self.dcm_dir = cfg.INPUT_PATHS.DCMS
        self.validation = validation(cfg,logger,net=None)
        self.cfg=cfg
        self.img_size = cfg.DATASET.CACHED_IMAGE_SIZE
        self.logger = logger
        self.dataset_type = cfg.DATASET.ANNOTATION_TYPE
        self.num_landmarks = cfg.DATASET.NUM_LANDMARKS
        self.label_dir = cfg.INPUT_PATHS.LABELS

        self.save_asdcms = cfg.TEST.SAVE_HEATMAPS_ASDCM
        self.save_txt=cfg.TEST.SAVE_TXT
        self.save_heatmap_land_img= cfg.TEST.SAVE_HEATMAPS_LANDMARKS_IMG
        self.save_img_landmarks_predandtrue = cfg.TEST.SAVE_IMG_LANG_PREDANDTRUE
        self.save_heatmap = cfg.TEST.SAVE_HEATMAPS_ALONE
        self.save_heatmap_as_np = cfg.TEST.SAVE_HEATMAPS_NP
        self.save_all_landmarks = cfg.TEST.SHOW_ALL_LANDMARKS

        self.loss_func = eval(cfg.TRAIN.LOSS)
        self.bs = cfg.TEST.BATCH_SIZE
        self.lr = cfg.TRAIN.LR
        self.momentum = cfg.TRAIN.MOMENTUM
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.class_cols = cfg.INPUT_PATHS.META_COLS_CLASSES

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
    
    def compare_metrics(self, id, pred, pred_map, true, true_map, pixelsize, orig_size):

        df = pd.DataFrame({"ID": [id]})
        for metric in self.comparison_metrics:
            func = eval(metric)
            if metric == 'fhc().get_fhc_pred':
                fhc_val = true[self.class_cols.index('FHC (%)')+1][0]
                if fhc_val == '>0.50':
                    fhc_true = 0.51
                elif fhc_val == '0.30-0.40':
                    fhc_true = 0.35
                else:
                    fhc_true = float(fhc_val)

                output = [func(pred, pred_map, pixelsize),['fhc true',fhc_true], ['fhc from adam',fhc_val]]
            else:
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
        txt_norm=""
        for val in alpha_thresh:
            txt += "{:.2f}%, (agreeance {:.2f}%, disagreeance {:.2f}%)\t".format(val[0],val[1],val[2])
            txt_norm += "{:.2f}%, (agreeance {:.2f}%, disagreeance {:.2f}%)\t".format(val[0],100*val[1]/val[0],100*val[2]/val[0])
    
        return txt, txt_norm 
    
    def resize_backto_original(self, pred_map, target_map, orig_size):
        #resize images
        orig_size = orig_size.to('cpu').numpy()
        pred = pred_map.detach().cpu().numpy()

        augmenter = Augmentation(self.cfg)
        pred = augmenter.reverse_downsample_and_pad(orig_size, pred)

        if isinstance(target_map[0], str):
            target = target_map[0]
            pass
        else:
            target = target_map.detach().cpu().numpy()
            target = augmenter.reverse_downsample_and_pad(orig_size, target)

        return pred, target
    
    def plot_hka_comparisons(self, pred_l, pred_r, true_hkas, save_name="/hka_comparison.png"):
        """
        pred_l     : (N,) torch tensor or array
        pred_r     : (N,) torch tensor or array
        true_hkas  : (N, 2, 2) torch tensor or array
                    [sample, clinician, (L, R)]
        """
        axis_val = 15

        clin1_L = true_hkas[:, 0, 0]
        clin1_R = true_hkas[:, 0, 1]
        clin2_L = true_hkas[:, 1, 0]
        clin2_R = true_hkas[:, 1, 1]

        df = pd.DataFrame({
            "pred_l": pred_l,
            "pred_r": pred_r,
            "clin1_L": clin1_L,
            "clin1_R": clin1_R,
            "clin2_L": clin2_L,
            "clin2_R": clin2_R,
        })

        plots = [
            ("pred_l", "clin1_L", "Pred L vs Clinician 1 (L)"),
            ("pred_r", "clin1_R", "Pred R vs Clinician 1 (R)"),
            ("pred_l", "clin2_L", "Pred L vs Clinician 2 (L)"),
            ("pred_r", "clin2_R", "Pred R vs Clinician 2 (R)"),
            ("clin1_L", "clin2_L", "Clinician 1 vs 2 (L)"),
            ("clin1_R", "clin2_R", "Clinician 1 vs 2 (R)"),
        ]

        fig, axes = plt.subplots(3, 2, figsize=(12, 20))
        axes = axes.flatten()

        for ax, (xcol, ycol, title) in zip(axes, plots):
            x = df[xcol].values
            y = df[ycol].values

            # mask out NaNs AND restrict to range [-axis_val, axis_val]
            good_mask = (~np.isnan(x)) & (~np.isnan(y)) & \
                        (x >= -axis_val) & (x <= axis_val) & \
                        (y >= -axis_val) & (y <= axis_val)

            x_filt = x[good_mask]
            y_filt = y[good_mask]

            # Debug: how many points remain
            print(f"{title}: {len(x_filt)} points after filtering")

            # scatter the filtered points
            ax.scatter(x_filt, y_filt)
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.set_title(title)
            ax.grid(True, linestyle=":", linewidth=0.5)

            # Calculate PCC only if we have >=2 points and non-constant arrays
            pcc_text = "PCC = n/a"
            if x_filt.size >= 2 and np.nanstd(x_filt) > 0 and np.nanstd(y_filt) > 0:
                pcc = np.corrcoef(x_filt, y_filt)[0, 1]
                pcc_text = f"PCC = {pcc:.2f}"
            else:
                # could be zero or one point, or constant which gives nan
                pcc = np.nan

            ax.text(
                0.95, 0.05, pcc_text,
                transform=ax.transAxes,
                horizontalalignment="right",
                verticalalignment="bottom",
                fontsize=12,
                color="orange",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="orange")
            )

            # identity line based on filtered min/max (fallback to axis_val if nothing)
            if x_filt.size > 0:
                mn = min(x_filt.min(), y_filt.min())
                mx = max(x_filt.max(), y_filt.max())
                if mn != mx:
                    ax.plot([mn, mx], [mn, mx], linestyle="--", color="orange")
            else:
                # no points — optional: draw identity across axis limits
                ax.plot([-axis_val, axis_val], [-axis_val, axis_val], linestyle="--", color="orange")

            ax.set_xlim(-axis_val, axis_val)
            ax.set_ylim(-axis_val, axis_val)
            ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.savefig(self.output_path + save_name, dpi=300)

        # --- Bland-Altman grid with consistent scales across subplots ---
        # First pass: collect ranges to compute global limits
        mean_mins = []
        mean_maxs = []
        diff_extremes = []  # will include diffs, and LoA bounds when available
        counts = []

        for (xcol, ycol, title) in plots:
            x = df[xcol].values
            y = df[ycol].values

            good_mask = (~np.isnan(x)) & (~np.isnan(y)) & \
                        (x >= -axis_val) & (x <= axis_val) & \
                        (y >= -axis_val) & (y <= axis_val)

            x_filt = x[good_mask]
            y_filt = y[good_mask]

            counts.append(len(x_filt))

            if x_filt.size == 0:
                # include a fallback range so empty plots don't break global limits
                mean_mins.append(-axis_val)
                mean_maxs.append(axis_val)
                diff_extremes.extend([-axis_val, axis_val])
                continue

            mean_vals = (x_filt + y_filt) / 2.0
            diff_vals = x_filt - y_filt

            mean_mins.append(mean_vals.min())
            mean_maxs.append(mean_vals.max())

            diff_extremes.append(diff_vals.min())
            diff_extremes.append(diff_vals.max())

            if diff_vals.size >= 2:
                sd = np.std(diff_vals, ddof=1)
                md = np.mean(diff_vals)
                upper = md + 1.96 * sd
                lower = md - 1.96 * sd
                diff_extremes.append(lower)
                diff_extremes.append(upper)

        # Compute global limits with padding (fallback to axis_val if degenerate)
        if len(mean_mins) > 0:
            global_mean_min = min(mean_mins)
            global_mean_max = max(mean_maxs)
        else:
            global_mean_min, global_mean_max = -axis_val, axis_val

        if global_mean_max > global_mean_min:
            global_xpad = 0.05 * (global_mean_max - global_mean_min)
        else:
            global_xpad = 0.5

        global_xmin = global_mean_min - global_xpad
        global_xmax = global_mean_max + global_xpad

        if len(diff_extremes) > 0:
            global_diff_min = min(diff_extremes)
            global_diff_max = max(diff_extremes)
        else:
            global_diff_min, global_diff_max = -axis_val, axis_val

        if global_diff_max > global_diff_min:
            global_ypad = 0.1 * (global_diff_max - global_diff_min)
        else:
            global_ypad = 0.5

        global_ymin = global_diff_min - global_ypad
        global_ymax = global_diff_max + global_ypad

        # Second pass: plotting using the computed global limits
        fig, axes = plt.subplots(3, 2, figsize=(12, 20))
        axes = axes.flatten()

        for ax, (xcol, ycol, title) in zip(axes, plots):
            x = df[xcol].values
            y = df[ycol].values

            good_mask = (~np.isnan(x)) & (~np.isnan(y)) & \
                        (x >= -axis_val) & (x <= axis_val) & \
                        (y >= -axis_val) & (y <= axis_val)

            x_filt = x[good_mask]
            y_filt = y[good_mask]

            # Debug: how many points remain
            print(f"{title}: {len(x_filt)} points after filtering")

            if x_filt.size == 0:
                ax.text(0.5, 0.5, "No data after filtering", transform=ax.transAxes,
                        ha="center", va="center")
                ax.set_title(title)
                ax.set_xlabel("mean")
                ax.set_ylabel("difference (x - y)")
                ax.grid(True, linestyle=":", linewidth=0.5)
                # enforce global limits even for empty panels
                ax.set_xlim(global_xmin, global_xmax)
                ax.set_ylim(global_ymin, global_ymax)
                continue

            mean_vals = (x_filt + y_filt) / 2.0
            diff_vals = x_filt - y_filt

            if diff_vals.size >= 2:
                md = np.mean(diff_vals)
                sd = np.std(diff_vals, ddof=1)
                loa = 1.96 * sd
                upper = md + loa
                lower = md - loa
            else:
                md = float(np.nan)
                sd = float(np.nan)
                loa = float("nan")
                upper = float("nan")
                lower = float("nan")

            ax.scatter(mean_vals, diff_vals, s=20, alpha=0.8)
            ax.set_xlabel("Mean of pair")
            ax.set_ylabel(f"Difference ({xcol} - {ycol})")
            ax.set_title(title)
            ax.grid(True, linestyle=":", linewidth=0.5)

            if not np.isnan(md):
                ax.axhline(md, linestyle="--", linewidth=1.0, label="Mean bias")
                ax.axhline(upper, linestyle="--", linewidth=1.0, label="Upper LoA")
                ax.axhline(lower, linestyle="--", linewidth=1.0, label="Lower LoA")
                ax.axhline(0, linestyle=":", linewidth=0.8, alpha=0.7)

                stats_text = (
                    f"n = {len(diff_vals)}\n"
                    f"Mean diff = {md:.3f}\n"
                    f"SD(diff) = {sd:.3f}\n"
                    f"LoA = ±{loa:.3f}\n"
                    f"Upper = {upper:.3f}\n"
                    f"Lower = {lower:.3f}"
                )
            else:
                stats_text = f"n = {len(diff_vals)}\nMean diff = n/a"

            ax.text(
                0.98, 0.02, stats_text,
                transform=ax.transAxes,
                horizontalalignment="right",
                verticalalignment="bottom",
                fontsize=10,
                color="orange",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="orange")
            )

            # enforce global limits for consistency across the grid
            ax.set_xlim(global_xmin, global_xmax)
            ax.set_ylim(global_ymin, global_ymax)

            ax.set_aspect("auto")

        plt.tight_layout()
        plt.savefig(self.output_path + save_name.replace('hka','hka_BlandA_'), dpi=300)
        return df

    def get_best_test_time_aug(self, data, meta_data):
        '''loads data, gets N number of augs, predicts all and takes the best model which is the one with lowest ere'''
        best_eres = 10000
        for i in range(self.cfg.TEST.TEST_TIME_AUG_NUM+1):
            if i == 10: 
                ##do not augment
                aug_data=data
            else:
                aug_seq = Augmentation(self.cfg)
                
                img_np = data.detach().cpu().numpy().squeeze(0)           # C,H,W
                img_hwc = np.transpose(img_np, (1,2,0))      
                pred_agg = aug_seq.tta_predict_with_config(self.net, img_hwc, meta_data, device=torch.device("cuda"))


                # aug_image = aug_seq(images=aug_data.detach().cpu().numpy())
                # #flip_back = det_aug.inverse(images=aug_image)
                # tta_seq = Augmentation(self.cfg).augmentation_fn_testtime()

                # # # Suppose aug_data is torch tensor (1, C, H, W), float in [0,1]
                # img_np = data.detach().cpu().numpy().squeeze(0)           # C,H,W
                # img_hwc = np.transpose(img_np, (1,2,0))                      # H,W,C
                # if img_hwc.dtype in [np.float32, np.float64] and img_hwc.max() <= 1.0:
                #     img_hwc = (img_hwc * 255).astype(np.uint8)

                # det = tta_seq.to_deterministic()
                # aug_img = det.augment_image(img_hwc) # H,W,C (uint8)
                # ##check
                # plt.imshow(img_np[0,:,:])
                # plt.imshow(aug_img[:,:,0])
        

            # aug_image_t = torch.from_numpy(aug_img).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            # pred = self.net(aug_image_t, meta_data)

            pred_back = 'help'
            _eres = 1
            if _eres < best_eres:

                best_pred = pred
                ###reverse back to original image

        return best_pred
    def run(self, dataloader):
        comparison_df = pd.DataFrame([])
        true_hkas = []

        for batch_idx, (data, target, landmarks, meta, id, orig_size, orig_img) in enumerate(dataloader):

            data = Variable(data).to(self.device)
            try:
                target = Variable(target).to(self.device)
            except:
                target = target
                print(target)
            meta_data = Variable(meta).to(self.device)
            orig_size = Variable(orig_size).to(self.device)

            ###if test time aug, run the model 5 times and select the one with the lowest eres.
            if self.cfg.TEST.TEST_TIME_AUG == True:
                pred = self.get_best_test_time_aug(data, meta_data)
            else:
                pred = self.net(data, meta_data)
            #predictions are heatmaps, points are taken as hottest point in each channel, which is scaled (multiplied by pixel size). 
            EH=evaluation_helper.evaluation_helper()

            #plot and caluclate values for each subject in the batch
            for i in range(self.bs):
                ### resize back to original for
                _data = orig_img[i].numpy()

                if self.label_dir == '':
                    _pred, _target = self.resize_backto_original(pred[i], target[i], orig_size[i])
                    _predicted_points=EH.get_landmarks_predonly(_pred, pixels_sizes=self.pixel_size.to('cpu'))
                    _predicted_points=_predicted_points.squeeze(0)
                    _target_points = target
                else:
                    _pred, _target = self.resize_backto_original(pred[i], target[i], orig_size[i])
                    _target_points,_predicted_points=EH.get_landmarks(_pred, _target, pixels_sizes=self.pixel_size.to('cpu'))
                    _target_points,_predicted_points= _target_points.squeeze(0),_predicted_points.squeeze(0)

                # if self.label_dir == '':
                #     #class cols is order of columns
                #     target_alpha = target[self.class_cols.index('Type')][0]
                #     target_class = target[self.class_cols.index('alpha (degrees)')+1][0]
                #     target_fhc = target[self.class_cols.index('FHC (%)')+1][0]

                
                if self.label_dir == '':
                    if self.save_img_landmarks_predandtrue == True:
                        ##
                        visuals(self.output_path+'/test/'+id[i], self.pixel_size[0], self.cfg,orig_size[i]).heatmaps(_data ,_pred,_target_points[0], _predicted_points, all_landmarks=self.save_all_landmarks, with_img = True)
                        if self.dataset_name == 'oai_nolandmarks':
                            true_hkas.append(_target_points[0].detach().cpu().numpy())
                    pass
                else:    
                    #### if the prediction does not have ground truth landmarks
                    if self.save_img_landmarks_predandtrue == True:
                        visuals(self.save_img_path+'/'+id[i], self.pixel_size[0], self.cfg, orig_size[i]).heatmaps(_data, _pred, _target_points, _predicted_points)

                    if self.save_asdcms == True:
                        out_dcm_dir = self.save_img_path+'/as_dcms' 
                        if os.path.exists(out_dcm_dir)==False:
                            os.mkdir(out_dcm_dir)

                        dcm_loc = self.dcm_dir +'/'+ id[i][:-1]+'_'+id[i][-1]+'.dcm'

                        if self.save_img_landmarks_predandtrue == True:
                            visuals(out_dcm_dir+'/'+id[i], self.pixel_size[0], self.cfg,orig_size[i]).heatmaps(_data ,_pred,_target_points, _predicted_points, all_landmarks=self.save_all_landmarks, with_img = True, as_dcm=True, dcm_loc=dcm_loc)

                        if self.save_heatmap_land_img == True:
                            visuals(out_dcm_dir+'/heatmap_'+id[i], self.pixel_size[0], self.cfg, orig_size[i]).heatmaps(_data ,_pred,_target_points, _predicted_points,w_landmarks=False,all_landmarks=self.save_all_landmarks, with_img = True, as_dcm=True, dcm_loc=dcm_loc)
                
                if self.save_heatmap_land_img == True:
                    visuals(self.save_img_path+'/heatmap_'+id[i], self.pixel_size[0], self.cfg, orig_size[i]).heatmaps(_data, _pred, _target_points, _predicted_points, w_landmarks=False, all_landmarks=self.save_all_landmarks)

                if self.save_txt == True:
                    visuals(self.output_path+'/txt/'+id[i],self.pixel_size, self.cfg, orig_size[i]).save_astxt(_data,_predicted_points,self.img_size,orig_size[i])
                
                if self.save_heatmap == True:
                    visuals(self.save_img_path+'/heatmap_only_'+id[i], self.pixel_size[0], self.cfg, orig_size[i]).heatmaps(_data ,_pred,_target_points, _predicted_points, w_landmarks=False, all_landmarks=self.save_all_landmarks, with_img = False)

                if self.save_heatmap_as_np == True:
                    visuals(self.output_path+'/np/numpy_heatmaps_'+id[i],self.pixel_size, self.cfg, orig_size[i]).save_np(_pred.squeeze(0))
                    # image also resizes here check**
                    visuals(self.output_path+'/np/numpy_heatmaps_uniformSize_'+id[i],self.pixel_size, self.cfg, orig_size[i]).save_np(pred[i])

                save_img=False
                if save_img ==True:
                    plt.imshow(np.squeeze(_data), cmap='Greys_r')
                    plt.axis('off')
                    plt.savefig(self.save_img_path+'/'+id[i]+'.png',dpi=1200, bbox_inches='tight', pad_inches = 0)       

                
                id_metric_df = self.compare_metrics(id[i], _predicted_points, _pred, _target_points, _target,self.pixel_size, orig_size[i])

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

        #
        #
        #
        #
        #

        self.logger.info("---------TEST RESULTS--------")

        ### if oai
        if self.dataset_name == 'oai_nolandmarks':
            ### plot a graph comparing the angles to each reviewer and the model
            # (when there is nan drop the value)
            pred_l, pred_r = comparison_df['L HKA'].to_numpy(), comparison_df['R HKA'].to_numpy()
            true_hkas = np.stack(true_hkas, axis=0)
            df = self.plot_hka_comparisons(pred_l,pred_r,true_hkas)
                         
        if self.label_dir == '':
            MRE = [None, None, None,None]
            comparsion_summary_ls = None
        else:
            #Get mean values from comparison summary ls, landmark metrics
            comparsion_summary_ls, MRE = self.comparison_summary(comparison_df)

        self.logger.info("MEAN VALUES (pix): {}".format(comparsion_summary_ls))
        self.logger.info("MRE: {} +/- {} pix".format(MRE[0], MRE[1]))
        self.logger.info("MRE no labrum: {} +/- {} pix".format(MRE[2], MRE[3]))


        #plot angles pred vs angles 
        if 'graf_angle_calc().graf_class_comparison' in self.cfg.TEST.COMPARISON_METRICS:
            if self.label_dir == '':   
                alpha_thresh_percentages,alpha_thresh_percentages_normalized= None, None
            else:
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
            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'graf&fhc pred i_ii&iii&iv', 'graf&fhc true i_ii&iii&iv',self.output_path, loc='test')._get_metrics(group=True,groups=[('n'),('a')])
            self.logger.info("Class Agreement i vs ii/iii/iv GRAF&FHC: {}".format(class_agreement))
            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'graf&fhc pred i&ii_iii&iv', 'graf&fhc true i&ii_iii&iv',self.output_path, loc='test')._get_metrics(group=True,groups=[('n'),('a')])
            self.logger.info("Class Agreement i/ii vs iii/iv GRAF&FHC: {}".format(class_agreement))
            
            comparison_df.to_csv(self.output_path+'/test/comparison_metrics.csv')
            print('Saving Results to comparison_metrics.csv')
        
        
        sdr_summary = ""

        if self.label_dir == '':
            pass
        else:
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
                   
                   
                    #plot angles pred vs angles 
                    if 'graf_angle_calc().graf_class_comparison' in self.cfg.TEST.COMPARISON_METRICS:
                                    #get mean alpha difference
                        self.logger.info('ALPHA MEAN DIFF:{}'.format(round(comparison_df['difference alpha'].mean(),3)))
                        self.logger.info('ALPHA ABSOLUTE MEAN DIFF:{}'.format(round(comparison_df['difference alpha'].apply(abs).mean(),3)))
                        #plot angles pred vs angles 
                        if 'graf_angle_calc().graf_class_comparison' in self.cfg.TEST.COMPARISON_METRICS:
                            visualisations.comparison(self.dataset_name,self.output_path,'graf').true_vs_pred_scatter(comparison_df['alpha pred'].to_numpy(),comparison_df['alpha true'].to_numpy(),loc='test')
                            visualisations.comparison(self.dataset_name,self.output_path,'fhc').true_vs_pred_scatter(comparison_df['fhc pred'].to_numpy(),comparison_df['fhc true'].to_numpy(),loc='test')
                        else:
                            visualisations.comparison(self.dataset_name, self.output_path,'graf').true_vs_pred_scatter(comparison_df['alpha pred'].to_numpy(),comparison_df['alpha true'].to_numpy(),loc='test')


                except:
                    raise ValueError('Check Landmark radial errors are calcuated')
                    

        return 
