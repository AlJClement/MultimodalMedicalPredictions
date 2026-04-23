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
from .comparison_metrics.landmark_metrics import landmark_metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm


class test():
    def __init__(self, cfg, logger):
        self.dataset_name = cfg.INPUT_PATHS.DATASET_NAME
        self.combine_graf_fhc=cfg.TRAIN.COMBINE_GRAF_FHC
        self.dcm_dir = cfg.INPUT_PATHS.DCMS
        self.validation = validation(cfg,logger,net=None)
        self.cfg=cfg
        self.img_size = cfg.DATASET.CACHED_IMAGE_SIZE
        self.logger = logger
        self.debug_testing = bool(getattr(cfg.TEST, "DEBUG_LOGGING", False))
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
        self.needs_orig_img = any([
            self.save_asdcms,
            self.save_txt,
            self.save_heatmap_land_img,
            self.save_img_landmarks_predandtrue,
            self.save_heatmap,
        ])

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
        self.base_test_output_dir_name = self._resolve_test_output_dir_name(cfg)
        self.test_output_dir_name = self.base_test_output_dir_name
        self.test_output_path = os.path.join(cfg.OUTPUT_PATH, self.test_output_dir_name)
        self.save_img_path = self.test_output_path
        if self.save_txt == True:
            if not os.path.isdir(self.output_path+'/'+'txt/'):
                os.makedirs(self.output_path+'/'+'txt/')
    
        if self.save_heatmap_as_np == True:
            if not os.path.isdir(self.output_path+'/np/'):
                os.makedirs(self.output_path+'/np/')
            if not os.path.isdir(self.output_path+'/np_test-time-aug/'):
                os.makedirs(self.output_path+'/np_test-time-aug/')
    
        if not os.path.exists(self.save_img_path):
            os.mkdir(self.save_img_path)

        if not os.path.exists(self.save_img_path+'-time-aug/'):
            os.mkdir(self.save_img_path+'-time-aug/')

        self.save_testtimeaug=self.save_img_path+'-time-aug/test-time-aug'
        if not os.path.exists(self.save_testtimeaug):
            os.mkdir(self.save_testtimeaug)
        
        self.pixel_size = torch.tensor(cfg.DATASET.PIXEL_SIZE).to(cfg.MODEL.DEVICE)
        self.pixel_size_cpu = self.pixel_size.detach().cpu()

        self.comparison_metrics=cfg.TEST.COMPARISON_METRICS
        self.dataset_name = cfg.INPUT_PATHS.DATASET_NAME
        self.sdr_thresholds = cfg.TEST.SDR_THRESHOLD
        self.sdr_units = cfg.TEST.SDR_UNITS
        self.wandb_enabled = wandb.run is not None
        self.wandb_failure_limit = getattr(cfg.TEST, 'WANDB_NUM_FAILURE_EXAMPLES', None)
        self.augmenter = Augmentation(self.cfg)
        self.eval_helper = evaluation_helper.evaluation_helper()
        self.landmark_metric_helper = landmark_metrics()
        self._prepare_run_paths(self.base_test_output_dir_name)

    def _resolve_test_output_dir_name(self, cfg):
        candidate_text = " ".join([
            str(getattr(cfg.INPUT_PATHS, "PARTITION", "")),
            str(getattr(cfg.INPUT_PATHS, "IMAGES", "")),
            str(getattr(cfg.INPUT_PATHS, "LABELS", "")),
            str(getattr(cfg.INPUT_PATHS, "META_PATH", "")),
            str(getattr(cfg.INPUT_PATHS, "DCMS", "")),
        ]).lower()

        if "mkuh" in candidate_text:
            return "test_mkuh"
        if "rnoh" in candidate_text:
            return "test_rnoh"
        if "retuve" in candidate_text:
            return "test_retuve"
        return "test"

    def _debug_print(self, *args, **kwargs):
        if self.debug_testing:
            print(*args, **kwargs)

    def _prepare_tta_preview_image(self, image):
        image = np.asarray(image)
        image = np.squeeze(image)

        if image.ndim == 3:
            if image.shape[-1] == 1:
                image = image[:, :, 0]
            else:
                image = image[:, :, 0]

        image = image.astype(np.float32)
        finite_mask = np.isfinite(image)
        if not np.any(finite_mask):
            return np.zeros_like(image, dtype=np.float32)

        min_val = np.min(image[finite_mask])
        max_val = np.max(image[finite_mask])
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image, dtype=np.float32)

        return image

    def _prepare_run_paths(self, output_dir_name):
        self.test_output_dir_name = output_dir_name
        self.test_output_path = os.path.join(self.output_path, self.test_output_dir_name)
        self.save_img_path = self.test_output_path

        os.makedirs(self.save_img_path, exist_ok=True)
        os.makedirs(self.save_img_path + '-time-aug/', exist_ok=True)

        self.save_testtimeaug = self.save_img_path + '-time-aug/test-time-aug'
        os.makedirs(self.save_testtimeaug, exist_ok=True)

    def load_network(self, model_path):
        model = model_init(self.cfg).get_net_from_conf(get_net_info=False)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
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

    def alpha_thresholds_structured(self, df, thresholds=[1,2,5,10]):
        df_alpha_diff = df['difference alpha']
        df_pred_class = df['class pred']
        df_true_class = df['class true']

        if isinstance(df_alpha_diff, pd.Series):
            alpha_diff = df_alpha_diff.to_numpy(dtype=float)
            pred_class = df_pred_class.to_numpy()
            true_class = df_true_class.to_numpy()
        else:
            alpha_diff = np.asarray(df_alpha_diff, dtype=float)
            pred_class = np.asarray(df_pred_class)
            true_class = np.asarray(df_true_class)

        np_agree = np.where(true_class == pred_class, 1.0, 0.0)
        np_disagree = np.where(true_class != pred_class, 1.0, 0.0)

        rows = []
        total = float(np.size(alpha_diff)) if np.size(alpha_diff) > 0 else 0.0

        for threshold in thresholds:
            mask = np.where(alpha_diff < threshold, 1.0, 0.0)
            percentage = 100.0 * np.sum(mask) / total if total else 0.0
            agree = 100.0 * np.sum(np_agree * mask) / total if total else 0.0
            disagree = 100.0 * np.sum(np_disagree * mask) / total if total else 0.0
            norm_agree = 100.0 * agree / percentage if percentage else 0.0
            norm_disagree = 100.0 * disagree / percentage if percentage else 0.0
            rows.append({
                "threshold_deg": threshold,
                "percentage": round(float(percentage), 2),
                "agree": round(float(agree), 2),
                "disagree": round(float(disagree), 2),
                "norm_agree": round(float(norm_agree), 2),
                "norm_disagree": round(float(norm_disagree), 2),
            })

        return rows
    
    def resize_backto_original(self, pred_map, target_map, orig_size):
        #resize images
        orig_size = orig_size.to('cpu').numpy()
        pred = pred_map.detach().cpu().numpy()

        pred = self.augmenter.reverse_downsample_and_pad(orig_size, pred)

        if isinstance(target_map[0], str):
            target = target_map[0]
            pass
        else:
            target = target_map.detach().cpu().numpy()
            target = self.augmenter.reverse_downsample_and_pad(orig_size, target)

        return pred, target

    def _sanitize_wandb_key(self, key):
        return str(key).strip().lower().replace(' ', '_').replace('/', '_')

    def _to_wandb_scalar(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.size == 1:
                value = value.item()
            else:
                return None
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, (float, int, str, bool)) or value is None:
            return value
        return str(value)

    def _prepare_display_image(self, image):
        image = np.asarray(image)
        image = np.squeeze(image)

        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.moveaxis(image, 0, -1)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]

        return image

    def _prepare_heatmap(self, pred_map):
        heatmap = np.asarray(pred_map)
        heatmap = np.squeeze(heatmap)

        if heatmap.ndim == 3:
            heatmap = np.sum(heatmap, axis=0)

        return heatmap

    def _format_example_caption(self, sample_id, metric_row):
        priority_keys = [
            'difference alpha',
            'alpha pred',
            'alpha true',
            'fhc pred',
            'fhc true',
            'L HKA',
            'R HKA',
        ]

        metric_parts = []
        for key in priority_keys:
            if key not in metric_row:
                continue
            value = self._to_wandb_scalar(metric_row[key])
            if isinstance(value, float):
                metric_parts.append(f"{key}: {value:.3f}")
            elif value is not None:
                metric_parts.append(f"{key}: {value}")

        if not metric_parts:
            for key, value in metric_row.items():
                if key == 'ID':
                    continue
                value = self._to_wandb_scalar(value)
                if isinstance(value, float):
                    metric_parts.append(f"{key}: {value:.3f}")
                elif isinstance(value, int):
                    metric_parts.append(f"{key}: {value}")
                if len(metric_parts) >= 4:
                    break

        if metric_parts:
            return f"{sample_id} | " + ", ".join(metric_parts)
        return str(sample_id)

    def _build_wandb_example(self, sample_id, image, pred_map, predicted_points, metric_row, target_points=None):
        display_image = self._prepare_display_image(image)
        display_heatmap = self._prepare_heatmap(pred_map)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(display_image, cmap='gray')
        ax.imshow(display_heatmap, cmap='inferno', alpha=0.35)

        predicted_points = np.asarray(predicted_points)
        if predicted_points.ndim == 2 and predicted_points.shape[1] >= 2:
            ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=10, label='pred')

        if target_points is not None and self.label_dir != '':
            target_points = np.asarray(target_points)
            if target_points.ndim == 2 and target_points.shape[1] >= 2:
                ax.scatter(target_points[:, 0], target_points[:, 1], color='lime', s=10, label='true')

        ax.set_title(str(sample_id))
        ax.axis('off')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='lower right')

        caption = self._format_example_caption(sample_id, metric_row)
        wandb_image = wandb.Image(fig, caption=caption)
        plt.close(fig)
        return wandb_image

    def _build_wandb_image_from_path(self, image_path, sample_id, metric_row):
        caption = self._format_example_caption(sample_id, metric_row)
        return wandb.Image(image_path, caption=caption)

    def _get_failure_score(self, metric_row):
        def _float_or_none(key):
            value = metric_row.get(key)
            value = self._to_wandb_scalar(value)
            if isinstance(value, (int, float)):
                return float(value)
            return None

        class_pred = metric_row.get('class pred')
        class_true = metric_row.get('class true')
        if class_pred is not None and class_true is not None and class_pred != class_true:
            alpha_diff = abs(_float_or_none('difference alpha') or 0.0)
            return 1_000_000.0 + alpha_diff

        fhc_pred = metric_row.get('fhc class pred')
        fhc_true = metric_row.get('fhc class true')
        if fhc_pred is not None and fhc_true is not None and fhc_pred != fhc_true:
            fhc_diff = abs((_float_or_none('fhc pred') or 0.0) - (_float_or_none('fhc true') or 0.0))
            return 500_000.0 + fhc_diff

        alpha_diff = _float_or_none('difference alpha')
        if alpha_diff is not None:
            return abs(alpha_diff)

        landmark_errors = []
        for key, value in metric_row.items():
            if 'landmark radial error' not in key:
                continue
            value = self._to_wandb_scalar(value)
            if isinstance(value, (int, float)):
                landmark_errors.append(float(value))
        if landmark_errors:
            return max(landmark_errors)

        l_hka = _float_or_none('L HKA')
        r_hka = _float_or_none('R HKA')
        if l_hka is not None or r_hka is not None:
            return max(abs(l_hka or 0.0), abs(r_hka or 0.0))

        return None

    def _add_failure_candidate(self, failure_examples, candidate):
        score = candidate.get("score")
        if score is None:
            return

        failure_examples.append(candidate)
        if self.wandb_failure_limit is None:
            return

        failure_examples.sort(key=lambda item: item["score"], reverse=True)
        if len(failure_examples) > int(self.wandb_failure_limit):
            del failure_examples[self.wandb_failure_limit:]

    def _add_class_agreement_summary(self, prefix, agreement_metrics, summary_metrics):
        metric_name_map = {
            'percision: ': 'precision',
            'recall: ': 'recall',
            'accuracy': 'accuracy',
            'sensitivity': 'sensitivity',
            'specificity': 'specificity',
        }

        for raw_name, value in agreement_metrics:
            if raw_name in {'TN:', 'FP:', 'FN:', 'TP:'}:
                continue
            metric_key = metric_name_map.get(raw_name, self._sanitize_wandb_key(raw_name))
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    summary_metrics[f"{prefix}_{metric_key}"] = float(value.reshape(-1)[0])
                elif value.size > 1:
                    summary_metrics[f"{prefix}_{metric_key}_mean"] = float(np.mean(value))
                    for idx, item in enumerate(value.reshape(-1)):
                        summary_metrics[f"{prefix}_{metric_key}_{idx}"] = float(item)
            elif isinstance(value, (float, int, np.floating, np.integer)):
                summary_metrics[f"{prefix}_{metric_key}"] = float(value)
            else:
                summary_metrics[f"{prefix}_{metric_key}"] = str(value)

    def _save_class_agreement_bar_plot(self, prefix, agreement_metrics):
        metric_lookup = {
            "accuracy": None,
            "sensitivity": None,
            "specificity": None,
        }

        for raw_name, value in agreement_metrics:
            key = {
                "accuracy": "accuracy",
                "sensitivity": "sensitivity",
                "specificity": "specificity",
            }.get(raw_name)
            if key is None:
                continue

            if isinstance(value, np.ndarray):
                arr = value.reshape(-1)
                metric_lookup[key] = float(np.mean(arr)) if arr.size > 0 else None
            elif isinstance(value, (float, int, np.floating, np.integer)):
                metric_lookup[key] = float(value)

        labels = []
        values = []
        for key in ["accuracy", "sensitivity", "specificity"]:
            val = metric_lookup[key]
            if val is None:
                continue
            labels.append(key.capitalize())
            values.append(val)

        if not values:
            return None

        os.makedirs(self.test_output_path, exist_ok=True)
        plot_path = os.path.join(self.test_output_path, f"{prefix}_bar_metrics.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=["#2f6db3", "#2a9d8f", "#e9c46a"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percent")
        ax.set_title(prefix.replace("_", " ").title())
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=10)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    def _save_metric_bar_plot_by_point(self, comparison_df, column_prefix, metric_label, filename, color):
        labels = []
        values = []

        for i in range(self.num_landmarks):
            col = f'{column_prefix} p{i+1}'
            if col not in comparison_df.columns:
                continue
            series = pd.to_numeric(comparison_df[col], errors='coerce').dropna()
            if series.empty:
                continue
            labels.append(f"p{i+1}")
            values.append(float(series.mean()))

        if not values:
            return None, None

        os.makedirs(self.test_output_path, exist_ok=True)
        plot_path = os.path.join(self.test_output_path, filename)
        mean_value = float(np.mean(values))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(labels, values, color=color)
        ax.axhline(mean_value, color="#1d4ed8", linestyle="--", linewidth=1.5, label=f"Average = {mean_value:.2f}")
        ax.set_ylabel(f"Mean {metric_label}")
        ax.set_xlabel("Landmark Point")
        ax.set_title(f"Mean {metric_label} by Landmark Point")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend(loc="upper right")

        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return plot_path, mean_value

    def _save_ere_bar_plot(self, comparison_df):
        return self._save_metric_bar_plot_by_point(
            comparison_df=comparison_df,
            column_prefix='ere',
            metric_label='ERE',
            filename='ere_bar_by_point.png',
            color="#d97706",
        )

    def _save_mre_bar_plot(self, comparison_df):
        return self._save_metric_bar_plot_by_point(
            comparison_df=comparison_df,
            column_prefix='landmark radial error',
            metric_label='MRE',
            filename='mre_bar_by_point.png',
            color="#0f766e",
        )

    def _save_sdr_bar_plot(self, sdr_values):
        if sdr_values is None:
            return None, None

        try:
            values = [float(v) for v in sdr_values]
        except (TypeError, ValueError):
            return None, None

        if not values:
            return None, None

        labels = []
        for threshold in self.sdr_thresholds[:len(values)]:
            if self.sdr_units == 'mm':
                labels.append(f"{threshold:g} mm")
            else:
                labels.append(f"{threshold:g} px")

        os.makedirs(self.test_output_path, exist_ok=True)
        plot_path = os.path.join(self.test_output_path, "sdr_all_landmarks.png")
        mean_value = float(np.mean(values))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(labels, values, color="#2563eb")
        ax.axhline(mean_value, color="#1d4ed8", linestyle="--", linewidth=1.5, label=f"Average = {mean_value:.2f}")
        ax.set_ylim(0, 100)
        ax.set_ylabel("SDR (%)")
        ax.set_xlabel("Threshold")
        ax.set_title("SDR All Landmarks")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend(loc="upper right")

        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return plot_path, mean_value

    def _save_single_value_point_plot(self, value, ylabel, title, filename, color):
        if value is None:
            return None

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if np.isnan(numeric_value):
            return None

        os.makedirs(self.test_output_path, exist_ok=True)
        plot_path = os.path.join(self.test_output_path, filename)

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter([0], [numeric_value], color=color, s=140, zorder=3)
        ax.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([0])
        ax.set_xticklabels([title])

        offset = 0.02 * max(abs(numeric_value), 1.0)
        text_y = numeric_value + offset if numeric_value >= 0 else numeric_value - offset
        va = "bottom" if numeric_value >= 0 else "top"
        ax.text(
            0,
            text_y,
            f"{numeric_value:.3f}",
            ha="center",
            va=va,
            fontsize=10,
        )

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    def _get_wrong_case_plot_info(self, sample_id, metric_row):
        class_pred = metric_row.get('class pred')
        class_true = metric_row.get('class true')
        fhc_pred = metric_row.get('fhc pred')
        fhc_true = metric_row.get('fhc true')

        if class_pred is None or class_true is None or fhc_pred is None or fhc_true is None:
            return None

        class_wrong = class_pred != class_true
        fhc_wrong = None
        try:
            fhc_wrong = ('n' if float(fhc_pred) > 0.50 else 'a') != ('n' if float(fhc_true) > 0.50 else 'a')
        except Exception:
            try:
                fhc_wrong = ('n' if float(fhc_pred) > 50 else 'a') != ('n' if float(fhc_true) > 50 else 'a')
            except Exception:
                return None

        if not class_wrong and not fhc_wrong:
            return None

        if fhc_wrong and class_wrong:
            folder = "wrong_classes"
        elif fhc_wrong:
            folder = "wrong_class_fhc"
        else:
            folder = "wrong_class_graf"

        return {
            "category": folder,
            "path": os.path.join(self.save_img_path, folder, f"{sample_id}.png"),
        }

    def _save_wrong_case_plot(self, sample_id, image, pred_map, target_points, predicted_points, orig_size):
        if self.label_dir == '':
            return None

        base_path = os.path.join(self.save_img_path, sample_id)
        visuals(base_path, self.pixel_size[0], self.cfg, orig_size).heatmaps(
            image,
            pred_map,
            target_points,
            predicted_points,
            all_landmarks=self.save_all_landmarks,
            with_img=True,
        )
        return base_path + ".png"

    def _build_summary_tables(self, summary_metrics, plot_paths):
        summary_rows = []
        excluded_summary_keys = {"fhc pred", "fhc true", "alpha pred", "alpha true", "difference alpha"}
        for key in sorted(summary_metrics.keys()):
            if str(key).strip().lower() in excluded_summary_keys:
                continue
            value = self._to_wandb_scalar(summary_metrics[key])
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if value is None:
                value = ""
            elif isinstance(value, (list, tuple, dict)):
                value = str(value)
            else:
                value = str(value)
            summary_rows.append([str(key), value])

        summary_table = wandb.Table(columns=["metric", "value"], data=summary_rows)

        plot_table = wandb.Table(columns=["plot_name", "local_path", "exists"])
        for key in sorted(plot_paths.keys()):
            path = plot_paths[key]
            plot_table.add_data(key, path, os.path.exists(path))

        return summary_table, plot_table

    def _extract_classification_triplet(self, class_agreement):
        metrics = {"accuracy": None, "precision": None, "recall": None}
        name_map = {
            "accuracy": "accuracy",
            "percision:": "precision",
            "precision:": "precision",
            "recall:": "recall",
        }

        for item in class_agreement:
            if len(item) < 2:
                continue
            raw_name = str(item[0]).strip().lower()
            metric_name = name_map.get(raw_name)
            if metric_name is None:
                continue

            value = item[1]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    value = None
                elif value.size == 1:
                    value = value.item()
                else:
                    value = float(np.mean(value))
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
            if value is not None:
                metrics[metric_name] = round(float(value), 2)

        return metrics

    def _build_experiment_summary_row(
        self,
        comparison_df,
        mre_stats,
        sdr_mm,
        alpha_threshold_rows,
        graf_i_vs_234,
        graf_12_vs_34,
        fhc_abnormal_vs_normal,
        graf_fhc_i_vs_234,
        graf_fhc_12_vs_34,
    ):
        row = {
            "mre_mm": None,
            "std_mm": None,
            "mre_pix": None,
            "std_pix": None,
            "mre_no_labrum_pix": None,
            "std_no_labrum_pix": None,
            "sdr_0_5mm": None,
            "sdr_1mm": None,
            "sdr_2mm": None,
            "sdr_4mm": None,
            "mean_diff_alpha_deg": None,
            "absolute_diff_alpha_deg": None,
        }

        pixel_size_mm = float(self.pixel_size_cpu[0].item()) if self.pixel_size_cpu.numel() > 0 else None
        if mre_stats and mre_stats[0] is not None:
            row["mre_pix"] = round(float(mre_stats[0]), 2)
            row["std_pix"] = round(float(mre_stats[1]), 2)
            if pixel_size_mm is not None:
                row["mre_mm"] = round(float(mre_stats[0]) * pixel_size_mm, 2)
                row["std_mm"] = round(float(mre_stats[1]) * pixel_size_mm, 2)
        if len(mre_stats) > 2 and mre_stats[2] is not None:
            row["mre_no_labrum_pix"] = round(float(mre_stats[2]), 2)
            row["std_no_labrum_pix"] = round(float(mre_stats[3]), 2)

        if sdr_mm is not None:
            for threshold, value in zip([0.5, 1, 2, 4], sdr_mm):
                if threshold == 0.5:
                    key = "sdr_0_5mm"
                elif threshold == 1:
                    key = "sdr_1mm"
                elif threshold == 2:
                    key = "sdr_2mm"
                else:
                    key = "sdr_4mm"
                row[key] = round(float(value), 2)

        if 'difference alpha' in comparison_df.columns:
            alpha_diff = pd.to_numeric(comparison_df['difference alpha'], errors='coerce').dropna()
            if not alpha_diff.empty:
                row["mean_diff_alpha_deg"] = round(float(alpha_diff.mean()), 2)
                row["absolute_diff_alpha_deg"] = round(float(alpha_diff.abs().mean()), 2)

        threshold_prefix_map = {
            1: "deg_1",
            2: "deg_2",
            5: "deg_5",
            10: "deg_10",
        }
        for threshold_row in alpha_threshold_rows or []:
            prefix = threshold_prefix_map.get(threshold_row["threshold_deg"])
            if prefix is None:
                continue
            row[f"{prefix}_percentage"] = threshold_row["percentage"]
            row[f"{prefix}_agree"] = threshold_row["agree"]
            row[f"{prefix}_disagree"] = threshold_row["disagree"]
            row[f"{prefix}_norm_agree"] = threshold_row["norm_agree"]
            row[f"{prefix}_norm_disagree"] = threshold_row["norm_disagree"]

        for prefix, metrics in [
            ("graf_1_vs_234", graf_i_vs_234),
            ("graf_12_vs_34", graf_12_vs_34),
            ("fhc_abnormal_vs_normal", fhc_abnormal_vs_normal),
            ("graf_fhc_1_vs_234", graf_fhc_i_vs_234),
            ("graf_fhc_12_vs_34", graf_fhc_12_vs_34),
        ]:
            metric_values = metrics or {}
            row[f"{prefix}_accuracy"] = metric_values.get("accuracy")
            row[f"{prefix}_precision"] = metric_values.get("precision")
            row[f"{prefix}_recall"] = metric_values.get("recall")

        return row

    def _build_experiment_summary_heatmap(self, summary_row):
        percentage_columns = [
            "sdr_0_5mm",
            "sdr_1mm",
            "sdr_2mm",
            "sdr_4mm",
            "deg_1_percentage",
            "deg_1_agree",
            "deg_1_disagree",
            "deg_1_norm_agree",
            "deg_1_norm_disagree",
            "deg_2_percentage",
            "deg_2_agree",
            "deg_2_disagree",
            "deg_2_norm_agree",
            "deg_2_norm_disagree",
            "deg_5_percentage",
            "deg_5_agree",
            "deg_5_disagree",
            "deg_5_norm_agree",
            "deg_5_norm_disagree",
            "deg_10_percentage",
            "deg_10_agree",
            "deg_10_disagree",
            "deg_10_norm_agree",
            "deg_10_norm_disagree",
            "graf_1_vs_234_accuracy",
            "graf_1_vs_234_precision",
            "graf_1_vs_234_recall",
            "graf_12_vs_34_accuracy",
            "graf_12_vs_34_precision",
            "graf_12_vs_34_recall",
            "fhc_abnormal_vs_normal_accuracy",
            "fhc_abnormal_vs_normal_precision",
            "fhc_abnormal_vs_normal_recall",
            "graf_fhc_1_vs_234_accuracy",
            "graf_fhc_1_vs_234_precision",
            "graf_fhc_1_vs_234_recall",
            "graf_fhc_12_vs_34_accuracy",
            "graf_fhc_12_vs_34_precision",
            "graf_fhc_12_vs_34_recall",
        ]

        labels = []
        values = []
        for column in percentage_columns:
            value = summary_row.get(column)
            if value is None:
                continue
            labels.append(column)
            values.append(float(value))

        if not values:
            return None

        heatmap_values = np.array([values], dtype=float)
        fig_width = max(12, len(values) * 0.45)
        fig, ax = plt.subplots(figsize=(fig_width, 2.8))
        im = ax.imshow(heatmap_values, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
        ax.set_yticks([0])
        ax.set_yticklabels(["summary"])
        ax.set_title("Evaluation Percentage Heatmap (50-100)")

        for idx, value in enumerate(values):
            ax.text(idx, 0, f"{value:.1f}", ha="center", va="center", fontsize=7, color="black")

        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        return wandb.Image(fig, caption="CSV-style evaluation summary heatmap")

    def _log_wandb_results(self, summary_metrics, failure_images, categorized_wrong_case_paths, plot_paths, comparison_csv_path, comparison_df, evaluation_summary_table=None, evaluation_summary_heatmap=None, wandb_namespace="test"):
        if not self.wandb_enabled:
            return

        scalar_summary = {}
        text_summary = {}
        excluded_summary_keys = {"fhc pred", "fhc true", "alpha pred", "alpha true", "difference alpha"}

        for key, value in summary_metrics.items():
            if str(key).strip().lower() in excluded_summary_keys:
                continue
            clean_key = self._sanitize_wandb_key(key)
            value = self._to_wandb_scalar(value)
            if isinstance(value, (float, int, bool)) or value is None:
                scalar_summary[f"{wandb_namespace}/{clean_key}"] = value
            elif value is not None:
                text_summary[f"{wandb_namespace}/{clean_key}"] = str(value)

        if scalar_summary:
            wandb.log(scalar_summary)

        for key, value in text_summary.items():
            wandb.run.summary[key] = value

        wandb.run.summary[f"{wandb_namespace}/comparison_rows"] = len(comparison_df)

        if failure_images:
            wandb.log({f"{wandb_namespace}/outputs": failure_images})

        if categorized_wrong_case_paths:
            wrong_case_payload = {}
            for category, paths in categorized_wrong_case_paths.items():
                existing_paths = [path for path in paths if os.path.exists(path)]
                if existing_paths:
                    wrong_case_payload[f"{wandb_namespace}/{category}"] = [wandb.Image(path) for path in existing_paths]
            if wrong_case_payload:
                wandb.log(wrong_case_payload)

        if plot_paths:
            plot_payload = {}
            for key, path in plot_paths.items():
                if os.path.exists(path):
                    plot_payload[f"{wandb_namespace}/plots/{key}"] = wandb.Image(path)
            if plot_payload:
                wandb.log(plot_payload)

        if evaluation_summary_table is not None:
            wandb.log({f"{wandb_namespace}/evaluation_summary_table": evaluation_summary_table})

        if evaluation_summary_heatmap is not None:
            wandb.log({f"{wandb_namespace}/evaluation_summary_heatmap": evaluation_summary_heatmap})

        if os.path.exists(comparison_csv_path):
            artifact = wandb.Artifact(f"{wandb_namespace}-comparison-metrics", type="evaluation")
            artifact.add_file(comparison_csv_path)
            wandb.log_artifact(artifact)
    
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
            self._debug_print(f"{title}: {len(x_filt)} points after filtering")

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
        plot_path = self.output_path + save_name
        plt.savefig(plot_path, dpi=300)

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
            self._debug_print(f"{title}: {len(x_filt)} points after filtering")

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
        bland_altman_path = self.output_path + save_name.replace('hka', 'hka_BlandA_', 1)
        plt.savefig(bland_altman_path, dpi=300)
        return df

    def get_best_test_time_aug(self, data, meta_data, id):
        '''loads data, gets N number of augs, predicts all and takes the best model which is the one with lowest ere'''
        metric = str(getattr(self.cfg.TEST, "TTA_SELECTION_METRIC", "ERE")).strip() or "ERE"
        EH = self.eval_helper
        pixels_sizes = self.pixel_size_cpu
        aug_seq = self.augmenter
        img_np = data.detach().cpu().numpy().squeeze(0) # C, H, W
        img_hwc = np.transpose(img_np, (1, 2, 0))
        best_pred = None
        best_aug_img = None
        best_affine_params = None
        if metric == 'ERE':
            best_metric = 10000
        elif metric == 'angle':
            best_metric = 10000
        elif metric == 'angle_and_ere':
            best_metric = 10000

        self._debug_print('ID:', id[0])
        for i in range(self.cfg.TEST.TEST_TIME_AUG_NUM+1):
            self._debug_print('AUG:', i)
            if i == 0: 
                ##do not augment
                pred = self.net(data, meta_data)                
            else:
                ### get augment and prediction with augment     
                pred, aug_img, affine_params = aug_seq.tta_predict_with_config(self.net, img_hwc, meta_data, device=torch.device("cuda"))

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
        
            ###Check eres
            ## if metric = 
            if metric == 'ERE':
                _pred = pred.cpu()
                pred_landmarks=EH.get_landmarks_predonly(_pred, pixels_sizes)
                ere_ls = self.landmark_metric_helper.get_eres(pred_landmarks, _pred, pixelsize=pixels_sizes)
                total_metric = sum(value for _, value in ere_ls)

            elif metric == 'angle':
                _pred = pred.cpu()
                pred_landmarks=EH.get_landmarks_predonly(_pred, pixels_sizes)
                hka = protractor_hka().hka_angles(pred_landmarks.squeeze(0), pred.squeeze(0).cpu(), pred_landmarks.squeeze(0), pred.squeeze(0).cpu(),pixelsize=pixels_sizes)
                hka_l, hka_r = hka[0][1], hka[3][1]
                hka_l_fl, hka_r_fl = hka[1][1], hka[4][1]
                hka_l_tib, hka_r_tib = hka[2][1], hka[5][1]

                ## see if  femur length is reasonable
                if abs(hka_r_fl/hka_l_fl) > 0.9 and abs(hka_r_fl /hka_l_fl) < 1.1:
                    ## see if tibia length is reasonale
                    if abs(hka_r_tib /hka_l_tib) > 0.9 and abs(hka_r_tib /hka_l_tib) < 1.1:
                        if abs(hka_l) < 18 and abs(hka_r) < 18:
                            total_metric = abs(hka_l) + abs(hka_r)
                            self._debug_print(total_metric)
                        else:
                            total_metric = 10000
                    else:
                        total_metric = 10000
                else:
                    total_metric = 10000
            elif metric == 'angle_and_ere':
                _pred = pred.cpu()
                pred_landmarks=EH.get_landmarks_predonly(_pred, pixels_sizes)
                hka = protractor_hka().hka_angles(pred_landmarks.squeeze(0), pred.squeeze(0).cpu(), pred_landmarks.squeeze(0), pred.squeeze(0).cpu(),pixelsize=pixels_sizes)
                hka_l, hka_r = hka[0][1], hka[3][1]
                hka_l_fl, hka_r_fl = hka[1][1], hka[4][1]
                hka_l_tib, hka_r_tib = hka[2][1], hka[5][1]

                ## see if  femur length is reasonable
                if abs(hka_r_fl/hka_l_fl) > 0.9 and abs(hka_r_fl /hka_l_fl) < 1.1:
                    ## see if tibia length is reasonale
                    if abs(hka_r_tib /hka_l_tib) > 0.9 and abs(hka_r_tib /hka_l_tib) < 1.1:
                        if abs(hka_l) < 18 and abs(hka_r) < 18:
                            ere_ls = self.landmark_metric_helper.get_eres(pred_landmarks, _pred, pixelsize=pixels_sizes)
                            total_metric = sum(value for _, value in ere_ls)
                            self._debug_print(total_metric)

                        else:
                            total_metric = 10000
                    else:
                        total_metric = 10000
                else:
                    total_metric = 10000

            
            else:
                raise ValueError('define metric for test time AU')

            if total_metric < best_metric:
                ##update best ere to beat
                best_metric = total_metric 

                ##update best prediction
                best_pred = pred
                if i > 0:
                    best_aug_img = aug_img
                    best_affine_params = affine_params
                    print(f"TTA selected augmented candidate for {id[0]} at aug {i} with {metric}={float(total_metric):.4f}", flush=True)
                else:
                    best_aug_img = None
                    best_affine_params = None
                    print(f"TTA selected original image for {id[0]} with {metric}={float(total_metric):.4f}", flush=True)

    
        ##plot best heatmap over augmented or normal image
        plt.ioff()
        plt.imshow(img_hwc[:, :, 0], cmap="gray")
        agg_np = best_pred.cpu().numpy()
        combined = np.sum(agg_np, axis=1)
        plt.imshow(combined[0])
        plt.savefig(self.save_testtimeaug+'/'+id[0]+'.jpg')
        plt.close()

        ##plot best heatmap over augmented or normal image
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        ## try
        try:
            if best_aug_img is None:
                raise ValueError("best prediction used the original image")
            if self.net.__class__.__name__ == 'hrnet':
                best_aug_img = best_aug_img.squeeze(0)
            aug_preview = self._prepare_tta_preview_image(best_aug_img)
            orig_preview = self._prepare_tta_preview_image(img_hwc)
            axes[0].imshow(aug_preview, cmap="gray", vmin=0.0, vmax=1.0)
            axes[0].set_title("Augmented")
            axes[0].axis("off")

            axes[1].imshow(orig_preview, cmap="gray", vmin=0.0, vmax=1.0)
            axes[1].set_title("Original")
            axes[1].axis("off")

            fig.text(0.5, 0.05, str(best_affine_params), 
                    ha="center", fontsize=11)                            
            plt.tight_layout()
            plt.savefig(self.save_testtimeaug+'/'+id[0]+'_AUG.jpg')
            plt.close()
        except:
            if best_metric == 10000:
                self._debug_print('defaulting back to the original image as all augmentations had unreasonable angles pr femur/tibia lengths')
                pred = self.net(data, meta_data)                
                best_pred = pred
            else:
                self._debug_print('the best prediction was on the original image!')


        return best_pred

    def run(self, dataloader, use_tta=None, output_dir_name=None, wandb_namespace="test"):
        use_tta = self.cfg.TEST.TEST_TIME_AUG if use_tta is None else bool(use_tta)
        output_dir_name = output_dir_name or self.base_test_output_dir_name
        self._prepare_run_paths(output_dir_name)

        comparison_rows = []
        true_hkas = []
        failure_candidates = []
        wandb_summary = {}
        wandb_plot_paths = {}
        categorized_wrong_case_paths = {
            "wrong_class_fhc": [],
            "wrong_class_graf": [],
            "wrong_classes": [],
        }
        EH = self.eval_helper
        start_time = datetime.datetime.now()
        self.net.eval()
        progress = tqdm(dataloader, desc="Testing", leave=True, disable=self.debug_testing)

        with torch.inference_mode():
            for batch_idx, (data, target, landmarks, meta, id, orig_size, orig_img) in enumerate(progress):

                data = data.to(self.device, non_blocking=True)
                try:
                    target = target.to(self.device, non_blocking=True)
                except Exception:
                    self._debug_print(target)
                meta_data = meta.to(self.device, non_blocking=True)
                orig_size = orig_size.to(self.device, non_blocking=True)

                if use_tta:
                    ###if test time aug, run the model 5 times and select the one with the lowest eres.
                    pred = self.get_best_test_time_aug(data, meta_data, id)
                else:
                    pred = self.net(data, meta_data)

                batch_size = data.shape[0]

                #plot and caluclate values for each subject in the batch
                for i in range(batch_size):
                ### resize back to original for
                    _data = orig_img[i].numpy() if self.needs_orig_img else None

                    if self.label_dir == '':
                        _pred, _target = self.resize_backto_original(pred[i], target[i], orig_size[i])
                        _predicted_points=EH.get_landmarks_predonly(_pred, pixels_sizes=self.pixel_size_cpu)
                        _predicted_points=_predicted_points.squeeze(0)
                        _target_points = target
                    else:
                        _pred, _target = self.resize_backto_original(pred[i], target[i], orig_size[i])
                        _target_points,_predicted_points=EH.get_landmarks(_pred, _target, pixels_sizes=self.pixel_size_cpu)
                        _target_points,_predicted_points= _target_points.squeeze(0),_predicted_points.squeeze(0)

                # if self.label_dir == '':
                #     #class cols is order of columns
                #     target_alpha = target[self.class_cols.index('Type')][0]
                #     target_class = target[self.class_cols.index('alpha (degrees)')+1][0]
                #     target_fhc = target[self.class_cols.index('FHC (%)')+1][0]

                
                    if self.label_dir == '':
                        if self.save_img_landmarks_predandtrue == True:
                            visuals(os.path.join(self.test_output_path, id[i]), self.pixel_size[0], self.cfg,orig_size[i]).heatmaps(_data ,_pred,_target_points[0], _predicted_points, all_landmarks=self.save_all_landmarks, with_img = True)
                            if self.dataset_name == 'oai_nolandmarks':
                                true_hkas.append(_target_points[0].detach().cpu().numpy())
                    else:
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
                        np_output_dir = self.output_path+'/np_test-time-aug' if use_tta else self.output_path+'/np'
                        visuals(np_output_dir+'/numpy_heatmaps_'+id[i],self.pixel_size, self.cfg, orig_size[i]).save_np(_pred.squeeze(0))
                        visuals(np_output_dir+'/numpy_heatmaps_uniformSize_'+id[i],self.pixel_size, self.cfg, orig_size[i]).save_np(pred[i])

                    id_metric_df = self.compare_metrics(id[i], _predicted_points, _pred, _target_points, _target,self.pixel_size, orig_size[i])
                    metric_row = id_metric_df.iloc[0].to_dict()
                    comparison_rows.append(metric_row)

                    wrong_case_info = self._get_wrong_case_plot_info(id[i], metric_row)
                    if wrong_case_info is not None:
                        if not os.path.exists(wrong_case_info["path"]):
                            self._save_wrong_case_plot(
                                sample_id=id[i],
                                image=_data,
                                pred_map=_pred,
                                target_points=_target_points,
                                predicted_points=_predicted_points,
                                orig_size=orig_size[i],
                            )
                        categorized_wrong_case_paths[wrong_case_info["category"]].append(wrong_case_info["path"])

                    if self.wandb_enabled and (self.wandb_failure_limit is None or int(self.wandb_failure_limit) > 0):
                        target_points_for_log = _target_points if self.label_dir != '' else None
                        wrong_case_image_path = wrong_case_info["path"] if wrong_case_info is not None else None
                        self._add_failure_candidate(
                            failure_candidates,
                            {
                                "score": self._get_failure_score(metric_row),
                                "sample_id": id[i],
                                "saved_wrong_case_path": wrong_case_image_path,
                                "image": np.array(_data, copy=True),
                                "pred_map": _pred.detach().cpu().numpy() if isinstance(_pred, torch.Tensor) else np.array(_pred, copy=True),
                                "predicted_points": _predicted_points.detach().cpu().numpy() if isinstance(_predicted_points, torch.Tensor) else np.array(_predicted_points, copy=True),
                                "metric_row": dict(metric_row),
                                "target_points": None if target_points_for_log is None else (
                                    target_points_for_log.detach().cpu().numpy()
                                    if isinstance(target_points_for_log, torch.Tensor)
                                    else np.array(target_points_for_log, copy=True)
                                ),
                            },
                        )

                progress.set_postfix(samples=len(comparison_rows), refresh=False)

        comparison_df = pd.DataFrame(comparison_rows)
        failure_images = []
        if self.wandb_enabled and failure_candidates:
            for candidate in failure_candidates:
                if candidate.get("saved_wrong_case_path") and os.path.exists(candidate["saved_wrong_case_path"]):
                    failure_images.append(
                        self._build_wandb_image_from_path(
                            image_path=candidate["saved_wrong_case_path"],
                            sample_id=candidate["sample_id"],
                            metric_row=candidate["metric_row"],
                        )
                    )
                else:
                    failure_images.append(
                        self._build_wandb_example(
                            sample_id=candidate["sample_id"],
                            image=candidate["image"],
                            pred_map=candidate["pred_map"],
                            predicted_points=candidate["predicted_points"],
                            metric_row=candidate["metric_row"],
                            target_points=candidate["target_points"],
                        )
                    )
        total_time = datetime.datetime.now() - start_time
        self._debug_print('Time taken for epoch = ', total_time)
        file_suffix = '_test-time-aug' if use_tta else ''
        comparison_csv_path = os.path.join(self.test_output_path, f'comparison_metrics{file_suffix}.csv')
        comparison_df.to_csv(comparison_csv_path)
        self._debug_print(f'Saving Results to {os.path.basename(comparison_csv_path)}')

        #
        #
        #
        #
        #

        self.logger.info("---------TEST RESULTS--------")
        self.logger.info("Test mode: %s", "tta" if use_tta else "standard")

        ### if oai
        if self.dataset_name == 'oai_nolandmarks':
            ### plot a graph comparing the angles to each reviewer and the model
            # (when there is nan drop the value)
            pred_l, pred_r = comparison_df['L HKA'].to_numpy(), comparison_df['R HKA'].to_numpy()
            true_hkas = np.stack(true_hkas, axis=0)
            hka_plot_name = f"/hka_comparison{file_suffix}.png"
            df = self.plot_hka_comparisons(pred_l, pred_r, true_hkas, save_name=hka_plot_name)
            wandb_plot_paths["hka_comparison"] = os.path.join(self.output_path, f"hka_comparison{file_suffix}.png")
            wandb_plot_paths["hka_bland_altman"] = os.path.join(self.output_path, f"hka_BlandA_comparison{file_suffix}.png")
                         
        if self.label_dir == '':
            MRE = [None, None, None,None]
            comparsion_summary_ls = None
        else:
            #Get mean values from comparison summary ls, landmark metrics
            comparsion_summary_ls, MRE = self.comparison_summary(comparison_df)
            for metric_name, metric_value in comparsion_summary_ls:
                if 'landmark radial error p' in metric_name.lower():
                    continue
                if 'ere p' in metric_name.lower():
                    continue
                if metric_name.lower() in {'alpha pred', 'alpha true', 'difference alpha'}:
                    continue
                wandb_summary[metric_name] = metric_value

        self.logger.info("MEAN VALUES (pix): {}".format(comparsion_summary_ls))
        self.logger.info("MRE: {} +/- {} pix".format(MRE[0], MRE[1]))
        self.logger.info("MRE no labrum: {} +/- {} pix".format(MRE[2], MRE[3]))
        ere_plot_path, ere_average = self._save_ere_bar_plot(comparison_df)
        if ere_plot_path is not None:
            wandb_plot_paths["ere_bar_by_point"] = ere_plot_path
        ere_average_plot_path = self._save_single_value_point_plot(
            value=ere_average,
            ylabel="Mean ERE",
            title="Mean ERE",
            filename="ere_mean_point.png",
            color="#d97706",
        )
        if ere_average_plot_path is not None:
            wandb_plot_paths["ere_mean_point"] = ere_average_plot_path
        mre_plot_path, mre_average = self._save_mre_bar_plot(comparison_df)
        if mre_plot_path is not None:
            wandb_plot_paths["mre_bar_by_point"] = mre_plot_path
        mre_average_plot_path = self._save_single_value_point_plot(
            value=mre_average,
            ylabel="Mean MRE",
            title="Mean MRE",
            filename="mre_mean_point.png",
            color="#0f766e",
        )
        if mre_average_plot_path is not None:
            wandb_plot_paths["mre_mean_point"] = mre_average_plot_path


        alpha_threshold_rows = None
        graf_i_vs_234_metrics = None
        graf_12_vs_34_metrics = None
        fhc_abnormal_vs_normal_metrics = None
        graf_fhc_i_vs_234_metrics = None
        graf_fhc_12_vs_34_metrics = None
        sdr_summary_mm = None

        #plot angles pred vs angles 
        if 'graf_angle_calc().graf_class_comparison' in self.cfg.TEST.COMPARISON_METRICS:
            if self.label_dir == '':   
                alpha_thresh_percentages,alpha_thresh_percentages_normalized= None, None
            else:
                alpha_thresh_percentages,alpha_thresh_percentages_normalized=self.alpha_thresholds(comparison_df)
                alpha_threshold_rows = self.alpha_thresholds_structured(comparison_df)

            self.logger.info("Alpha Thresholds: {}".format(alpha_thresh_percentages))
            self.logger.info("Alpha Thresholds Normalized: {}".format(alpha_thresh_percentages_normalized))
            wandb_summary["alpha_thresholds"] = alpha_thresh_percentages
            wandb_summary["alpha_thresholds_normalized"] = alpha_thresh_percentages_normalized

            #from df get class agreement metrics TP, TN, FN, FP
            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'class pred', 'class true',self.output_path)._get_metrics(group=True,groups=[('i'),('ii','iii/iv')])
            graf_i_vs_234_metrics = self._extract_classification_triplet(class_agreement)
            self.logger.info("Class Agreement - i vs ii/iii/iv : {}".format(class_agreement[4]))
            self.logger.info("Class Agreement - i vs ii/iii/iv : {}".format(class_agreement[5]))
            self.logger.info("Class Agreement - i vs ii/iii/iv : {}".format(class_agreement[6]))
            plot_path = self._save_class_agreement_bar_plot("class_agreement_i_vs_ii_iii_iv", class_agreement)
            if plot_path is not None:
                wandb_plot_paths["class_agreement_i_vs_ii_iii_iv_bar"] = plot_path

            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'class pred', 'class true', self.output_path)._get_metrics(group=True,groups=[('i','ii'),('iii/iv')])
            graf_12_vs_34_metrics = self._extract_classification_triplet(class_agreement)

            self.logger.info("Class Agreement - i/ii vs iii/iv : {}".format(class_agreement[4]))
            self.logger.info("Class Agreement - i/ii vs iii/iv : {}".format(class_agreement[5]))
            self.logger.info("Class Agreement - i/ii vs iii/iv : {}".format(class_agreement[6]))
            plot_path = self._save_class_agreement_bar_plot("class_agreement_i_ii_vs_iii_iv", class_agreement)
            if plot_path is not None:
                wandb_plot_paths["class_agreement_i_ii_vs_iii_iv_bar"] = plot_path



        if self.combine_graf_fhc==True and self.dataset_name == 'ddh':
        # #add fhc cols for normal and abnormal (n and a)
            comparison_df['fhc class pred']=comparison_df['fhc pred'].apply(lambda x: 'n' if x > .50 else 'a')
            comparison_df['fhc class true']=comparison_df['fhc true'].apply(lambda x: 'n' if x > .50 else 'a')

            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'fhc class pred', 'fhc class true',self.output_path, loc=self.test_output_dir_name)._get_metrics(group=True,groups=[('n'),('a')])
            fhc_abnormal_vs_normal_metrics = self._extract_classification_triplet(class_agreement)
            self.logger.info("Class Agreement FHC: {}".format(class_agreement))
            plot_path = self._save_class_agreement_bar_plot("class_agreement_fhc", class_agreement)
            if plot_path is not None:
                wandb_plot_paths["class_agreement_fhc_bar"] = plot_path


            ## Concensus of FHC and Graf
            comparison_df = self.validation.get_combined_agreement(comparison_df,'graf&fhc pred i_ii&iii&iv', 'graf&fhc true i_ii&iii&iv', groups=[('i'),('ii','iii/iv')])
            comparison_df = self.validation.get_combined_agreement(comparison_df,'graf&fhc pred i&ii_iii&iv', 'graf&fhc true i&ii_iii&iv', groups=[('i','ii'),('iii/iv')])
            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'graf&fhc pred i_ii&iii&iv', 'graf&fhc true i_ii&iii&iv',self.output_path, loc=self.test_output_dir_name)._get_metrics(group=True,groups=[('n'),('a')])
            graf_fhc_i_vs_234_metrics = self._extract_classification_triplet(class_agreement)
            self.logger.info("Class Agreement i vs ii/iii/iv GRAF&FHC: {}".format(class_agreement))
            plot_path = self._save_class_agreement_bar_plot("class_agreement_graf_fhc_i_vs_ii_iii_iv", class_agreement)
            if plot_path is not None:
                wandb_plot_paths["class_agreement_graf_fhc_i_vs_ii_iii_iv_bar"] = plot_path
            class_agreement = class_agreement_metrics(self.dataset_name, comparison_df, 'graf&fhc pred i&ii_iii&iv', 'graf&fhc true i&ii_iii&iv',self.output_path, loc=self.test_output_dir_name)._get_metrics(group=True,groups=[('n'),('a')])
            graf_fhc_12_vs_34_metrics = self._extract_classification_triplet(class_agreement)
            self.logger.info("Class Agreement i/ii vs iii/iv GRAF&FHC: {}".format(class_agreement))
            plot_path = self._save_class_agreement_bar_plot("class_agreement_graf_fhc_i_ii_vs_iii_iv", class_agreement)
            if plot_path is not None:
                wandb_plot_paths["class_agreement_graf_fhc_i_ii_vs_iii_iv_bar"] = plot_path
            
            comparison_df.to_csv(comparison_csv_path)
            self._debug_print(f'Saving Results to {os.path.basename(comparison_csv_path)}')
        
        
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
                    wandb_summary["sdr_all_landmarks"] = ",".join([str(round(x, 2)) for x in sdr_summary])
                    wandb_summary["sdr_all_landmarks_no_labrum"] = ",".join([str(round(x, 2)) for x in sdr_summary_nolab])
                    sdr_plot_path, sdr_plot_average = self._save_sdr_bar_plot(np.round(sdr_summary, 2))
                    if sdr_plot_path is not None:
                        wandb_plot_paths["sdr_all_landmarks"] = sdr_plot_path

                    for threshold, value in zip(self.sdr_thresholds, sdr_summary):
                        threshold_key = str(threshold).replace('.', '_')
                        wandb_summary[f"sdr_all_landmarks_{threshold_key}_{self.sdr_units}"] = round(float(value), 2)

                    for threshold, value in zip(self.sdr_thresholds, sdr_summary_nolab):
                        threshold_key = str(threshold).replace('.', '_')
                        wandb_summary[f"sdr_all_landmarks_no_labrum_{threshold_key}_{self.sdr_units}"] = round(float(value), 2)

                    sdr_summary_mm = []
                    for mm_threshold in [0.5, 1, 2, 4]:
                        per_landmark_sdr = []
                        for i in range(self.num_landmarks):
                            col = 'landmark radial error p'+str(i+1)
                            sdr_stats_mm, _ = landmark_overall_metrics(self.pixel_size, 'mm').get_sdr_statistics(comparison_df[col], [mm_threshold])
                            per_landmark_sdr.append(sdr_stats_mm[0])
                        sdr_summary_mm.append(round(float(np.mean(per_landmark_sdr)), 2))

                    if 'fhc pred' in comparison_df.columns and 'fhc true' in comparison_df.columns:
                        fhc_abs_mean_diff = round(
                            (pd.to_numeric(comparison_df['fhc pred'], errors='coerce') - pd.to_numeric(comparison_df['fhc true'], errors='coerce'))
                            .abs()
                            .mean(),
                            3,
                        )
                        if not np.isnan(fhc_abs_mean_diff):
                            self.logger.info('FHC ABSOLUTE MEAN DIFF:{}'.format(fhc_abs_mean_diff))
                            wandb_summary["fhc_abs_mean_diff"] = fhc_abs_mean_diff
                   
                   
                    #plot angles pred vs angles 
                    if 'graf_angle_calc().graf_class_comparison' in self.cfg.TEST.COMPARISON_METRICS:
                        alpha_abs_mean_diff = round(comparison_df['difference alpha'].apply(abs).mean(), 3)
                        self.logger.info('ALPHA ABSOLUTE MEAN DIFF:{}'.format(alpha_abs_mean_diff))
                        wandb_summary["alpha_abs_mean_diff"] = alpha_abs_mean_diff

                except:
                    raise ValueError('Check Landmark radial errors are calcuated')

        evaluation_summary_table = None
        evaluation_summary_heatmap = None
        if self.wandb_enabled and self.label_dir != '':
            evaluation_summary_row = self._build_experiment_summary_row(
                comparison_df=comparison_df,
                mre_stats=MRE,
                sdr_mm=sdr_summary_mm,
                alpha_threshold_rows=alpha_threshold_rows,
                graf_i_vs_234=graf_i_vs_234_metrics,
                graf_12_vs_34=graf_12_vs_34_metrics,
                fhc_abnormal_vs_normal=fhc_abnormal_vs_normal_metrics,
                graf_fhc_i_vs_234=graf_fhc_i_vs_234_metrics,
                graf_fhc_12_vs_34=graf_fhc_12_vs_34_metrics,
            )
            evaluation_summary_table = wandb.Table(
                columns=list(evaluation_summary_row.keys()),
                data=[[evaluation_summary_row[column] for column in evaluation_summary_row.keys()]],
            )
            evaluation_summary_heatmap = self._build_experiment_summary_heatmap(evaluation_summary_row)

        self._log_wandb_results(
            summary_metrics=wandb_summary,
            failure_images=failure_images,
            categorized_wrong_case_paths=categorized_wrong_case_paths,
            plot_paths=wandb_plot_paths,
            comparison_csv_path=comparison_csv_path,
            comparison_df=comparison_df,
            evaluation_summary_table=evaluation_summary_table,
            evaluation_summary_heatmap=evaluation_summary_heatmap,
            wandb_namespace=wandb_namespace,
        )

        return {
            "summary_row": evaluation_summary_row if self.label_dir != '' else None,
            "comparison_csv_path": comparison_csv_path,
            "output_dir": self.test_output_dir_name,
            "wandb_namespace": wandb_namespace,
            "use_tta": bool(use_tta),
        }
