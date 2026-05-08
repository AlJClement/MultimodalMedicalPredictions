import os
import time
import logging
import torch
import re
from .default_config import get_cfg_defaults
from datetime import datetime

class helper():
    def __init__(self, cfg_name, set) -> None:
        '''Helper function contains functions to 'help' the overall organization of the outputs/repos
        ex. logger functions etc'''
        self.cfg=self.load_cfg(os.path.join(os.getcwd(),'experiments',cfg_name+'.yaml'))#
        self.cfg_name = cfg_name
        self.log_path = self.cfg.OUTPUT_PATH +'/'+cfg_name+"_"+self._get_datetime()+set+'.txt'
        self.dataset_name = self.cfg.INPUT_PATHS.DATASET_NAME
        self.output_path = self.cfg.OUTPUT_PATH
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        pass

    def _get_datetime(self):
        return str(datetime.now().year)+str(datetime.now().month)+str(datetime.now().day)
    
    def _get_cfg(self):
        return self.cfg
    
    def load_cfg(self, cfg_path):
        # get config
        cfg = get_cfg_defaults()
        cfg.merge_from_file(cfg_path)
        cfg.freeze()
        return cfg

    def setup_logger(self):
        logging.basicConfig(filename=self.log_path,
                            format='%(message)s')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)
        return logger

    def _infer_split_suffix(self):
        candidates = [
            str(getattr(self.cfg.INPUT_PATHS, "PARTITION", "")),
            str(getattr(self.cfg, "OUTPUT_PATH", "")),
            str(self.cfg_name),
        ]

        for candidate in candidates:
            match = re.search(r"0\.\d{5}", candidate)
            if match:
                return match.group(0)

        return None

    def resolve_reference_csv_path(self, metric_name):
        metric_key = str(metric_name).strip().lower()
        if metric_key == "alpha":
            explicit_path = str(getattr(self.cfg.INPUT_PATHS, "ALPHA_CSV", "")).strip()
            default_dir = "ALPHAS"
            default_file = "alpha_range.csv"
        elif metric_key == "fhc":
            explicit_path = str(getattr(self.cfg.INPUT_PATHS, "FHC_CSV", "")).strip()
            default_dir = "FHC"
            default_file = "fhc_range.csv"
        else:
            raise ValueError(f"Unknown metric_name '{metric_name}'. Expected 'alpha' or 'fhc'.")

        if explicit_path:
            return explicit_path

        split_suffix = self._infer_split_suffix()
        if split_suffix:
            return os.path.join(".", "output", f"{default_dir}_{split_suffix}", default_file)

        return os.path.join(".", "output", default_dir, default_file)
    
    def save_model(self, logger, model_name, ensemble):
        model_run_path = os.path.join(self.output_path, "run:{}_models".format(model_name))

        if not os.path.exists(model_run_path):
            os.makedirs(model_run_path)
        for model_idx in range(len(ensemble)):
            our_model = ensemble[model_idx]
            save_model_path = os.path.join(model_run_path, "{}_model_run:{}_idx:{}.pth".format(self.dataset_name, model_idx))
            logger.info("Saving Model {}'s State Dict to {}".format(model_idx, save_model_path))
            torch.save(our_model.state_dict(), save_model_path)

    @staticmethod
    def _get_network_parameters(net):
        print('net dev:',next(net.parameters()).device)
        # Calculate the number of traininable params
        params = [p.numel() for p in net.parameters()]
        params_total = sum(params)
        print('Trainable params: ', params_total)

    @staticmethod
    def _dataset_shape(dataset_arr):
        #dataloader function contains 
        print('Input shape of data:', len(dataset_arr.data))        
        print('Input shape of annotations:', len(dataset_arr.target))
        print('Input shape of meta:', len(dataset_arr.meta))
