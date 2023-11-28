import argparse
from support import helper
from torch.utils.data import DataLoader
from preprocessing import dataloader
from main import test
import numpy as np
import os
import torch
torch.cuda.empty_cache() 

def parse_args():
    parser = argparse.ArgumentParser(description='Train a network to detect landmarks')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg_name = args.cfg

    # print the arguments into the log
    help = helper(cfg_name)
    logger = help.setup_logger()
    cfg = help._get_cfg()

    #preprocess data (put into a numpy array)
    test_dataset=dataloader(cfg,'testing')
    help._dataset_shape(test_dataset)

    #load data into data loader (imports all data into a dataloader)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, drop_last = True)
    
    test(cfg,logger).run(test_dataloader)

if __name__ == '__main__':
    main()