import argparse
from support import helper
from torch.utils.data import Dataset, DataLoader
from preprocessing import dataloader
from main import training
from main import validation
import numpy as np
import os
import torch
torch.cuda.empty_cache() 
from main.loss import plot_all_loss
import numpy as _np
if not hasattr(_np, "bool"):
    _np.bool = bool

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
    help = helper(cfg_name, 'training')
    logger = help.setup_logger()
    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # print the configuration into the log
    logger.info("-----------Configuration-----------")
    cfg = help._get_cfg()
    logger.info(cfg)
    logger.info("")

    #preprocess data (put into a numpy array)
    train_dataset=dataloader(cfg,'training')
    help._dataset_shape(train_dataset)

    val_dataset=dataloader(cfg,'validation')    
    help._dataset_shape(val_dataset)

    #load data into data loader (imports all data into a dataloader)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, drop_last = False)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, drop_last = False)

    losses = []
    max_epochs = cfg.TRAIN.EPOCHS
    L2_REG = cfg.TRAIN.L2_REG

    train = training(cfg, logger,L2_REG)
    net = train._get_network()
    validate= validation(cfg,logger,net, L2_REG)
    
    for epoch in range(1, max_epochs+1):  
        train_loss = train.train_meta(train_dataloader, epoch)
        val_loss = validate.val_meta(val_dataloader, epoch)
        losses.append([train_loss, val_loss])

    losses = np.array(losses).T
    plot_all_loss(losses, max_epochs, cfg.OUTPUT_PATH)

    logger.info('-----------Saving Models-----------')
    best_model=train._get_network()
    runs = [1] #would be multiple if ensembles

    for model_idx in runs:
        model_run_path = os.path.join(cfg.OUTPUT_PATH, "model:{}".format(model_idx))
        if not os.path.exists(model_run_path):
            os.makedirs(model_run_path)
        #our_model = ensemble[model_idx]
        save_model_path = os.path.join(model_run_path, "_model_run:{}_idx.pth".format(model_idx))
        logger.info("Saving Model {}'s State Dict to {}".format(model_idx, save_model_path))
        torch.save(best_model.state_dict(), save_model_path)

if __name__ == '__main__':
    main()