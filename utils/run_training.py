import argparse
from support import helper
from torch.utils.data import Dataset, DataLoader
from preprocessing import dataloader
from main import training
import numpy as np
import os
import torch
from main.loss import plot_all_loss
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

    for epoch in range(1, max_epochs+1):  
        train = training(cfg, logger)
        train_loss = train.train_meta(train_dataloader, epoch)
        val_loss = train.val_meta(val_dataloader, epoch)
        losses.append([train_loss, val_loss])
    losses = np.array(losses).T
    plot_all_loss(losses, max_epochs, cfg.OUTPUT_PATH)

    logger.info('-----------Saving Models-----------')
    best_model=train._get_network()
    run = 1 #would be multiple if ensembles
    model_run_path = os.path.join(cfg.OUTPUT_PATH, "model:{}".format(run))
    if not os.path.exists(model_run_path):
        os.makedirs(model_run_path)
    for model_idx in range(len(run)):
        #our_model = ensemble[model_idx]
        save_model_path = os.path.join(model_run_path, "{}_model_run:{}_idx:{}.pth".format(yaml_file_name, run, model_idx))
        logger.info("Saving Model {}'s State Dict to {}".format(model_idx, save_model_path))
        torch.save(best_model.state_dict(), save_model_path)

if __name__ == '__main__':
    main()