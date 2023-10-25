import argparse
from support import helper
from torch.utils.data import Dataset, DataLoader
from preprocessing import dataloader
from main import training
import numpy as np

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
        train = training(cfg)
        train_loss = train.train_meta(train_dataloader, epoch)
        val_loss = train.val_meta(val_dataloader, epoch)
        losses.append([train_loss, val_loss])

    losses = np.array(losses).T

    # its = np.linspace(1, max_epochs, max_epochs)
    # plt.figure()
    # plt.plot(its, losses[0,:])
    # plt.plot(its, losses[1,:])
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend(['Train', 'Validation'])



if __name__ == '__main__':
    main()