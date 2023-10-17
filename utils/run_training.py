import argparse
from support import helper
from torch.utils.data import Dataset, DataLoader
from preprocessing import dataloader

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
    #output is : [im_arr, annotation_arr, meta_arr]
    train_dataset=dataloader(cfg).get_numpy_dataset('training')
    val_dataset=dataloader(cfg).get_numpy_dataset('validation')

    #load data into data loader
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, drop_last=True)


if __name__ == '__main__':
    main()