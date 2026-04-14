import argparse
from support import helper
from torch.utils.data import DataLoader
from preprocessing import dataloader
from main import test
import numpy as np
import os
import torch
torch.cuda.empty_cache() 
import wandb

def _cfg_to_dict(cfg_node):
    if hasattr(cfg_node, "items"):
        return {key: _cfg_to_dict(value) for key, value in cfg_node.items()}
    if isinstance(cfg_node, (list, tuple)):
        return [_cfg_to_dict(value) for value in cfg_node]
    if isinstance(cfg_node, np.ndarray):
        return cfg_node.tolist()
    return cfg_node


def _collect_dataset_tags(cfg):
    tags = set()
    candidates = [
        getattr(cfg.INPUT_PATHS, "DATASET_NAME", ""),
        getattr(cfg.INPUT_PATHS, "IMAGES", ""),
        getattr(cfg.INPUT_PATHS, "LABELS", ""),
        getattr(cfg.INPUT_PATHS, "META_PATH", ""),
        getattr(cfg.INPUT_PATHS, "PARTITION", ""),
    ]

    for value in candidates:
        text = str(value).upper()
        if "MKUH" in text:
            tags.add("MKUH")
        if "RNOH" in text:
            tags.add("RNOH")

    return sorted(tags)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network to detect landmarks')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)
    parser.add_argument('--wandb-project',
                        help='Optional W&B project override',
                        required=False,
                        type=str)
    parser.add_argument('--wandb-group',
                        help='Optional W&B group for collecting multiple test runs together',
                        required=False,
                        type=str)
    parser.add_argument('--wandb-run-suffix',
                        help='Optional suffix appended to the W&B run name',
                        required=False,
                        type=str)
    parser.add_argument('--wandb-tags',
                        help='Optional comma-separated W&B tags',
                        required=False,
                        type=str)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg_name = args.cfg

    # print the arguments into the log
    help = helper(cfg_name, 'test')
    logger = help.setup_logger()
    cfg = help._get_cfg()
    wandb_cfg = _cfg_to_dict(cfg)

    cfg_stem = os.path.basename(cfg_name).split('.')[0]
    test_track_name = args.wandb_project or f"{cfg_stem}_test"
    run_name = cfg_stem if not args.wandb_run_suffix else f"{cfg_stem}_{args.wandb_run_suffix}"
    run_group = args.wandb_group or cfg_stem
    run_tags = ["test", cfg.INPUT_PATHS.DATASET_NAME, cfg.MODEL.NAME]
    run_tags.extend(_collect_dataset_tags(cfg))
    if args.wandb_tags:
        run_tags.extend([tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()])
    run_tags = list(dict.fromkeys(run_tags))
    
    wandb.init(
        project="MultimodalMedicalPredictions",
        group=run_group,
        job_type="test",
        config=wandb_cfg,
        name=run_name,
        tags=run_tags,
    )
    wandb.config.update(wandb_cfg)  # ensure all config parameters are logged

    #preprocess data (put into a numpy array)
    test_dataset=dataloader(cfg,'testing')
    help._dataset_shape(test_dataset)

    num_workers = int(getattr(cfg.TEST, "NUM_WORKERS", 0))
    pin_memory = bool(getattr(cfg.TEST, "PIN_MEMORY", False))
    persistent_workers = bool(getattr(cfg.TEST, "PERSISTENT_WORKERS", False))
    prefetch_factor = getattr(cfg.TEST, "PREFETCH_FACTOR", None)
    dataloader_kwargs = {
        "batch_size": cfg.TEST.BATCH_SIZE,
        "shuffle": False,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)

    logger.info(f"Test DataLoader settings: {dataloader_kwargs}")

    #load data into data loader (imports all data into a dataloader)
    test_dataloader = DataLoader(test_dataset, **dataloader_kwargs)
    
    try:
        test(cfg,logger).run(test_dataloader)
    finally:
        if wandb.run is not None:
            wandb.finish()

if __name__ == '__main__':
    main()
