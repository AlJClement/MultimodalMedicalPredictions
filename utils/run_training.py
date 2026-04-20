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
from main.loss import plot_all_loss, plot_metric_history
import wandb
import numpy as _np
if not hasattr(_np, "bool"):
    _np.bool = bool


def _cfg_to_dict(cfg_node):
    if hasattr(cfg_node, "items"):
        return {key: _cfg_to_dict(value) for key, value in cfg_node.items()}
    if isinstance(cfg_node, (list, tuple)):
        return [_cfg_to_dict(value) for value in cfg_node]
    if isinstance(cfg_node, np.ndarray):
        return cfg_node.tolist()
    return cfg_node


def _resolve_cfg_name(cfg_arg):
    cfg_basename = os.path.basename(str(cfg_arg).strip())
    if cfg_basename.lower().endswith(".yaml"):
        return cfg_basename[:-5]
    return cfg_basename


def _configure_wandb_dirs(output_path):
    base_dir = os.path.join(os.getcwd(), "wandb")
    defaults = {
        "WANDB_CONFIG_DIR": os.path.join(base_dir, "config"),
        "WANDB_DATA_DIR": os.path.join(base_dir, "data"),
        "WANDB_CACHE_DIR": os.path.join(base_dir, "cache"),
        "WANDB_DIR": os.path.join(base_dir, "runs"),
    }

    for env_key, default_path in defaults.items():
        target_path = os.environ.get(env_key, default_path)
        os.makedirs(target_path, exist_ok=True)
        os.environ.setdefault(env_key, target_path)

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
                        help='Optional W&B group for collecting multiple training runs together',
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
    resolved_cfg_name = _resolve_cfg_name(cfg_name)

    # print the arguments into the log
    help = helper(resolved_cfg_name, 'training')
    logger = help.setup_logger()
    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # print the configuration into the log
    logger.info("-----------Configuration-----------")
    cfg = help._get_cfg()
    _configure_wandb_dirs(cfg.OUTPUT_PATH)
    logger.info(cfg)
    logger.info("")
    wandb_cfg = _cfg_to_dict(cfg)

    run_name = resolved_cfg_name if not args.wandb_run_suffix else f"{resolved_cfg_name}_{args.wandb_run_suffix}"
    run_group = args.wandb_group or resolved_cfg_name
    run_tags = ["train", cfg.INPUT_PATHS.DATASET_NAME, cfg.MODEL.NAME]
    encoder_name = str(getattr(cfg.MODEL, "ENCODER_NAME", "")).strip()
    if encoder_name:
        run_tags.append(encoder_name)
    if args.wandb_tags:
        run_tags.extend([tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()])
    run_tags = list(dict.fromkeys(run_tags))

    wandb.init(
        project=args.wandb_project or "MultimodalMedicalPredictions",
        group=run_group,
        job_type="train",
        config=wandb_cfg,
        name=run_name,
        tags=run_tags,
    )
    wandb.config.update(wandb_cfg)

    #preprocess data (put into a numpy array)
    train_dataset=dataloader(cfg,'training')
    help._dataset_shape(train_dataset)

    val_dataset=dataloader(cfg,'validation')    
    help._dataset_shape(val_dataset)

    num_workers = int(getattr(cfg.TRAIN, "NUM_WORKERS", 4))
    pin_memory = bool(getattr(cfg.TRAIN, "PIN_MEMORY", torch.cuda.is_available()))
    persistent_workers = bool(getattr(cfg.TRAIN, "PERSISTENT_WORKERS", num_workers > 0))
    prefetch_factor = getattr(cfg.TRAIN, "PREFETCH_FACTOR", None)
    dataloader_kwargs = {
        "batch_size": cfg.TRAIN.BATCH_SIZE,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)

    #load data into data loader (imports all data into a dataloader)
    train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
    val_dataloader = DataLoader(val_dataset, **dataloader_kwargs)

    losses = []
    val_mres = []
    val_mres_no_labrum = []
    max_epochs = cfg.TRAIN.EPOCHS
    L2_REG = cfg.TRAIN.L2_REG

    train = training(cfg, logger,L2_REG)
    net = train._get_network()
    validate= validation(cfg,logger,net, L2_REG)

    try:
        for epoch in range(1, max_epochs+1):  
            train_loss = float(train.train_meta(train_dataloader, epoch))
            train_mre = float(getattr(train, "last_mre", float("nan")))
            train_mre_std = float(getattr(train, "last_mre_std", float("nan")))
            val_loss = float(validate.val_meta(val_dataloader, epoch))
            val_mre = float(getattr(validate, "last_mre", float("nan")))
            val_mre_no_labrum = float(getattr(validate, "last_mre_no_labrum", float("nan")))
            losses.append([train_loss, val_loss])
            val_mres.append(val_mre)
            val_mres_no_labrum.append(val_mre_no_labrum)

            logger.info(
                "Epoch %s/%s - train_loss: %.6f - val_loss: %.6f - val_mre: %.4f",
                epoch,
                max_epochs,
                train_loss,
                val_loss,
                val_mre,
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_mre": train_mre,
                    "val_loss": val_loss,
                    "val_mre": val_mre,
                },
                step=epoch,
            )
            if np.isfinite(train_mre_std):
                wandb.log({"train_mre_std": train_mre_std}, step=epoch)
            if np.isfinite(val_mre_no_labrum):
                wandb.log({"val_mre_no_labrum": val_mre_no_labrum}, step=epoch)

        losses = np.array(losses).T
        plot_all_loss(losses, max_epochs, cfg.OUTPUT_PATH)
        plot_metric_history(val_mres, cfg.OUTPUT_PATH, "mre_fig.png", "MRE (pixels)", legend_label="Validation")
        if any(np.isfinite(np.asarray(val_mres_no_labrum, dtype=float))):
            plot_metric_history(
                val_mres_no_labrum,
                cfg.OUTPUT_PATH,
                "mre_no_labrum_fig.png",
                "MRE No Labrum (pixels)",
                legend_label="Validation",
            )

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
    finally:
        if wandb.run is not None:
            wandb.finish()

if __name__ == '__main__':
    main()
