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


def _resolve_cfg_name(cfg_arg):
    cfg_basename = os.path.basename(str(cfg_arg).strip())
    if cfg_basename.lower().endswith(".yaml"):
        return cfg_basename[:-5]
    return cfg_basename


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


def _sanitize_namespace(text):
    text = str(text).strip().lower()
    safe = []
    for ch in text:
        if ch.isalnum():
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "test"


def _build_eval_cfgs(cfg):
    eval_datasets = getattr(cfg.TEST, "EVAL_DATASETS", [])
    if not eval_datasets:
        return [(None, cfg)]

    eval_cfgs = []
    for idx, dataset_cfg in enumerate(eval_datasets):
        run_cfg = cfg.clone()
        run_cfg.defrost()

        dataset_name = dataset_cfg.get("DATASET_NAME", getattr(run_cfg.INPUT_PATHS, "DATASET_NAME", ""))
        partition = dataset_cfg.get("PARTITION", getattr(run_cfg.INPUT_PATHS, "PARTITION", ""))
        images = dataset_cfg.get("IMAGES", getattr(run_cfg.INPUT_PATHS, "IMAGES", ""))
        labels = dataset_cfg.get("LABELS", getattr(run_cfg.INPUT_PATHS, "LABELS", ""))
        meta_path = dataset_cfg.get("META_PATH", getattr(run_cfg.INPUT_PATHS, "META_PATH", ""))
        dcms = dataset_cfg.get("DCMS", getattr(run_cfg.INPUT_PATHS, "DCMS", ""))
        alias = dataset_cfg.get("NAME") or dataset_cfg.get("ALIAS") or dataset_name or f"eval_{idx+1}"

        run_cfg.INPUT_PATHS.DATASET_NAME = dataset_name
        run_cfg.INPUT_PATHS.PARTITION = partition
        run_cfg.INPUT_PATHS.IMAGES = images
        run_cfg.INPUT_PATHS.LABELS = labels
        run_cfg.INPUT_PATHS.META_PATH = meta_path
        run_cfg.INPUT_PATHS.DCMS = dcms
        run_cfg.freeze()

        eval_cfgs.append((alias, run_cfg))

    return eval_cfgs


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
    resolved_cfg_name = _resolve_cfg_name(cfg_name)

    # print the arguments into the log
    help = helper(resolved_cfg_name, 'test')
    logger = help.setup_logger()
    cfg = help._get_cfg()
    wandb_cfg = _cfg_to_dict(cfg)

    run_name = resolved_cfg_name if not args.wandb_run_suffix else f"{resolved_cfg_name}_{args.wandb_run_suffix}"
    run_group = args.wandb_group or resolved_cfg_name
    run_tags = ["test", cfg.INPUT_PATHS.DATASET_NAME, cfg.MODEL.NAME]
    run_tags.extend(_collect_dataset_tags(cfg))
    if args.wandb_tags:
        run_tags.extend([tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()])
    run_tags = list(dict.fromkeys(run_tags))
    
    wandb.init(
        project=args.wandb_project or "MultimodalMedicalPredictions",
        group=run_group,
        job_type="test",
        config=wandb_cfg,
        name=run_name,
        tags=run_tags,
    )
    wandb.config.update(wandb_cfg)  # ensure all config parameters are logged

    try:
        eval_cfgs = _build_eval_cfgs(cfg)
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

        for alias, eval_cfg in eval_cfgs:
            log_name = alias or "default"
            logger.info("Running test evaluation for %s", log_name)
            test_dataset = dataloader(eval_cfg, 'testing')
            help._dataset_shape(test_dataset)
            test_dataloader = DataLoader(test_dataset, **dataloader_kwargs)
            tester = test(eval_cfg, logger)

            if alias is None:
                output_dir_name = tester.base_test_output_dir_name
                wandb_namespace = "test"
            else:
                output_dir_name = _sanitize_namespace(alias)
                wandb_namespace = f"test_{_sanitize_namespace(alias)}"
            tester.run(
                test_dataloader,
                use_tta=False,
                output_dir_name=output_dir_name,
                wandb_namespace=wandb_namespace,
            )

            if bool(getattr(eval_cfg.TEST, "TEST_TIME_AUG", False)):
                logger.info("Running test-time augmentation evaluation for %s", log_name)
                tester.run(
                    test_dataloader,
                    use_tta=True,
                    output_dir_name=f"{output_dir_name}_tta",
                    wandb_namespace=f"{wandb_namespace}_tta",
                )
    finally:
        if wandb.run is not None:
            wandb.finish()

if __name__ == '__main__':
    main()
