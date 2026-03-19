#!/usr/bin/env python3
import argparse
import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from support import helper
from preprocessing import dataloader
from main import training

# import the refactored class (name in file is TemperatureScaling)
from main.temperature_scaling import TemperatureScaling as TemperatureScaler

torch.cuda.empty_cache()
from main.evaluation_helper import evaluation_helper



def parse_args():
    parser = argparse.ArgumentParser(description='Run temperature scaling on a trained model')

    parser.add_argument('--cfg',
                        help='Path to configuration file',
                        required=True,
                        type=str)

    parser.add_argument('--model_path',
                        help='Optional explicit path to a .pth/.pt checkpoint (overrides search in cfg.OUTPUT_PATH/model:1)',
                        required=False,
                        type=str,
                        default=None)

    parser.add_argument('--tolerance_px',
                        help='Optional radial tolerance in pixels for reliability diagram correctness (overrides cfg.EVAL.TOLERANCE_PX)',
                        required=False,
                        type=float,
                        default=None)

    args = parser.parse_args()
    return args


def find_checkpoint_in_folder(folder_path):
    if folder_path is None:
        return None
    if os.path.isfile(folder_path) and folder_path.endswith(('.pth', '.pt')):
        return folder_path
    if not os.path.isdir(folder_path):
        return None
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith('.pth') or fname.endswith('.pt'):
            return os.path.join(folder_path, fname)
    return None


def split_dataset_half(dataset):
    n = len(dataset)
    idx1 = list(range(0, n // 2))
    idx2 = list(range(n // 2, n))
    return Subset(dataset, idx1), Subset(dataset, idx2)


def main():
    args = parse_args()

    # Setup helper + config
    help = helper(args.cfg, 'temperature_scaling')
    logger = help.setup_logger()

    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    cfg = help._get_cfg()

    logger.info("-----------Configuration-----------")
    logger.info(cfg)
    logger.info("")

    # Build validation dataset (we'll split this into calibration train/val halves)
    full_val_dataset = dataloader(cfg, 'validation')
    help._dataset_shape(full_val_dataset)

    # split into two halves: first half used to fit temperature, second half used for validation (ECE)
    # try:
    #     train_calib_ds, val_calib_ds = split_dataset_half(full_val_dataset)
    #     logger.info("Split validation dataset into %d (train calib) / %d (val calib)",
    #                 len(train_calib_ds), len(val_calib_ds))
    # except Exception as e:
    #     # fallback: use full set for both (not ideal)
    #     logger.warning("Could not split validation dataset: %s — falling back to using full validation set for calibration", e)
    # 
    train_calib_ds = full_val_dataset
    val_calib_ds = full_val_dataset

    train_calib_loader = DataLoader(train_calib_ds, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=False)
    val_calib_loader = DataLoader(val_calib_ds, batch_size=1, shuffle=False, drop_last=False)

    # Initialize training object only to build model architecture
    trainer = training(cfg, logger, cfg.TRAIN.L2_REG)
    net = trainer._get_network()
    device = trainer.device
    net.to(device)
    net.eval()

    # LOAD EXISTING MODEL (explicit arg overrides)
    if args.model_path:
        checkpoint_path = args.model_path
    else:
        model_folder = os.path.join(cfg.OUTPUT_PATH, "model:1")
        checkpoint_path = find_checkpoint_in_folder(model_folder)
        # fallback candidates
        if checkpoint_path is None:
            candidates = [
                os.path.join(cfg.OUTPUT_PATH, "_model_run:1_idx.pth"),
                os.path.join(cfg.OUTPUT_PATH, "model_with_temperature.pth"),
                os.path.join(cfg.OUTPUT_PATH, "best_model.pth"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    checkpoint_path = c
                    break

    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        logger.error("No checkpoint found. Provide --model_path or ensure a .pth is present in cfg.OUTPUT_PATH/model:1")
        raise FileNotFoundError("Checkpoint not found.")

    logger.info(f"Loading model from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    # handle dict wrappers
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
        sd = ckpt.get("state_dict", ckpt.get("model_state_dict"))
    else:
        sd = ckpt
    net.load_state_dict(sd)
    net.to(device)
    net.eval()
    logger.info("Model loaded successfully.")

    # Prepare save directory
    save_dir = os.path.join(cfg.OUTPUT_PATH, "temperature_scaled")
    os.makedirs(save_dir, exist_ok=True)

    # Create TemperatureScaler wrapper
    scaler = TemperatureScaler(cfg, net, device=device, init_temp=float(getattr(cfg.TRAIN, "TEMPERATURE_SCALING_INIT_TEMP", 1.0)))
    fit = 'to ere'
    if fit == 'to ere':
        logger.info("Evaluation helper found — running fit_with_evaluation (ECE + reliability diagrams).")
        result = scaler.fit_with_evaluation(
            train_loader=train_calib_loader,
            val_loader=val_calib_loader,
            save_dir=save_dir,
            n_epochs=int(getattr(cfg.TRAIN, "TEMPERATURE_SCALING_EPOCHS", 30)),
            lr=float(getattr(cfg.TRAIN, "TEMPERATURE_SCALING_LR", 1e-3)),
            weight_decay=0.01,
            tol_px=1.5,
            n_bins=int(getattr(cfg.TRAIN, "TEMPERATURE_SCALING_N_BINS", 10)),
            device=device,
            logger=logger,
            csv_name="temperature_scaling_history.csv"
        )

        final_T = result.get("final_temp", None)
        logger.info("Temperature scaling (with evaluation) finished. Final temperature: %s", str(final_T))
        logger.info("History and artifacts saved in: %s", save_dir)

    else:
        # simple fit on the val_dataloader
        logger.info("No evaluation_helper available — running simple fit() on validation set (no ECE / reliability curves).")
        final_T = scaler.fit(
            fine_tune_loader=val_calib_loader,
            val_loader=None,
            n_epochs=int(getattr(cfg.TRAIN, "TEMPERATURE_SCALING_EPOCHS", 30)),
            lr=float(getattr(cfg.TRAIN, "TEMPERATURE_SCALING_LR", 1e-2)),
            weight_decay=0.0,
            device=device,
            verbose=True
        )
        logger.info("Temperature scaling (simple) finished. Final temperature: %s", str(final_T))

        # Save a minimal CSV and plot from scaler.history if available
        try:
            hist = scaler.history
            # if scaler has history and a plotting function was used, save a simple CSV
            csv_path = os.path.join(save_dir, "temperature_scaling_history_simple.csv")
            import csv as _csv
            with open(csv_path, "w", newline="") as f:
                w = _csv.writer(f)
                # choose keys that exist
                keys = ["train_loss", "val_loss", "temperature"]
                # try historic fields (older refactors might store different names)
                if "train_nll" in hist:
                    keys = ["train_nll", "val_nll", "temperature"]
                w.writerow(keys)
                n = max(len(hist.get(k, [])) for k in keys)
                for i in range(n):
                    row = [hist.get(k, [None] * n)[i] if i < len(hist.get(k, [])) else None for k in keys]
                    w.writerow(row)
            logger.info("Saved simple history CSV to %s", csv_path)
        except Exception as e:
            logger.warning("Failed to save simple history CSV: %s", e)

    # Save final model state_dict (with attached temperature param if present)
    model_save_path = os.path.join(save_dir, "model_with_temperature.pth")
    try:
        torch.save(net.state_dict(), model_save_path)
        logger.info("Saved calibrated model to: %s", model_save_path)
    except Exception as e:
        logger.warning("Failed to save calibrated model: %s", e)

    # Save temperature separately
    temp_save_path = os.path.join(save_dir, "temperature.pt")
    try:
        if hasattr(net, "temperatures"):
            torch.save({"temperatures": net.temperatures.detach().cpu()}, temp_save_path)
        else:
            # compute readable float temperature
            tval = float(torch.nn.functional.softplus(scaler._get_temperature()).item())
            torch.save({"temperature": tval}, temp_save_path)
        logger.info("Saved temperature to: %s", temp_save_path)
    except Exception as e:
        logger.warning("Failed to save temperature: %s", e)


if __name__ == '__main__':
    main()