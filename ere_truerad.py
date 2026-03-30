#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
UTILS_DIR = ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from support.helper import helper
from preprocessing.augmentation import Augmentation
from preprocessing.dataloader import dataloader
from main.training import training
from main.evaluation_helper import evaluation_helper
from main.comparison_metrics.landmark_metrics import landmark_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ERE and mean radial error on validation and test sets."
    )
    parser.add_argument("--cfg", required=True, type=str, help="Config name inside experiments/ without .yaml")
    parser.add_argument(
        "--model_path",
        required=False,
        type=str,
        default=None,
        help="Optional explicit checkpoint path. If omitted, search cfg.OUTPUT_PATH/model:1 first.",
    )
    parser.add_argument(
        "--batch_size",
        required=False,
        type=int,
        default=None,
        help="Optional dataloader batch size override.",
    )
    return parser.parse_args()


def find_checkpoint_in_folder(folder_path: str | None) -> str | None:
    if folder_path is None:
        return None
    if os.path.isfile(folder_path) and folder_path.endswith((".pth", ".pt")):
        return folder_path
    if not os.path.isdir(folder_path):
        return None
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".pth") or fname.endswith(".pt"):
            return os.path.join(folder_path, fname)
    return None


def load_checkpoint_path(cfg, explicit_model_path: str | None) -> str:
    if explicit_model_path:
        checkpoint_path = explicit_model_path
    else:
        model_folder = os.path.join(cfg.OUTPUT_PATH, "model:1")
        checkpoint_path = find_checkpoint_in_folder(model_folder)
        if checkpoint_path is None:
            candidates = [
                os.path.join(cfg.OUTPUT_PATH, "_model_run:1_idx.pth"),
                os.path.join(cfg.OUTPUT_PATH, "model_with_temperature.pth"),
                os.path.join(cfg.OUTPUT_PATH, "best_model.pth"),
            ]
            for candidate in candidates:
                if os.path.isfile(candidate):
                    checkpoint_path = candidate
                    break

    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "No checkpoint found. Provide --model_path or ensure a .pth/.pt exists in the output folder."
        )
    return checkpoint_path


def resize_backto_original(cfg, pred_map, target_map, orig_size):
    orig_size_np = orig_size.detach().cpu().numpy() if torch.is_tensor(orig_size) else np.asarray(orig_size)

    if torch.is_tensor(pred_map):
        pred_np = pred_map.detach().cpu().numpy()
    else:
        pred_np = np.asarray(pred_map)

    augmenter = Augmentation(cfg)
    pred_np = augmenter.reverse_downsample_and_pad(orig_size_np, pred_np)

    if torch.is_tensor(target_map):
        target_np = target_map.detach().cpu().numpy()
    else:
        target_np = np.asarray(target_map)
    target_np = augmenter.reverse_downsample_and_pad(orig_size_np, target_np)

    return pred_np, target_np


def _metric_values(metric_output: Iterable[list]) -> list[float]:
    values = []
    for _, value in metric_output:
        values.append(float(value))
    return values


@torch.no_grad()
def collect_split_metrics(cfg, model, loader, device, logger, split_name: str):
    eh = evaluation_helper()
    lm = landmark_metrics()
    pixel_size_cpu = torch.tensor(cfg.DATASET.PIXEL_SIZE).to("cpu")

    rows = []
    model.eval()

    for batch_idx, (data, target, landmarks, meta, ids, orig_size, orig_img) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)
        meta = meta.to(device)

        pred = model(data, meta)
        batch_size = data.shape[0]

        for sample_idx in range(batch_size):
            sample_id = ids[sample_idx]
            pred_resized, target_resized = resize_backto_original(
                cfg,
                pred[sample_idx],
                target[sample_idx],
                orig_size[sample_idx],
            )

            target_points, predicted_points = eh.get_landmarks(
                pred_resized,
                target_resized,
                pixels_sizes=pixel_size_cpu,
            )
            target_points = target_points.squeeze(0)
            predicted_points = predicted_points.squeeze(0)

            ere_values = _metric_values(
                lm.get_eres(predicted_points, torch.as_tensor(pred_resized), pixelsize=pixel_size_cpu)
            )
            radial_values = _metric_values(
                lm.get_radial_errors(
                    predicted_points,
                    torch.as_tensor(pred_resized),
                    target_points,
                    torch.as_tensor(target_resized),
                    pixel_size_cpu,
                )
            )

            rows.append(
                {
                    "split": split_name,
                    "id": sample_id,
                    "ere_mean": float(np.mean(ere_values)),
                    "ere_std": float(np.std(ere_values)),
                    "mre_mean": float(np.mean(radial_values)),
                    "mre_std": float(np.std(radial_values)),
                }
            )

        logger.info("Processed %s batch %d/%d", split_name, batch_idx + 1, len(loader))

    if not rows:
        raise ValueError(f"No samples were processed for split '{split_name}'.")

    return rows


def save_metrics_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "id", "ere_mean", "ere_std", "mre_mean", "mre_std"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_ere_vs_mre(rows, out_path: Path, title: str):
    ere = np.asarray([row["ere_mean"] for row in rows], dtype=float)
    mre = np.asarray([row["mre_mean"] for row in rows], dtype=float)

    mask = np.isfinite(ere) & np.isfinite(mre)
    ere = ere[mask]
    mre = mre[mask]

    if ere.size == 0:
        raise ValueError("No finite ERE/MRE values available for plotting.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ere, mre, s=28, alpha=0.8)

    if ere.size >= 2:
        slope, intercept = np.polyfit(ere, mre, deg=1)
        xs = np.linspace(ere.min(), ere.max(), 200)
        ax.plot(xs, slope * xs + intercept, linewidth=2, label=f"Fit: y={slope:.3f}x+{intercept:.3f}")
        corr = np.corrcoef(ere, mre)[0, 1]
        ax.text(
            0.03,
            0.97,
            f"N={ere.size}\nr={corr:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    ax.set_title(title)
    ax.set_xlabel("ERE")
    ax.set_ylabel("Mean Radial Error (MRE)")
    ax.grid(True, linestyle="--", alpha=0.3)
    if ere.size >= 2:
        ax.legend(frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_loader(cfg, split_name: str, batch_size: int):
    dataset = dataloader(cfg, split_name)
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def main():
    args = parse_args()

    help_obj = helper(args.cfg, "ere_truerad")
    logger = help_obj.setup_logger()
    cfg = help_obj._get_cfg()

    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")
    logger.info("-----------Configuration-----------")
    logger.info(cfg)
    logger.info("")

    batch_size = args.batch_size or cfg.TRAIN.BATCH_SIZE

    val_dataset, val_loader = build_loader(cfg, "validation", batch_size)
    test_dataset, test_loader = build_loader(cfg, "testing", batch_size)
    help_obj._dataset_shape(val_dataset)
    help_obj._dataset_shape(test_dataset)

    trainer = training(cfg, logger, cfg.TRAIN.L2_REG)
    model = trainer._get_network()
    device = trainer.device
    model.to(device)
    model.eval()

    checkpoint_path = load_checkpoint_path(cfg, args.model_path)
    logger.info("Loading model from: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict"))
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")

    save_dir = Path(cfg.OUTPUT_PATH) / "ere_truerad"
    save_dir.mkdir(parents=True, exist_ok=True)

    val_rows = collect_split_metrics(cfg, model, val_loader, device, logger, "validation")
    test_rows = collect_split_metrics(cfg, model, test_loader, device, logger, "testing")

    save_metrics_csv(val_rows, save_dir / "validation_ere_mre.csv")
    save_metrics_csv(test_rows, save_dir / "testing_ere_mre.csv")
    save_metrics_csv(val_rows + test_rows, save_dir / "all_ere_mre.csv")

    plot_ere_vs_mre(val_rows, save_dir / "validation_ere_vs_mre.png", "Validation: ERE vs MRE")
    plot_ere_vs_mre(test_rows, save_dir / "testing_ere_vs_mre.png", "Testing: ERE vs MRE")

    logger.info(
        "Validation summary: mean ERE=%.4f, mean MRE=%.4f",
        np.mean([row["ere_mean"] for row in val_rows]),
        np.mean([row["mre_mean"] for row in val_rows]),
    )
    logger.info(
        "Testing summary: mean ERE=%.4f, mean MRE=%.4f",
        np.mean([row["ere_mean"] for row in test_rows]),
        np.mean([row["mre_mean"] for row in test_rows]),
    )
    logger.info("Saved outputs to: %s", save_dir)


if __name__ == "__main__":
    main()
