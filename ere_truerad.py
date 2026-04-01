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
from tqdm.auto import tqdm


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
    parser.add_argument(
        "--bin_size",
        required=False,
        type=int,
        default=36,
        help="Number of landmarks per ERE bin when building the binned correlation plot.",
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


def _point_scale_tensor(pixel_size, device):
    scale = torch.as_tensor(pixel_size, dtype=torch.float32, device=device)
    if scale.ndim == 0:
        scale = scale.repeat(2)
    if scale.numel() != 2:
        raise ValueError("cfg.DATASET.PIXEL_SIZE must be a scalar or length-2 iterable.")
    return scale.view(1, 2)


def _compute_ere_values_mm(eh, pred_map, predicted_points, point_scale):
    if not torch.is_tensor(pred_map):
        pred_map = torch.as_tensor(pred_map, dtype=torch.float32)
    if not torch.is_tensor(predicted_points):
        predicted_points = torch.as_tensor(predicted_points, dtype=torch.float32)

    if pred_map.ndim == 3:
        pred_map = pred_map.unsqueeze(0)
    if predicted_points.ndim == 2:
        predicted_points = predicted_points.unsqueeze(0)

    thresholded = eh.get_thresholded_heatmap(pred_map, predicted_points)
    point_scale = point_scale.to(pred_map.device).view(1, 2)

    ere_values_mm = []
    for pred_thresh, predicted_point in zip(thresholded[0], predicted_points[0]):
        indices = torch.nonzero(pred_thresh)
        if indices.numel() == 0:
            ere_values_mm.append(0.0)
            continue
        significant_values = pred_thresh[indices[:, 0], indices[:, 1]]
        scaled_indices = torch.flip(indices, dims=[1]).float() * point_scale
        scaled_predicted_point = predicted_point.float().view(1, 2) * point_scale
        distances_mm = torch.norm(scaled_indices - scaled_predicted_point, dim=1)
        ere_mm = torch.sum(significant_values * distances_mm)
        ere_values_mm.append(float(ere_mm.detach().cpu()))
    return ere_values_mm


@torch.no_grad()
def collect_split_metrics(cfg, model, loader, device, logger, split_name: str):
    eh = evaluation_helper()
    lm = landmark_metrics()
    pixel_size_cpu = torch.tensor(cfg.DATASET.PIXEL_SIZE).to("cpu")
    point_scale_cpu = _point_scale_tensor(cfg.DATASET.PIXEL_SIZE, torch.device("cpu"))

    image_rows = []
    landmark_rows = []
    model.eval()

    progress_bar = tqdm(loader, desc=f"{split_name} batches", total=len(loader))
    for batch_idx, (data, target, landmarks, meta, ids, orig_size, orig_img) in enumerate(progress_bar):
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
            ere_values_mm = _compute_ere_values_mm(eh, pred_resized, predicted_points, point_scale_cpu)
            radial_values = _metric_values(
                lm.get_radial_errors(
                    predicted_points,
                    torch.as_tensor(pred_resized),
                    target_points,
                    torch.as_tensor(target_resized),
                    pixel_size_cpu,
                )
            )
            displacement_mm = (predicted_points.float() - target_points.float()) * point_scale_cpu
            radial_values_mm = displacement_mm.norm(dim=1).detach().cpu().numpy().astype(float).tolist()

            image_rows.append(
                {
                    "split": split_name,
                    "id": sample_id,
                    "ere_mean": float(np.mean(ere_values)),
                    "ere_std": float(np.std(ere_values)),
                    "ere_mean_mm": float(np.mean(ere_values_mm)),
                    "ere_std_mm": float(np.std(ere_values_mm)),
                    "mre_mean_px": float(np.mean(radial_values)),
                    "mre_std_px": float(np.std(radial_values)),
                    "mre_mean_mm": float(np.mean(radial_values_mm)),
                    "mre_std_mm": float(np.std(radial_values_mm)),
                }
            )

            for landmark_idx, (ere_value, ere_value_mm, radial_value_px, radial_value_mm) in enumerate(
                zip(ere_values, ere_values_mm, radial_values, radial_values_mm), start=1
            ):
                landmark_rows.append(
                    {
                        "split": split_name,
                        "id": sample_id,
                        "landmark_index": landmark_idx,
                        "ere": float(ere_value),
                        "ere_mm": float(ere_value_mm),
                        "true_radial_error_px": float(radial_value_px),
                        "true_radial_error_mm": float(radial_value_mm),
                    }
                )
        progress_bar.set_postfix(batch=f"{batch_idx + 1}/{len(loader)}")

    if not image_rows:
        raise ValueError(f"No samples were processed for split '{split_name}'.")

    return image_rows, landmark_rows


def save_metrics_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "id",
        "ere_mean",
        "ere_std",
        "ere_mean_mm",
        "ere_std_mm",
        "mre_mean_px",
        "mre_std_px",
        "mre_mean_mm",
        "mre_std_mm",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_landmark_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "id", "landmark_index", "ere", "ere_mm", "true_radial_error_px", "true_radial_error_mm"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_binned_landmark_rows(rows, bin_size: int, radial_error_key: str, ere_key: str = "ere"):
    if bin_size <= 0:
        raise ValueError("bin_size must be a positive integer.")

    finite_rows = [
        row for row in rows if np.isfinite(row[ere_key]) and np.isfinite(row[radial_error_key])
    ]
    if not finite_rows:
        raise ValueError("No finite landmark-level ERE/true radial error values available for binning.")

    sorted_rows = sorted(finite_rows, key=lambda row: row[ere_key])
    binned_rows = []
    for bin_idx, start in enumerate(range(0, len(sorted_rows), bin_size), start=1):
        chunk = sorted_rows[start : start + bin_size]
        ere_values = np.asarray([row[ere_key] for row in chunk], dtype=float)
        radial_values = np.asarray([row[radial_error_key] for row in chunk], dtype=float)
        binned_rows.append(
            {
                "bin_index": bin_idx,
                "bin_start_rank": start + 1,
                "bin_end_rank": start + len(chunk),
                "num_landmarks": len(chunk),
                "ere_key": ere_key,
                "radial_error_key": radial_error_key,
                "ere_mean": float(np.mean(ere_values)),
                "ere_std": float(np.std(ere_values)),
                "true_radial_error_mean": float(np.mean(radial_values)),
                "true_radial_error_std": float(np.std(radial_values)),
            }
        )
    return binned_rows


def save_binned_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bin_index",
        "bin_start_rank",
        "bin_end_rank",
        "num_landmarks",
        "ere_key",
        "radial_error_key",
        "ere_mean",
        "ere_std",
        "true_radial_error_mean",
        "true_radial_error_std",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_correlation_stats(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        raise ValueError("No finite values available for correlation.")

    stats = {"n": int(x.size), "r": float("nan")}
    if x.size >= 2:
        stats["r"] = float(np.corrcoef(x, y)[0, 1])
    return stats


def save_correlation_summary(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "comparison", "n", "r"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_bin_sizes(default_bin_size: int):
    smaller = max(1, default_bin_size // 2)
    candidate_sizes = [1, 2, 9, smaller, default_bin_size, default_bin_size * 2]
    return sorted({size for size in candidate_sizes if size > 0})


def _scatter_with_fit(ax, x_values, y_values, title: str, xlabel: str, ylabel: str):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        raise ValueError("No finite values available for plotting.")

    ax.scatter(x, y, s=20, alpha=0.5)

    if x.size >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, slope * xs + intercept, linewidth=2, label=f"Fit: y={slope:.3f}x+{intercept:.3f}")
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(
            0.03,
            0.97,
            f"N={x.size}\nr={corr:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    if x.size >= 2:
        ax.legend(frameon=False)


def plot_landmark_ere_vs_mre(rows, out_path: Path, title: str):
    ere = [row["ere"] for row in rows]
    mre = [row["true_radial_error_px"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter_with_fit(ax, ere, mre, title, "ERE", "MRE (pixels)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ere_vs_mre(rows, out_path: Path, title: str):
    ere = [row["ere_mean"] for row in rows]
    mre = [row["mre_mean_px"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter_with_fit(ax, ere, mre, title, "ERE", "Mean Radial Error (MRE, pixels)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_binned_ere_vs_true_radial_error(rows, out_path: Path, title: str):
    ere = [row["ere_mean"] for row in rows]
    true_radial_error = [row["true_radial_error_mean"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter_with_fit(
        ax,
        ere,
        true_radial_error,
        title,
        "Average ERE per bin",
        "Average True Radial Error per bin (pixels)",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paired_landmark_ere_vs_mre(rows, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter_with_fit(
        axes[0],
        [row["ere"] for row in rows],
        [row["true_radial_error_px"] for row in rows],
        f"{title} (pixels)",
        "ERE",
        "MRE (pixels)",
    )
    _scatter_with_fit(
        axes[1],
        [row["ere_mm"] for row in rows],
        [row["true_radial_error_mm"] for row in rows],
        f"{title} (mm)",
        "ERE (mm)",
        "MRE (mm)",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paired_image_ere_vs_mre(rows, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter_with_fit(
        axes[0],
        [row["ere_mean"] for row in rows],
        [row["mre_mean_px"] for row in rows],
        f"{title} (pixels)",
        "ERE",
        "Mean Radial Error (MRE, pixels)",
    )
    _scatter_with_fit(
        axes[1],
        [row["ere_mean_mm"] for row in rows],
        [row["mre_mean_mm"] for row in rows],
        f"{title} (mm)",
        "ERE (mm)",
        "Mean Radial Error (MRE, mm)",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_multibin_paired_ere(rows, out_path: Path, title: str, bin_sizes: list[int]):
    fig, axes = plt.subplots(len(bin_sizes), 2, figsize=(14, 5 * len(bin_sizes)))
    axes = np.atleast_2d(axes)

    for row_idx, bin_size in enumerate(bin_sizes):
        px_rows = build_binned_landmark_rows(rows, bin_size, "true_radial_error_px")
        mm_rows = build_binned_landmark_rows(rows, bin_size, "true_radial_error_mm", ere_key="ere_mm")
        _scatter_with_fit(
            axes[row_idx, 0],
            [row["ere_mean"] for row in px_rows],
            [row["true_radial_error_mean"] for row in px_rows],
            f"{title} (pixels, bin={bin_size})",
            "Average ERE per bin",
            "Average True Radial Error per bin (pixels)",
        )
        _scatter_with_fit(
            axes[row_idx, 1],
            [row["ere_mean"] for row in mm_rows],
            [row["true_radial_error_mean"] for row in mm_rows],
            f"{title} (mm, bin={bin_size})",
            "Average ERE per bin (mm)",
            "Average True Radial Error per bin (mm)",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_perlandmark_paired_ere(rows, out_path: Path, title: str):
    landmark_indices = sorted({int(row["landmark_index"]) for row in rows})
    if not landmark_indices:
        raise ValueError("No landmark rows available for per-landmark plotting.")

    fig, axes = plt.subplots(len(landmark_indices), 2, figsize=(14, 4 * len(landmark_indices)))
    axes = np.atleast_2d(axes)

    for row_idx, landmark_index in enumerate(landmark_indices):
        landmark_rows = [row for row in rows if int(row["landmark_index"]) == landmark_index]
        _scatter_with_fit(
            axes[row_idx, 0],
            [row["ere"] for row in landmark_rows],
            [row["true_radial_error_px"] for row in landmark_rows],
            f"{title} Landmark {landmark_index} (pixels, bin=1)",
            "ERE",
            "True Radial Error (pixels)",
        )
        _scatter_with_fit(
            axes[row_idx, 1],
            [row["ere_mm"] for row in landmark_rows],
            [row["true_radial_error_mm"] for row in landmark_rows],
            f"{title} Landmark {landmark_index} (mm, bin=1)",
            "ERE (mm)",
            "True Radial Error (mm)",
        )

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

    val_image_rows, val_landmark_rows = collect_split_metrics(cfg, model, val_loader, device, logger, "validation")
    test_image_rows, test_landmark_rows = collect_split_metrics(cfg, model, test_loader, device, logger, "testing")

    save_metrics_csv(val_image_rows, save_dir / "validation_ere_mre.csv")
    save_metrics_csv(test_image_rows, save_dir / "testing_ere_mre.csv")
    save_metrics_csv(val_image_rows + test_image_rows, save_dir / "all_ere_mre.csv")

    save_landmark_csv(val_landmark_rows, save_dir / "validation_landmark_ere_true_radial_error.csv")
    save_landmark_csv(test_landmark_rows, save_dir / "testing_landmark_ere_true_radial_error.csv")
    save_landmark_csv(
        val_landmark_rows + test_landmark_rows,
        save_dir / "all_landmark_ere_true_radial_error.csv",
    )

    bin_sizes = get_bin_sizes(args.bin_size)
    val_binned_rows = build_binned_landmark_rows(val_landmark_rows, args.bin_size, "true_radial_error_px")
    test_binned_rows = build_binned_landmark_rows(test_landmark_rows, args.bin_size, "true_radial_error_px")

    save_binned_csv(val_binned_rows, save_dir / "validation_binned_ere_true_radial_error.csv")
    save_binned_csv(test_binned_rows, save_dir / "testing_binned_ere_true_radial_error.csv")

    val_landmark_corr = compute_correlation_stats(
        [row["ere"] for row in val_landmark_rows],
        [row["true_radial_error_px"] for row in val_landmark_rows],
    )
    test_landmark_corr = compute_correlation_stats(
        [row["ere"] for row in test_landmark_rows],
        [row["true_radial_error_px"] for row in test_landmark_rows],
    )
    val_binned_corr = compute_correlation_stats(
        [row["ere_mean"] for row in val_binned_rows],
        [row["true_radial_error_mean"] for row in val_binned_rows],
    )
    test_binned_corr = compute_correlation_stats(
        [row["ere_mean"] for row in test_binned_rows],
        [row["true_radial_error_mean"] for row in test_binned_rows],
    )
    save_correlation_summary(
        [
            {
                "split": "validation",
                "comparison": "landmark_ere_vs_mre",
                "n": val_landmark_corr["n"],
                "r": val_landmark_corr["r"],
            },
            {
                "split": "validation",
                "comparison": "binned_ere_vs_true_radial_error",
                "n": val_binned_corr["n"],
                "r": val_binned_corr["r"],
            },
            {
                "split": "testing",
                "comparison": "landmark_ere_vs_mre",
                "n": test_landmark_corr["n"],
                "r": test_landmark_corr["r"],
            },
            {
                "split": "testing",
                "comparison": "binned_ere_vs_true_radial_error",
                "n": test_binned_corr["n"],
                "r": test_binned_corr["r"],
            },
        ],
        save_dir / "correlation_summary.csv",
    )

    plot_landmark_ere_vs_mre(
        val_landmark_rows,
        save_dir / "validation_landmark_ere_vs_mre.png",
        "Validation: Landmark-level ERE vs MRE",
    )
    plot_landmark_ere_vs_mre(
        test_landmark_rows,
        save_dir / "testing_landmark_ere_vs_mre.png",
        "Testing: Landmark-level ERE vs MRE",
    )
    plot_ere_vs_mre(val_image_rows, save_dir / "validation_ere_vs_mre.png", "Validation: ERE vs MRE")
    plot_ere_vs_mre(test_image_rows, save_dir / "testing_ere_vs_mre.png", "Testing: ERE vs MRE")
    plot_paired_landmark_ere_vs_mre(
        val_landmark_rows,
        save_dir / "validation_landmark_ere_vs_mre_px_mm.png",
        "Validation: Landmark-level ERE vs MRE",
    )
    plot_paired_landmark_ere_vs_mre(
        test_landmark_rows,
        save_dir / "testing_landmark_ere_vs_mre_px_mm.png",
        "Testing: Landmark-level ERE vs MRE",
    )
    plot_paired_image_ere_vs_mre(
        val_image_rows,
        save_dir / "validation_image_ere_vs_mre_px_mm.png",
        "Validation: Image-level ERE vs MRE",
    )
    plot_paired_image_ere_vs_mre(
        test_image_rows,
        save_dir / "testing_image_ere_vs_mre_px_mm.png",
        "Testing: Image-level ERE vs MRE",
    )
    plot_binned_ere_vs_true_radial_error(
        val_binned_rows,
        save_dir / "validation_binned_ere_vs_true_radial_error.png",
        f"Validation: Binned ERE vs True Radial Error (bin={args.bin_size})",
    )
    plot_binned_ere_vs_true_radial_error(
        test_binned_rows,
        save_dir / "testing_binned_ere_vs_true_radial_error.png",
        f"Testing: Binned ERE vs True Radial Error (bin={args.bin_size})",
    )
    plot_multibin_paired_ere(
        val_landmark_rows,
        save_dir / "validation_binned_ere_vs_true_radial_error_px_mm_multibin.png",
        "Validation: Binned ERE vs True Radial Error",
        bin_sizes,
    )
    plot_multibin_paired_ere(
        test_landmark_rows,
        save_dir / "testing_binned_ere_vs_true_radial_error_px_mm_multibin.png",
        "Testing: Binned ERE vs True Radial Error",
        bin_sizes,
    )
    plot_perlandmark_paired_ere(
        val_landmark_rows,
        save_dir / "validation_binned_ere_vs_true_radial_error_px_mm_perlandmark.png",
        "Validation: Per-landmark ERE vs True Radial Error",
    )
    plot_perlandmark_paired_ere(
        test_landmark_rows,
        save_dir / "testing_binned_ere_vs_true_radial_error_px_mm_perlandmark.png",
        "Testing: Per-landmark ERE vs True Radial Error",
    )

    logger.info(
        "Validation summary: mean ERE=%.4f, mean MRE=%.4f",
        np.mean([row["ere_mean"] for row in val_image_rows]),
        np.mean([row["mre_mean_px"] for row in val_image_rows]),
    )
    logger.info(
        "Testing summary: mean ERE=%.4f, mean MRE=%.4f",
        np.mean([row["ere_mean"] for row in test_image_rows]),
        np.mean([row["mre_mean_px"] for row in test_image_rows]),
    )
    logger.info(
        "Validation landmark-level Pearson r (ERE vs MRE): %.4f over %d landmarks.",
        val_landmark_corr["r"],
        val_landmark_corr["n"],
    )
    logger.info(
        "Validation binned correlation built from %d landmarks into %d bins of up to %d landmarks.",
        len(val_landmark_rows),
        len(val_binned_rows),
        args.bin_size,
    )
    logger.info(
        "Validation binned Pearson r (ERE vs True Radial Error): %.4f over %d bins.",
        val_binned_corr["r"],
        val_binned_corr["n"],
    )
    logger.info("Validation multibin figure saved with bin sizes: %s", bin_sizes)
    logger.info("Validation per-landmark figure saved with one row per landmark (equivalent to bin=1).")
    logger.info(
        "Testing landmark-level Pearson r (ERE vs MRE): %.4f over %d landmarks.",
        test_landmark_corr["r"],
        test_landmark_corr["n"],
    )
    logger.info(
        "Testing binned correlation built from %d landmarks into %d bins of up to %d landmarks.",
        len(test_landmark_rows),
        len(test_binned_rows),
        args.bin_size,
    )
    logger.info(
        "Testing binned Pearson r (ERE vs True Radial Error): %.4f over %d bins.",
        test_binned_corr["r"],
        test_binned_corr["n"],
    )
    logger.info("Testing multibin figure saved with bin sizes: %s", bin_sizes)
    logger.info("Testing per-landmark figure saved with one row per landmark (equivalent to bin=1).")
    logger.info("Saved outputs to: %s", save_dir)


if __name__ == "__main__":
    main()
