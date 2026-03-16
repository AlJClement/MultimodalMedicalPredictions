# temperature_scaling.py
import os
import math
import types
from typing import Optional, Callable, Dict, Any, Tuple, Sequence
from .evaluation_helper import evaluation_helper
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import numpy as np


class TemperatureScaling(nn.Module):
    """
    Learn a single positive scalar temperature to scale logits before a spatial softmax.
    - Use `fit()` to optimize temperature minimizing spatial NLL on fine_tune_loader.
    - Use `fit_with_evaluation()` to also compute ECE/reliability diagrams using an evaluation helper.
    """

    def __init__(self, cfg, model: nn.Module, device: Optional[torch.device] = None, init_temp: float = 1.0):
        super().__init__()
        # Device selection: prefer provided device -> model param device -> cpu.
        self.cfg = cfg
        if device is None:
            try:
                param = next(model.parameters())
                inferred_device = param.device
            except StopIteration:
                inferred_device = torch.device("cpu")
            device = inferred_device
        self.device = device
        self.model = model.to(self.device)

        # If model already exposes a temperatures Tensor, wrap into Parameter if needed.
        self._owns_param = False
        if hasattr(self.model, "temperatures") and isinstance(getattr(self.model, "temperatures"), torch.Tensor):
            existing_t = getattr(self.model, "temperatures")
            if not isinstance(existing_t, nn.Parameter):
                setattr(self.model, "temperatures", nn.Parameter(existing_t.to(self.device)))
            self.log_temp = None
            self._owns_param = False
        else:
            # Keep internal unconstrained parameter in log-space; map to positive with softplus
            init_val = math.log(math.exp(init_temp) - 1.0) if init_temp > 1e-3 else math.log(1e-3)
            self.log_temp = nn.Parameter(torch.tensor(float(init_val), device=self.device))
            self._owns_param = True

        # If model lacks a scale() method, we'll attach one after fitting (so code can call model.scale()).
        self._patched_scale = False

        # History stored after fit or fit_with_evaluation
        self.history: Dict[str, list] = {"train_nll": [], "val_nll": [], "temperature": [], "ece": []}

    # ----------------------
    # Low-level helpers
    # ----------------------
    def _get_temperature(self) -> torch.Tensor:
        """Return a positive tensor temperature (>0)."""
        if hasattr(self.model, "temperatures") and isinstance(self.model.temperatures, torch.Tensor):
            t = self.model.temperatures  # shape (C,) or scalar
            t_pos = F.softplus(t) + 1e-6  # shape (C,) -> broadcasts
            return t_pos.view(1, -1, 1, 1)  # broadcastable into (B,C,H,W)#
        else:
            if self.log_temp is None:
                raise RuntimeError("No temperature parameter found.")
            return F.softplus(self.log_temp) + 1e-6

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return logits divided by temperature (broadcastable)."""
        T = self._get_temperature()
        return logits / T

    @staticmethod
    def two_d_softmax(x: torch.Tensor) -> torch.Tensor:
        """
        Spatial softmax over H*W dims for logits shaped (B, C, H, W).
        Returns probabilities in same shape.
        """
        if x.dim() < 3:
            raise ValueError("Expected logits with spatial dims, e.g., (B, C, H, W).")
        B, C = x.shape[0], x.shape[1]
        flat = x.view(B, C, -1)
        probs = torch.softmax(flat, dim=2)
        return probs.view_as(x)

    @staticmethod
    def heatmap_nll(pred_logits: torch.Tensor, target_heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Compute NLL between spatial distributions (pred_logits after scaling and softmax should be passed
        or provide logits and we will compute log_softmax inside).
        - pred_logits expected shape: (B, C, H, W) (logits or already softmaxed - we assume logits and use log_softmax)
        - target_heatmaps: (B, C, H, W) non-negative target maps
        """
        if pred_logits.dim() < 3:
            raise ValueError("pred_logits must have >=3 dims (B,C,...)")

        B = pred_logits.shape[0]
        C = pred_logits.shape[1]
        pred_flat = pred_logits.view(B, C, -1)
        tgt_flat = target_heatmaps.view(B, C, -1)

        tgt_sums = tgt_flat.sum(dim=2, keepdim=True)
        zero_mask = (tgt_sums <= 0)
        normalized = tgt_flat / (tgt_sums + 1e-12)
        # if the channel sum is zero, produce an all-zero target map for that channel
        tgt_norm = torch.where(zero_mask, torch.zeros_like(tgt_flat), normalized)

        log_probs = F.log_softmax(pred_flat, dim=2)
        nll = - (tgt_norm * log_probs).sum(dim=2)  # shape (B, C)
        loss = nll.mean()
        return loss
    
    def reliability_diagram(self, all_radial_errors, all_mode_probabilities, save_path, tol,
                            n_of_bins=10, x_max=0.2, do_not_save=False):
        """
        - Accuracy: blue
        - Confidence bars: lime (alpha=0.5)
        Returns:
        (ece, avg_conf_for_each_bin, avg_acc_for_each_bin, count_for_each_bin)
        """
        # ensure inputs are numpy arrays
        all_mode_probabilities = np.asarray(all_mode_probabilities).ravel()
        all_radial_errors = np.asarray(all_radial_errors).ravel()
        x_min, x_max = min(all_mode_probabilities), max(all_mode_probabilities)

        # bins fixed on [0,1] (confidence lies in [0,1]).
        bins = np.linspace(0.0, x_max, n_of_bins + 1)
        widths = bins[1:] - bins[:-1]

        # determine correctness per sample using area-based tolerance
        radius = math.sqrt((tol**2) / math.pi)
        correct_predictions = (all_radial_errors < radius)

        # counts and sum of confidences per bin (safe even if some bins empty)
        count_for_each_bin, _ = np.histogram(all_mode_probabilities, bins=bins)
        total_confidence_for_each_bin, _, _ = stats.binned_statistic(
            all_mode_probabilities, all_mode_probabilities, 'sum', bins=bins)

        # compute number correct per bin using digitize (1..len(bins))
        bin_indices = np.digitize(all_mode_probabilities, bins)
        no_of_correct_preds = np.zeros(len(bins) - 1, dtype=float)
        for idx, correct in zip(bin_indices, correct_predictions):
            if 1 <= idx <= len(bins) - 1:
                no_of_correct_preds[idx - 1] += float(correct)

        # avoid division by zero: only divide where count > 0
        avg_conf_for_each_bin = np.zeros(len(bins) - 1, dtype=float)
        avg_acc_for_each_bin = np.zeros(len(bins) - 1, dtype=float)
        mask = count_for_each_bin > 0
        if np.any(mask):
            avg_conf_for_each_bin[mask] = (total_confidence_for_each_bin[mask] /
                                           count_for_each_bin[mask].astype(float))
            avg_acc_for_each_bin[mask] = (no_of_correct_preds[mask] /
                                          count_for_each_bin[mask].astype(float))

        # compute ECE in percentage; if no samples, set nan
        n = float(np.sum(count_for_each_bin))
        if n <= 0.0:
            ece = float("nan")
        else:
            ece = float(np.sum((count_for_each_bin / n) * np.abs(avg_acc_for_each_bin - avg_conf_for_each_bin)) * 100.0)

        # -------- plotting (preserve your original style/colors) --------
        plt.rcParams["figure.figsize"] = (6, 6)
        fig, ax = plt.subplots(1, 1)
        ax.grid(zorder=0)

        plt.subplots_adjust(left=0.15)
        plt.xlabel('Confidence', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(zorder=0)
        # plt.xlim(0.0, plot_xmax)
        # plt.ylim(0.0, 1.0)  # accuracy/confidence are in [0,1]
        plt.xlim(x_min,x_max)
        y_max = max(avg_acc_for_each_bin)
        plt.ylim(0,y_max)
        # Accuracy bars (blue)
        ax.bar(bins[:-1], avg_acc_for_each_bin, align='edge', width=widths,
               color='blue', edgecolor='black', label='Accuracy', zorder=3)

        # Confidence bars (lime, semi-transparent)
        ax.bar(bins[:-1], avg_conf_for_each_bin, align='edge', width=widths,
               color='lime', edgecolor='black', alpha=0.5, label='Gap', zorder=3)

        ax.legend(fontsize=12, loc="upper left", prop={'size': 12})
        ax.text(0.71, 0.075, f'ECE={ece:.2f}', backgroundcolor='white',
                fontsize='large', transform=ax.transAxes)

        if not do_not_save:
            plt.savefig(save_path)
        plt.close()

        return ece, avg_conf_for_each_bin, avg_acc_for_each_bin, count_for_each_bin
    
    def fit(self,
            fine_tune_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            n_epochs: int = 50,
            lr: float = 1e-2,
            weight_decay: float = 0.0,
            device: Optional[torch.device] = None,
            verbose: bool = True) -> float:
        """
        Basic temperature fitting: minimize spatial NLL on fine_tune_loader.
        Leaves model weights frozen; optimizes only the temperature parameter.
        Returns final positive temperature scalar (float).
        """
        device = device or self.device
        self.model = self.model.to(device)

        # choose which parameter to optimize
        if self._owns_param:
            opt_params = [self.log_temp]
        else:
            t = getattr(self.model, "temperatures")
            if not isinstance(t, nn.Parameter):
                setattr(self.model, "temperatures", nn.Parameter(t.to(device)))
            opt_params = [getattr(self.model, "temperatures")]

        optimizer = torch.optim.Adam(opt_params, lr=lr, weight_decay=weight_decay)

        for epoch in range(n_epochs):
            self.model.eval()
            total_loss = 0.0
            count = 0
            for batch in fine_tune_loader:
                # robustly pull inputs and targets (assume second element is channels/targets)
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs = batch[0].to(device)
                    targets = batch[1].to(device)
                else:
                    raise RuntimeError("fine_tune_loader batch format not recognized; expected (input, target, ...).")

                with torch.no_grad():
                    # flexible forward signature handling
                    try:
                        out = self.model(inputs)
                    except TypeError:
                        # maybe model expects (inputs, meta)
                        if len(batch) >= 3:
                            meta = batch[2]
                            out = self.model(inputs, meta)
                        else:
                            raise

                scaled = self.scale_logits(out)
                loss = self.heatmap_nll(scaled, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                count += 1

            avg_loss = total_loss / max(1, count)
            if verbose:
                print(f"[TempScale] Epoch {epoch+1}/{n_epochs}  fine-tune NLL: {avg_loss:.6f}")

        # attach final temperature to model as parameter (so it can be saved)
        # robust final_temp extraction: return scalar float if single-element, else list
        t_raw = self._get_temperature().detach().cpu()
        t_vals = t_raw.view(-1)
        if t_vals.numel() == 1:
            final_temp = float(t_vals.item())
        else:
            final_temp = t_vals.tolist()

        if not hasattr(self.model, "temperatures") or not isinstance(getattr(self.model, "temperatures"), torch.Tensor):
            setattr(self.model, "temperatures", nn.Parameter(torch.tensor(final_temp if isinstance(final_temp, float) else np.array(final_temp), device=device)))

        # ensure model.scale exists
        if not hasattr(self.model, "scale") or not callable(getattr(self.model, "scale")):
            def _scale(self_model, out):
                T = F.softplus(self_model.temperatures) + 1e-6
                return out / T
            self.model.scale = types.MethodType(_scale, self.model)
            self._patched_scale = True

        return final_temp
    import numpy as np

    def normalize_channels_95(self,out_np):
        """
        out_np: numpy array with shape (1, C, H, W)
        returns normalized array same shape
        """
        
        out = out_np.copy().astype(np.float32)
        B, C, H, W = out.shape
        
        for c in range(C):
            channel = out[0, c]

            # 95% confidence interval
            lo, hi = np.percentile(channel, [2.5, 97.5])
            # mask inside range
            mask = (channel >= lo) & (channel <= hi)

            if np.any(mask):
                vmin = channel[mask].min()
                vmax = channel[mask].max()
            else:
                vmin, vmax = 0.0, 0.0

            # remove outside values
            channel[~mask] = 0.0

            # normalize
            if vmax > vmin:
                channel = (channel - vmin) / (vmax - vmin)
            else:
                channel = np.zeros_like(channel)

            out[0, c] = np.clip(channel, 0, 1)

        return out

    def normalize_to_probability(self, out_np):
        """
        Convert out_np shaped (B, C, H, W) logits/probs to valid probability maps:
        For each (b,c): if sum>0 -> map /= sum, else leave zeros.
        Returns array same shape, dtype float32.
        """
        out = out_np.astype(np.float32).copy()
        B, C, H, W = out.shape
        for b in range(B):
            for c in range(C):
                arr = out[b, c]
                s = float(arr.sum())
                if s > 0.0:
                    out[b, c] = arr / s
                else:
                    out[b, c] = np.zeros_like(arr)
        return out
    
    def get_center_window_probability(self,
            heatmap: torch.Tensor,
            radius: int = 2, ## how many pixels outwards to check - make same as plot
            remove_percent: float = 10.0, ### removes all zeros in heatmap and rescales 
        ) -> torch.Tensor:
            """
            heatmap: (B, C, H, W)
            radius : int, pixel radius around the center to include (radius=1 -> 3x3)
            remove_percent: float in [0,100), percent of lowest values to zero-out per channel

            Returns:
                Tensor of shape (B, C) with the sum of probabilities inside the center window
            """

            if heatmap.dim() != 4:
                raise ValueError("heatmap must be (B, C, H, W)")

            hm = heatmap.clone()

            B, C, H, W = hm.shape
            device = hm.device
            dtype = hm.dtype

            # compute per-channel threshold across spatial dims
            # shape (B, C, 1, 1) to broadcast
            flat = hm.view(B, C, -1)
            q = remove_percent / 100.0
            if q <= 0:
                threshold = torch.full((B, C, 1, 1), float("-inf"), device=device, dtype=dtype)
            elif q >= 1:
                threshold = torch.full((B, C, 1, 1), float("inf"), device=device, dtype=dtype)
            else:
                # torch.quantile supports dim for recent PyTorch; use dim=2 on flattened
                threshold_vals = torch.quantile(flat, q, dim=2)          # shape (B, C)
                threshold = threshold_vals.view(B, C, 1, 1)              # broadcastable

            # zero-out below threshold
            hm = torch.where(hm < threshold, torch.zeros_like(hm), hm)

            # renormalize per (B,C)
            total = hm.sum(dim=(2, 3), keepdim=True)   # shape (B, C, 1, 1)
            nonzero = total > 0
            hm = torch.where(nonzero, hm / total, hm)  # keep zeros when total==0

            # center window coordinates (clamped to image bounds)
            cy = H // 2
            cx = W // 2
            y0 = max(0, cy - radius)
            y1 = min(H, cy + radius + 1)   # python slice end-exclusive
            x0 = max(0, cx - radius)
            x1 = min(W, cx + radius + 1)

            # sum over the center window
            center_sum = hm[:, :, y0:y1, x0:x1].sum(dim=(2, 3))  # shape (B, C)

            return center_sum

    def fit_with_evaluation(self,
                            train_loader: torch.utils.data.DataLoader,
                            val_loader: torch.utils.data.DataLoader,
                            save_dir: str,
                            n_epochs: int = 30,
                            lr: float = 1e-2,
                            weight_decay: float = 0.0,
                            tol_px: float = 1.0, ## num pixels
                            n_bins: int = 10,
                            device: Optional[torch.device] = None,
                            logger: Optional[Any] = None,
                            csv_name: str = "temperature_scaling_history.csv") -> Dict[str, Any]:
        """
        Fit temperature using train_loader; at each epoch evaluate on val_loader via evaluation_helper
        to compute radial errors & mode probabilities. 

        radial errors: are mres
        mode probabilities: max value of flattened heatmap
        
        Produces ECE + reliability diagrams per epoch,
        saves best model (by lowest ECE) and a CSV with history.

        """
        device = device or self.device
        os.makedirs(save_dir, exist_ok=True)
        self.model = self.model.to(device)
        
        # ensure temperature parameter present (vector per-channel)
        if not hasattr(self.model, "temperatures"):
            C = self.cfg.DATASET.NUM_LANDMARKS
            self.model.temperatures = nn.Parameter(torch.ones(C, device=device))
        # ensure it's a parameter and requires grad
        if not isinstance(self.model.temperatures, nn.Parameter):
            self.model.temperatures = nn.Parameter(self.model.temperatures.to(device))
        self.model.temperatures.requires_grad = True

        # If we previously had a scalar log_temp, stop using it — we want to optimize per-channel temps.
        self._owns_param = False
        self.log_temp = None

        # --- Recreate optimizer so it points to the correct parameter object ---
        optimizer = torch.optim.Adam([self.model.temperatures], lr=lr, weight_decay=weight_decay)

        # Debug check (optional, safe)
        print("[INIT] id(model.temperatures) =", id(self.model.temperatures), "shape:", self.model.temperatures.shape)
        for i, g in enumerate(optimizer.param_groups):
            for j, op in enumerate(g['params']):
                print(f"[INIT] id(opt param) group{i} param{j} =", id(op), "shape:", getattr(op,'shape',None))

        # prepare CSV
        csv_path = os.path.join(save_dir, csv_name)
        with open(csv_path, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["epoch", "train_nll", "val_nll", "temperature", "ece"])

        best_ece = float("inf")
        best_temp = None
        best_model_path = None

        # main loop
        for epoch in range(n_epochs):
            if logger: logger.info(f"TemperatureScaling: epoch {epoch+1}/{n_epochs}")
            self.model.eval()

            # --- train temperature on train_loader ---
            train_losses = []
            for batch in train_loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs = batch[0].to(device)
                    targets = batch[1].to(device)
                else:
                    raise RuntimeError("train_loader batch format not recognized; expected (input, target, ...).")

                with torch.no_grad():
                    try:
                        out = self.model(inputs)
                    except TypeError:
                        meta = batch[2] if len(batch) >= 3 else None
                        out = self.model(inputs, meta) if meta is not None else self.model(inputs)

                # apply temperature (model.scale preferred)
                if hasattr(self.model, "scale") and callable(getattr(self.model, "scale")):
                    scaled = self.model.scale(out)
                else:
                    temp = F.softplus(self.model.temperatures) + 1e-6
                    temp=temp.view(1,-1,1,1)
                    scaled = out / temp

                # NOTE: heatmap_nll expects logits (it does log_softmax internally),
                # so pass 'scaled' (logits) rather than 'probs'.
                loss = self.heatmap_nll(scaled, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(float(loss.item()))

            train_nll = float(np.mean(train_losses)) if train_losses else float("nan")

            # --- validation: compute val NLL + radial errors + mode probs ---
            val_losses = []
            radial_all = []
            mode_prob_all = []

            with torch.no_grad():
                # precompute things once
                pix_size = np.array(self.cfg.DATASET.PIXEL_SIZE)  # if you need pixel scaling later
                device = next(self.model.parameters()).device

                val_losses = []
                radial_all = []
                mode_prob_all = []

                for batch_idx, batch in enumerate(val_loader):
                    #print(f"Validation batch: {batch_idx}")

                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        inputs = batch[0].to(device, non_blocking=True)
                        targets = batch[1].to(device, non_blocking=True)  # assume heatmaps
                        # if you have metadata with coords, use it instead of argmax on targets
                        meta = batch[3] if len(batch) > 3 else None
                    else:
                        raise RuntimeError("val_loader batch format not recognized; expected (input, target, ...).")

                    # forward
                    try:
                        out = self.model(inputs) if meta is None else self.model(inputs, meta)
                    except TypeError:
                        out = self.model(inputs)

                    # apply temperature (reuse same code)
                    if hasattr(self.model, "scale") and callable(getattr(self.model, "scale")):
                        scaled = self.model.scale(out)
                    else:
                        temp = F.softplus(self.model.temperatures) + 1e-6
                        temp = temp.view(1, -1, 1, 1)   # shape (1, C, 1, 1)
                        scaled = out / temp

                    # NLL on GPU (still fine)
                    val_losses.append(float(self.heatmap_nll(scaled, targets).item()))

                    # compute probs on GPU (keep as torch tensor)
                    # two_d_softmax likely does softmax over H*W per-channel; implement inline for speed if needed
                    # Here we try to use your function if it accepts tensors; if not, inline:
                    try:
                        probs_t = self.two_d_softmax(scaled)  # expect torch.Tensor (B,C,H,W)
                        if not isinstance(probs_t, torch.Tensor):
                            probs_t = torch.from_numpy(probs_t).to(device)
                    except Exception:
                        # inline 2D softmax: (B,C,H,W) -> softmax over last two dims
                        B, C, H, W = scaled.shape
                        probs_t = torch.softmax(scaled.view(B, C, -1), dim=-1).view(B, C, H, W)

                    # --- Vectorized extraction of mode (max prob) and predicted coords ---
                    # mode_prob: max value per (B,C)
                    mode_vals = self.get_center_window_probability(probs_t, radius=1, remove_percent=5)

                    # convert flat index to (x,y)
                    B, C, H, W = probs_t.shape
                    probs_flat = probs_t.view(B, C, -1)
                    _, mode_idx = torch.max(probs_flat, dim=-1)
                    mode_idx = mode_idx.long()
                    ys = (mode_idx // W).to(torch.float32)
                    xs = (mode_idx % W).to(torch.float32)
                    pred_coords = torch.stack([xs, ys], dim=-1)         # (B, C, 2) in heatmap pixel space

                    # targets are heatmaps, argmax them to get GT coords
                    if targets is not None and targets.dim() == 4:
                        tgt_flat = targets.view(B, C, -1)
                        _, tgt_idx = torch.max(tgt_flat, dim=-1)
                        tgt_ys = (tgt_idx // W).to(torch.float32)
                        tgt_xs = (tgt_idx % W).to(torch.float32)
                        gt_coords = torch.stack([tgt_xs, tgt_ys], dim=-1)  # (B, C, 2)
                    else:
                        # If you already have explicit (x,y) coordinates in meta, compute gt_coords from meta here
                        raise RuntimeError("Expected targets as heatmaps to extract GT coords. If you have GT coords already, modify this block.")

                    # --- compute radial errors (Euclidean) on GPU - multiply by pixel size to get mm away
                    diffs = (pred_coords - gt_coords)*torch.tensor(pix_size).to(self.device)           # (B, C, 2)
                    dists = torch.norm(diffs, dim=-1)        # (B, C) distances in pixel units (heatmap resolution)
                    # If your pixel scaling differs, multiply by pixel size here, e.g. dists *= pix_size[0]

                    # append results (move small tensors to CPU once)
                    radial_all.append(dists.cpu().numpy())     # keep per-batch arrays, small-ish
                    mode_prob_all.append(mode_vals.cpu().numpy())

                # after loop: concatenate
                radial_all = np.concatenate(radial_all, axis=0)      # shape (N_samples, C)
                mode_prob_all = np.concatenate(mode_prob_all, axis=0)  # shape (N_samples, C)

                # flatten as your downstream code expects
                radial_flat = radial_all.ravel()
                mode_flat = mode_prob_all.ravel()

                # now you can compute reliability / ece using your existing helper:
                # e.g. valid_mask = np.isfinite(radial_flat)
                valid_mask = np.isfinite(radial_flat)
                mode_valid = mode_flat[valid_mask]
                radial_valid = radial_flat[valid_mask]

                # rest of your code: convert to correctness, plotting, etc.
                tol_px = tol_px *self.cfg.DATASET.PIXEL_SIZE[0]
                plot_path = os.path.join(save_dir, f"reliability_epoch_{epoch+1}.png")
                ece, bin_conf, bin_acc, bin_counts = self.reliability_diagram(
                    radial_valid, mode_valid, plot_path, tol_px ,n_bins
                )

            # --- After iterating all validation batches: flatten, sanitize, compute ECE once ---
            val_nll = float(np.mean(val_losses)) if val_losses else float("nan")

            if len(mode_prob_all) > 0:
                mode_flat = np.concatenate([np.ravel(m) for m in mode_prob_all])
                radial_flat = np.concatenate([np.ravel(r) for r in radial_all])
            else:
                mode_flat = np.array([])
                radial_flat = np.array([])

            # sanitize: remove NaN/inf and ensure same length
            mask = np.isfinite(mode_flat) & np.isfinite(radial_flat)
            mode_flat = mode_flat[mask]
            radial_flat = radial_flat[mask]

            # quick shape/coverage diagnostics
            N_pairs = mode_flat.size
            if logger: logger.info("TempScale: validation pairs for ECE: %d (B*C total)", N_pairs)
            # fallback: if no valid pairs, skip ECE
            if N_pairs == 0:
                ece = float("nan")
                if logger: logger.warning("No valid (confidence,radial) pairs this epoch; skipping ECE.")
            else:
                # auto-suggest n_bins (reduce if too few samples)
                def suggest_bins(N_pairs, requested= n_bins, target_per_bin=30, min_per_bin=5):
                    if N_pairs < requested * min_per_bin:
                        max_bins = max(1, N_pairs // min_per_bin)
                        return max_bins
                    if N_pairs < requested * target_per_bin:
                        return max(2, N_pairs // target_per_bin)
                    return requested

                chosen_bins = suggest_bins(N_pairs, n_bins)
                if logger:
                    logger.info("Using n_bins=%d for ECE (requested %d).", chosen_bins, n_bins)

                # compute ECE + plot once
                plot_path = os.path.join(save_dir, f"reliability_epoch_{epoch+1}.png")
                ece, bin_conf, bin_acc, bin_counts = self.reliability_diagram(radial_flat, mode_flat,
                                                                               plot_path, tol_px,
                                                                               n_of_bins=chosen_bins, x_max=0.2, do_not_save=False)
                if logger: logger.info("Saved reliability diagram to %s (ECE=%.4f)", plot_path, ece)

                # Save model if ECE improved
                if not math.isnan(ece) and ece < best_ece:
                    best_ece = ece
                    best_temps = F.softplus(self.model.temperatures).detach().cpu()
                    # record best_temp as scalar or list
                    best_temp = float(best_temps.item()) if best_temps.numel() == 1 else best_temps.view(-1).tolist()
                    best_model_path = os.path.join(save_dir, "best_temperature_model.pth")
                    torch.save(self.model.state_dict(), best_model_path)
                    temp_save = os.path.join(save_dir, "best_temperature.pt")
                    try:
                        torch.save({"temperatures": best_temps}, temp_save)
                    except Exception:
                        torch.save({"temperatures": best_temps.numpy()}, temp_save)
                    if logger:
                        logger.info("New best ECE %.6f; saved model + temperatures %s", best_ece, best_temps.tolist())


        # ensure final temperature attached and scale method available
        # robust final_temp extraction: scalar float if single-element, else list
        t_raw = self._get_temperature().detach().cpu()
        t_vals = t_raw.view(-1)
        if t_vals.numel() == 1:
            final_temp = float(t_vals.item())
        else:
            final_temp = t_vals.tolist()

        if not hasattr(self.model, "temperatures") or not isinstance(getattr(self.model, "temperatures"), torch.Tensor):
            if isinstance(final_temp, float):
                setattr(self.model, "temperatures", nn.Parameter(torch.tensor(final_temp, device=device)))
            else:
                setattr(self.model, "temperatures", nn.Parameter(torch.tensor(np.array(final_temp), device=device)))

        if not hasattr(self.model, "scale") or not callable(getattr(self.model, "scale")):
            def _scale(self_model, out):
                T = F.softplus(self_model.temperatures) + 1e-6
                return out / T
            self.model.scale = types.MethodType(_scale, self.model)
            self._patched_scale = True

        # store best info in returned dict
        result = {
            "final_temp": final_temp,
            "history": self.history,
            "best_ece": best_ece if 'best_ece' in locals() else None,
            "best_temp": best_temp if 'best_temp' in locals() else None,
            "best_model_path": best_model_path if 'best_model_path' in locals() else None,
            "csv_path": csv_path
        }
        return result