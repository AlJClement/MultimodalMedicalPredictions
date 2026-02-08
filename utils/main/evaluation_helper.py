import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
class evaluation_helper():
    def __init__(self) -> None:
        self.output_path = './check_gumbel'
        return

    def plot_gumbel_hist_per_channel(
        self, gumbel_noise,
        bins=150,
        alpha=0.5,
        title="Histogram of Gumbel Noise per Channel"
    ):
        """
        Plot histogram of Gumbel noise per channel, aggregated over batch and spatial dims.

        Args:
            gumbel_noise (torch.Tensor): shape (B, C, N)
            bins (int): number of histogram bins
            alpha (float): transparency for overlapping histograms
            title (str): plot title
        """
        B, C, N = gumbel_noise.shape

        plt.figure(figsize=(10, 6))

        for c in range(C):
            channel_values = gumbel_noise[:, c, :].reshape(-1).cpu().numpy()
            plt.hist(
                channel_values,
                bins=bins,
                density=True,
                alpha=alpha,
                label=f"channel {c}"
            )

        plt.xlabel("Gumbel noise value")
        plt.ylabel("Density")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path+'gumbel_noise_check.png')
        plt.close()
        return
    
    def heatmaps_to_logits(self, hm, mask=None, eps=1e-6, scale_if_flat=10.0, neg_inf=-1e9):
        """
        Convert heatmaps -> logits suitable for Gumbel-Softmax.

        Args:
            hm: tensor [B, C, H, W] with values expected in [0,1] (may contain exact zeros)
            mask: optional tensor [B, C, H, W] with 1 for valid pixels and 0 for background.
                If None, pixels with hm <= eps are treated as background for masking purposes.
            eps: small value for numerical stability when clamping probabilities
            scale_if_flat: multiplier applied to logits when spatial contrast is extremely low
            neg_inf: value used to mask logits (use a large negative number, not actual -inf)

        Returns:
            logits: tensor [B, C, H, W] (unbounded) ready to be flattened and passed to Gumbel-Softmax.
        """
        if not torch.is_tensor(hm):
            raise TypeError("hm must be a torch.Tensor")
        B, C, H, W = hm.shape
        device = hm.device

        # clamp for numerical stability (avoid exact 0/1)
        hm_clamped = hm.clamp(min=eps, max=1.0 - eps)

        # detect if spatially normalized (sum across H*W ~ 1)
        spatial_sum = hm_clamped.sum(dim=(2, 3))  # (B, C)
        is_spatial_prob = torch.allclose(spatial_sum, torch.ones_like(spatial_sum), atol=1e-3)

        # compute logits depending on interpretation
        if is_spatial_prob:
            # inverse of spatial softmax (up to additive constant)
            logits = torch.log(hm_clamped)
        else:
            # per-pixel probabilities -> logit (log(p/(1-p)))
            # prefer torch.logit if available for clarity
            try:
                logits = torch.logit(hm_clamped)  # stable native op in modern PyTorch
            except Exception:
                logits = torch.log(hm_clamped) - torch.log1p(-hm_clamped)

        # Build mask: if user provided mask use it, otherwise treat very-small probs as background
        if mask is None:
            # mask 1 where original hm is > eps, 0 otherwise
            active_mask = (hm > eps).to(device)
        else:
            # ensure boolean-like mask
            active_mask = (mask != 0).to(device)

        # Apply masking in *logit* space so masked pixels can never win Gumbel argmax
        # Use a very large negative number (neg_inf) instead of -inf to keep device dtype finite
        logits = logits.masked_fill(~active_mask, float(neg_inf))

        # Optional: if logits are too flat (low top1-top2 gap), optionally scale them to increase contrast
        K = H * W
        flat = logits.view(B, C, K)

        # If a channel is fully masked or has less than 2 valid pixels, topk might return
        # many identical -neg_inf values, so handle it safely:
        # - compute topk; if second best is ~neg_inf then gap will be large and no scaling needed.
        top2 = torch.topk(flat, k=2, dim=2).values  # (B, C, 2)
        gap_per_channel = (top2[:, :, 0] - top2[:, :, 1]).abs()  # (B, C)
        mean_gap = gap_per_channel.mean().item()  # scalar across batch+channels

        if mean_gap < 1e-3:
            logits = logits * float(scale_if_flat)

        return logits

    # def heatmaps_to_logits(self, hm, eps=1e-6, scale_if_flat=10.0):
    #     """
    #     hm: tensor [B, C, H, W] whose values may be clipped to [0,1]
    #     Returns: logits tensor same shape (unbounded) suitable for gumbel-softmax
    #     Strategy:
    #     - if per-channel spatial sum ≈ 1.0 -> treat as spatial probs (softmax-like)
    #         and use logits = log(p + eps)
    #     - else -> treat as pixelwise probabilities in [0,1], use logit(p) = log(p/(1-p))
    #     - fallback: if very flat (low top1-top2 gap), multiply logits by `scale_if_flat`
    #     """
    #     if not torch.is_tensor(hm):
    #         raise TypeError("hm must be a torch.Tensor")
    #     B, C, H, W = hm.shape
    #     device = hm.device

    #     # clamp for numerical stability
    #     hm_clamped = hm.clamp(min=eps, max=1.0 - eps)

    #     # detect if spatially normalized (sum across H*W ~ 1)
    #     spatial_sum = hm_clamped.sum(dim=(2,3))  # (B, C)
    #     is_spatial_prob = torch.allclose(spatial_sum, torch.ones_like(spatial_sum), atol=1e-3)

    #     if is_spatial_prob:
    #         # If already probabilities over spatial positions, logits = log(p)
    #         # Note: additive constant doesn't affect argmax, so this is fine.
    #         logits = torch.log(hm_clamped)
    #     else:
    #         # Pixelwise probability in (0,1) then computer logit
    #         logits = torch.log(hm_clamped) - torch.log1p(-hm_clamped)  # log(p/(1-p))

    #     return logits
    
    def get_hottest_points(self, img, gumbel=False, increase_tau=False, cfg=None, tau=1.0, eps=1e-8,
                        return_int=False, seed=None, hard=False):
        """
        img: tensor [B, C, W, H]  (e.g. [4, 6, 512, 512])
        Returns:
        points: float tensor [B, C, 2] with (y, x) coordinates (float)
        if return_int: also returns points_int: long tensor [B, C, 2] with integer coords (y, x)

        Notes:
        - Flattening order is idx = x * H + y (x major, y minor).
        - Both branches compute coords with the same grid so outputs are comparable.
        - hard=True uses straight-through estimator for Gumbel branch (forward discrete).
        """
        if cfg is not None:
            tau_decay = cfg.TRAIN.TAU_DECAY 
            tau = cfg.TRAIN.TAU

        if seed is not None:
            torch.manual_seed(seed)

        B, C, W, H = img.size()
        # flatten spatial dims -> (B, C, W*H)
        img = self.heatmaps_to_logits(img)
        flattened = img.view(B, C, W * H)

        device = img.device
        # build coordinate grids matching idx = x*H + y ordering
        xs = torch.arange(W, device=device).unsqueeze(1).repeat(1, H).view(-1).float()  # (W*H,)
        ys = torch.arange(H, device=device).unsqueeze(0).repeat(W, 1).view(-1).float()  # (W*H,)
        xs = xs.view(1, 1, -1)  # (1,1,W*H)
        ys = ys.view(1, 1, -1)

        if not gumbel:
            # deterministic argmax and compute coords with same procedure
            max_idx = torch.argmax(flattened, dim=2, keepdim=True)               # (B, C, 1)
            one_hot = torch.zeros_like(flattened).scatter_(2, max_idx, 1.0)      # (B, C, W*H)
            x_f = torch.sum(one_hot * xs, dim=2)  # (B, C) floats
            y_f = torch.sum(one_hot * ys, dim=2)  # (B, C) floats
            points = torch.stack((y_f, x_f), dim=2)  # (B, C, 2) floats

        else:
            print('gumbel used! tau:', tau)
            # sample gumbel noise and softmax over flattened spatial dim
            u = torch.rand_like(flattened).clamp(min=eps, max=1.0 - eps) 
            ## u creates a tensor with the same shape as flattened
            # values are sampled independently from a Uniform(0, 1) distribution
            gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
            # self.plot_gumbel_hist_per_channel(gumbel_noise)
            # masked_logits = flattened.clone()
            # masked_logits = masked_logits.masked_fill(~mask, -1e9)  # effectively -inf for softmax

            if increase_tau == False:                    
                logits = (flattened + gumbel_noise) / float(tau)
                probs = F.softmax(logits, dim=2)
            else:
                #if increase_tau > 10:
                tau = tau - tau_decay* increase_tau
                if tau <tau_decay:
                    tau == tau_decay
                print('tau has reduced to:', tau)
                logits = (flattened + gumbel_noise) / float(tau)
                probs = F.softmax(logits, dim=2)  
                # else:
                #     logits = (flattened + gumbel_noise) / float(tau)
                #     probs = F.softmax(logits, dim=2)    

            if hard:
                # straight-through hard one-hot: forward discrete, backward through probs
                _, max_idx = probs.max(dim=2, keepdim=True)                 # (B, C, 1)
                one_hot = torch.zeros_like(probs).scatter_(2, max_idx, 1.0) # (B, C, W*H)
                chosen = one_hot - probs.detach() + probs                   # (B, C, W*H)
            else:
                # soft expected coordinate
                chosen = probs  # (B, C, W*H)

            x_f = torch.sum(chosen * xs, dim=2)  # (B, C)
            y_f = torch.sum(chosen * ys, dim=2)  # (B, C)
            points = torch.stack((y_f, x_f), dim=2)  # (B, C, 2) floats
            
            if np.isnan(points.detach().cpu().numpy()).any():
                print("points contains at least one NaN")
            ####plot the heatmap with the gumbel to confirm they are close by
            import matplotlib.pyplot as plt
            pts = points.detach().cpu().numpy()
            scan = 0
            img_np = img.detach().cpu().numpy()[scan]
            combined_img = img_np.sum(axis=0) 
            plt.imshow(combined_img) 
            for i in range(6):
                plt.scatter(pts[scan][i][0], pts[scan][i][1], s=5, color='red')#
                timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
                if not os.path.exists(self.output_path):
                    os.makedirs(self.output_path)               
            plt.savefig(self.output_path+f'/gumbel_check_{timestamp}.png')
            plt.close()

        if return_int:
            points_int = points.long()
            return points, points_int

        return points
    

    def get_thresholded_heatmap(self, pred, predicted_points_scaled, significant_radius=0.05):
        '''This function takes the output channels and thresholds the heatmaps to the significant radius, then renormalizes those values'''
        #dimension depend on the size of the pred. which depends on the number of annotators/if we are combining annotators to one map
        #flip points
        predicted_points_scaled = torch.flip(predicted_points_scaled, dims=[2])
        #print('out stack size', pred.shape)
        flattened_heatmaps = torch.flatten(pred, start_dim=2)
        #print('flattened_heatmaps size:', flattened_heatmaps.shape)
        max_per_heatmap, _ = torch.max(flattened_heatmaps, dim=2, keepdim=True)
        max_per_heatmap = torch.unsqueeze(max_per_heatmap, dim=3)
        #print(' max_per_heatmap size', max_per_heatmap.shape)

        normalized_heatmaps = torch.div(pred, max_per_heatmap)

        zero_tensor = torch.tensor(0.0).cuda() if pred.is_cuda else torch.tensor(0.0)
        filtered_heatmaps = torch.where(normalized_heatmaps > significant_radius, normalized_heatmaps,
                                        zero_tensor)
        flattened_filtered_heatmaps = torch.flatten(filtered_heatmaps, start_dim=2)
        sum_per_heatmap = torch.sum(flattened_filtered_heatmaps, dim=2, keepdim=True)
        sum_per_heatmap = torch.unsqueeze(sum_per_heatmap, dim=3)
        thresholded_output = torch.div(filtered_heatmaps, sum_per_heatmap)

        return thresholded_output
    
    def get_landmarks(self, pred, target_points, pixels_sizes, gumbel=False, increase_tau = False, cfg=None):
        # Predicted points has shape (B, N, 2)

        if not gumbel:
            predicted_points = self.get_hottest_points(pred)
            scaled_predicted_points = torch.multiply(predicted_points, pixels_sizes)

            # Get landmark center (B, N, 2)
            target_points = self.get_hottest_points(target_points) #should be center of gaussian
            scaled_target_points = torch.multiply(target_points, pixels_sizes)

            return  target_points, predicted_points
        else:
            predicted_points = self.get_hottest_points(pred, gumbel, increase_tau, cfg)

            # Get landmark center (B, N, 2)m
            target_points = self.get_hottest_points(target_points, gumbel,increase_tau, cfg) #should be center of gaussian

            return  target_points, predicted_points
    
    def get_landmarks_predonly(self, pred, pixels_sizes):
        # Predicted points has shape (B, N, 2)
        predicted_points = self.get_hottest_points(pred)
        scaled_predicted_points = torch.multiply(predicted_points, pixels_sizes)

        return  predicted_points