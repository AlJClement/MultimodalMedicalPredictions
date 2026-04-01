import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import datetime
from .model_init import model_init
from .loss import *
torch.cuda.empty_cache() 
from .comparison_metrics import graf_angle_calc
from .evaluation_helper import evaluation_helper
from .comparison_metrics import fhc
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler as AMPGradScaler

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

class training():
    def __init__(self, cfg, logger, l2_reg=True):
        self.plot_target = False
        self.cfg = cfg
        self.model_init = model_init(cfg)
        self.logger = logger
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.debug_training = bool(getattr(cfg.TRAIN, "DEBUG_LOGGING", False))
        
        #get specific models/feature loaders
        self.net, self.net_param = self.model_init.get_net_from_conf()
        self.net = self.net.to(self.device)
        self.l2_reg=l2_reg
        if l2_reg == True:
            self.loss_func = eval('L2RegLoss(cfg.TRAIN.LOSS)')
        else:
            self.loss_func = eval(cfg.TRAIN.LOSS)
        
        if (cfg.TRAIN.LOSS).split('_')[-1]=='wclass':
            self.add_class_loss = True
            self.gamma = cfg.TRAIN.GAMMA
        else:
            self.add_class_loss = False

        if (cfg.TRAIN.LOSS).split('_')[-1]=='walpha':
            self.add_alpha_loss = True
            self.gamma = cfg.TRAIN.GAMMA
        else:
            self.add_alpha_loss = False

        if (cfg.TRAIN.LOSS).split('_')[-1]=='cosinelandmarkvector':
            self.add_landmark_loss = True
            self.gamma = cfg.TRAIN.GAMMA
        else:
            self.add_landmark_loss = False
        

        if (cfg.TRAIN.LOSS).split('_')[-1]=='walphafhc':
            self.add_alphafhc_loss = True
            self.gamma = cfg.TRAIN.GAMMA
        else:
            self.add_alphafhc_loss = False

        if 'diff' in cfg.TRAIN.LOSS:
            self.add_gumbel = True
            self.gamma = cfg.TRAIN.GAMMA
            self.delay_gumbel_loss = cfg.TRAIN.DELAY_GUMBEL_LOSS
            self.tau_decay = cfg.TRAIN.TAU_DECAY
        else:
            self.add_gumbel = False

        self.class_calculation = graf_angle_calc()
        self.fhc_calc = fhc()

        self.gamma = cfg.TRAIN.GAMMA

        self.bs = cfg.TRAIN.BATCH_SIZE
        self.lr = cfg.TRAIN.LR
        self.momentum = 0.99
        self.momentum_0 = 0.90
        self.optimizer = self._get_optimizer(self.net)
        # if cfg.MODEL.DEVICE == 'cuda':
        #     torch.cuda.empty_cache()
        self.pixel_size = torch.tensor(cfg.DATASET.PIXEL_SIZE, device=self.device)

        self.grad_accumulation_steps = cfg.TRAIN.GRAD_ACCUMULATION_STEPS
        self.grad_accumulation_steps_min = cfg.TRAIN.GRAD_ACCUMULATION_STEPS_MIN ## min value to stop
        self.grad_accumulation_steps_reduce = cfg.TRAIN.GRAD_ACCUMULATION_STEPS_REDUCE

        # AMP toggle + GradScaler (modern API). Use device type (e.g., 'cuda' or 'cpu').
        device_type = self.device.type  # 'cuda' or 'cpu'
        self.use_amp = getattr(cfg.TRAIN, "USE_AMP", True)
        # self.scaler = AMPGradScaler(device_type, init_scale=2**8) if self.use_amp else None
        if self.use_amp and torch.cuda.is_available() and self.device.type == 'cuda':
            from torch.cuda.amp import GradScaler as AMPGradScaler
            self.scaler = AMPGradScaler(init_scale=2**8)
        else:
            self.scaler = None
            self.use_amp = False
        # Early stopping config (parse safely from cfg)
        # Support both attribute-style and dict-style EARLY_STOPPING config.
        if hasattr(cfg.TRAIN, "EARLY_STOPPING") and cfg.TRAIN.EARLY_STOPPING is not None:
            es_cfg = cfg.TRAIN.EARLY_STOPPING
            # If it's a dict-like:
            if isinstance(es_cfg, dict):
                self.early_stopping_enabled = bool(es_cfg.get("ENABLED", False))
                self.early_stopping_patience = int(es_cfg.get("PATIENCE", 3))
                self.early_stopping_min_delta = float(es_cfg.get("MIN_DELTA", 1.0))
                self.early_stopping_mode = es_cfg.get("MODE", "min")
            else:
                # attribute style fallback
                self.early_stopping_enabled = bool(getattr(cfg.TRAIN, "EARLY_STOPPING_ENABLED", False))
                self.early_stopping_patience = int(getattr(cfg.TRAIN, "EARLY_STOPPING_PATIENCE", 3))
                self.early_stopping_min_delta = float(getattr(cfg.TRAIN, "EARLY_STOPPING_MIN_DELTA", 1.0))
                self.early_stopping_mode = getattr(cfg.TRAIN, "EARLY_STOPPING_MODE", "min")
        else:
            # defaults: disabled
            self.early_stopping_enabled = False
            self.early_stopping_patience = 3
            self.early_stopping_min_delta = 1.0
            self.early_stopping_mode = 'min'
        
                # initialize internal ES trackers (persist across epochs)
        self._es_best = None
        self._es_wait = 0

        pass

    def _debug_print(self, *args, **kwargs):
        if self.debug_training:
            print(*args, **kwargs)

    def _get_network(self):
        return self.net
    
    def _get_optimizer(self,net):
        #optim = torch.optim.SGD(net.parameters(), lr = self.lr, momentum=self.momentum)
        # optim = torch.optim.Adam(net.parameters(), lr=self.lr, betas=(self.momentum_0, self.momentum))

        ## updated for frozen models
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.lr, betas=(self.momentum_0, self.momentum))
        return optimizer
    
    def _freeze_layers(self, net):
        for layer in list(net.encoder.children())[:self.freeze]:
            for param in layer.parameters():
                param.requires_grad = False
        return net


    def train_meta(self, dataloader, epoch, val_dataloader=None):
        """
        Train for one epoch.

        Args:
            dataloader: training dataloader
            epoch: current epoch index (int)
            val_dataloader: optional validation dataloader (same item structure as train)
        Returns:
            av_loss (float) average training loss for this epoch
        other/also:
            - updates self.train_losses and self.val_losses lists
            - updates early stopping state self._es_should_stop (True/False)
            - saves a plot every 10 epochs to ./train_val_loss_epoch{epoch}.png
        """
        total_params = sum(p.numel() for p in self.net.parameters())
        trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self._debug_print(f"Total params: {total_params}")
        self._debug_print(f"Trainable params: {trainable_params}")
        self._debug_print(f"Frozen params: {total_params - trainable_params}")

        self.net.train()
        total_loss = 0.0
        batches = 0

        accum_steps = max(1, getattr(self, "grad_accumulation_steps", 1))
        if accum_steps > 1:
            if epoch % 50 == 0 and accum_steps > self.grad_accumulation_steps_min:
                new_acc = max(self.grad_accumulation_steps_min, accum_steps // 2)
                self.grad_accumulation_steps = new_acc
                accum_steps = new_acc
                self._debug_print(f"Reducing gradient accumulation steps to {accum_steps}")
            self._debug_print(f"Gradient accumulation enabled: {accum_steps} steps (effective batch size = {accum_steps * self.bs})")

        # zero grads at epoch start
        self.optimizer.zero_grad(set_to_none=True)
        epoch_start = datetime.datetime.now()

        for batch_idx, (data, target, landmarks, meta, id, orig_size, orig_img) in enumerate(dataloader):
            batches += 1
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            meta = meta.to(self.device, non_blocking=True)

            # debug prints for micro-batches
            if batch_idx % 10 == 0:
                self._debug_print(f"\n--- batch_idx={batch_idx} ---")
                # print(f"data.shape={tuple(data.shape)}, target.shape={tuple(target.shape)}")
                # if self.use_amp:
                #     try:
                #         print("AMP enabled. GradScaler scale =", self.scaler.get_scale())
                        
                #     except Exception:
                #         print("Could not read scaler scale")

            # forward + loss under autocast
            with autocast(enabled=self.use_amp):
                pred = self.net(data, meta)
                if batch_idx % 10 == 0:
                    if isinstance(pred, torch.Tensor):
                        self._debug_print(f"pred.shape={pred.shape}, pred min={float(pred.detach().min()):.6e}, max={float(pred.detach().max()):.6e}, mean={float(pred.detach().mean()):.6e}")

                # compute targets if needed
                if self.add_class_loss or self.add_alpha_loss or self.add_alphafhc_loss:
                    pred_alphas, pred_classes, target_alphas, target_classes = self.class_calculation.get_class_from_output(pred, target, self.pixel_size)
                if self.add_alphafhc_loss:
                    pred_fhc, target_fhc = self.fhc_calc.get_fhc_batches(pred, target, self.pixel_size)

                # compute main loss (preserves your branching)
                if self.l2_reg:
                    if self.add_class_loss:
                        loss = self.loss_func(pred, target, self.net, self.gamma, pred_alphas, target_alphas, pred_classes, target_classes)
                    elif self.add_alphafhc_loss:
                        loss = self.loss_func(pred, target, self.net, self.gamma, pred_alphas, target_alphas, pred_classes, target_classes, pred_fhc, target_fhc)
                    elif self.add_landmark_loss:
                        loss = self.loss_func(pred, target, self.net, self.gamma)
                    elif self.add_alpha_loss:
                        loss = self.loss_func(pred, target, self.net, self.gamma, pred_alphas, target_alphas)
                    elif self.add_gumbel:
                        loss = self.loss_func(pred, target, self.net, self.gamma, self.cfg)
                    else:
                        loss = self.loss_func(pred, target, self.net, self.gamma)
                else:
                    if self.add_class_loss:
                        loss = self.loss_func(pred, target, self.net, pred_alphas, target_alphas, pred_classes, target_classes, self.gamma)
                    elif self.add_alpha_loss:
                        loss = self.loss_func(pred, target, self.net, pred_alphas, target_alphas, self.gamma)
                    else:
                        loss = self.loss_func(pred, target, self.net)

            # sanity checks on loss
            if not isinstance(loss, torch.Tensor):
                raise RuntimeError("Loss returned not a torch.Tensor — cannot call backward()")

            if batch_idx % 10 == 0:
                try:
                    raw_loss = float(loss.item())
                    pixels = float(data.numel())
                    self._debug_print(f"raw loss = {raw_loss:.6e}, loss/pixel = {raw_loss/pixels:.6e}, pixels = {int(pixels)}")
                    if not torch.isfinite(loss).all():
                        self._debug_print(">>>> Loss is not finite (NaN or Inf) at batch", batch_idx)
                except Exception:
                    self._debug_print("Could not read loss.item()")

            total_loss += loss.item()

            # scale/divide for accumulation
            loss = loss / accum_steps

            # backward (AMP aware)
            if self.use_amp:
                if batch_idx % 50 == 0:
                    try:
                        self._debug_print("scaler.get_scale() before backward =", self.scaler.get_scale())
                    except Exception:
                        pass
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # debug: grad existence & sample stats
            if batch_idx % 10 == 0:
                grads_exist = [p.grad is not None for p in self.net.parameters() if p.requires_grad]
                any_grad = any(grads_exist)
                self._debug_print(f"[batch {batch_idx}] any_grad after backward: {any_grad}")
                max_grad = None
                total_norm = 0.0
                found = False
                sample_count = 0
                for p in self.net.parameters():
                    if p.requires_grad and p.grad is not None:
                        try:
                            g = p.grad.detach()
                            g_abs_max = float(g.abs().max())
                            if max_grad is None or g_abs_max > max_grad:
                                max_grad = g_abs_max
                            total_norm += float(torch.norm(g).item() ** 2)
                            found = True
                            sample_count += 1
                            if sample_count >= 5:
                                break
                        except Exception:
                            pass
                if found:
                    total_norm = total_norm ** 0.5
                    self._debug_print(f"[batch {batch_idx}] grad sample max abs = {max_grad:.6e}, grad sample norm = {total_norm:.6e}")
                else:
                    self._debug_print(f"[batch {batch_idx}] No gradients found on sampled params after backward")

            # optimizer step (every accum_steps)
            if (batch_idx + 1) % accum_steps == 0:
                self._debug_print(f"[batch {batch_idx}] Performing optimizer step (accum count reached).")
                if self.use_amp:
                    # unscale so checks/clipping are meaningful
                    try:
                        self.scaler.unscale_(self.optimizer)
                    except Exception as e:
                        self._debug_print("Warning: scaler.unscale_() failed:", e)

                    # check grads for NaN/Inf or huge values
                    bad_grad = False
                    for name, p in self.net.named_parameters():
                        if not p.requires_grad or p.grad is None:
                            continue
                        g = p.grad.detach()
                        if torch.isnan(g).any() or torch.isinf(g).any():
                            self._debug_print(f"NON-FINITE GRAD: {name} min/max = {float(g.min()):.3e}/{float(g.max()):.3e}")
                            bad_grad = True
                            break

                    if bad_grad:
                        # skip optimizer step, let the scaler reduce its scale
                        self._debug_print(f"[batch {batch_idx}] Skipping optimizer.step() due to non-finite grads. Calling scaler.update() and zeroing grads.")
                        try:
                            self.scaler.update()
                        except Exception as e:
                            self._debug_print("Warning: scaler.update() failed:", e)
                        self.optimizer.zero_grad(set_to_none=True)
                    else:
                        # OPTIONAL: clip grads here if you want (after unscale)
                        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.net.parameters()), max_norm=5.0)

                        try:
                            self.scaler.step(self.optimizer)
                        except Exception as e:
                            self._debug_print("scaler.step() raised:", e)
                            self.optimizer.zero_grad(set_to_none=True)
                            try:
                                self.scaler.update()
                            except Exception:
                                pass
                        else:
                            self.scaler.update()
                else:
                    self.optimizer.step()

                # zero grads for next accumulation block
                self.optimizer.zero_grad(set_to_none=True)
                self._debug_print(f"optimizer.step() at batch {batch_idx}")

            # periodic logging
            if batch_idx % 50 == 0:
                display_loss = (loss.item() * accum_steps)
                print(f"Train Epoch: {epoch} [{(batch_idx + 1) * len(data)}/{len(dataloader.dataset)} ({100. * (batch_idx + 1) / len(dataloader):.0f}%)]\tLoss: {display_loss:.6f}", flush=True)

            # cleanup to free memory early
            del pred
            del loss
            if 'pred_alphas' in locals(): del pred_alphas
            if 'pred_classes' in locals(): del pred_classes
            if 'target_alphas' in locals(): del target_alphas
            if 'target_classes' in locals(): del target_classes
            if 'pred_fhc' in locals(): del pred_fhc
            if 'target_fhc' in locals(): del target_fhc

        # final leftover step (if any)
        if (batch_idx + 1) % accum_steps != 0:
            self._debug_print("Final leftover optimizer step (not aligned to accum_steps).")
            if self.use_amp:
                try:
                    self.scaler.unscale_(self.optimizer)
                except Exception:
                    pass
                try:
                    self.scaler.step(self.optimizer)
                except Exception as e:
                    self._debug_print("scaler.step() (final) raised:", e)
                    try:
                        self.scaler.update()
                    except Exception:
                        pass
                else:
                    self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._debug_print("optimizer.step() at final leftover batch")

        av_loss = total_loss / batches
        print(f"\nTraining Set Average loss: {av_loss:.6f}", flush=True)
        epoch_end = datetime.datetime.now()
        print('Time taken for epoch = ', epoch_end - epoch_start)

        # === VALIDATION PASS (optional) ===
        val_loss = None
        if val_dataloader is not None:
            was_training = self.net.training
            self.net.eval()
            total_val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for v_batch_idx, (v_data, v_target, v_landmarks, v_meta, v_id, v_orig_size, v_orig_img) in enumerate(val_dataloader):
                    v_data = v_data.to(self.device, non_blocking=True)
                    v_target = v_target.to(self.device, non_blocking=True)
                    v_meta = v_meta.to(self.device, non_blocking=True)

                    # compute forward (no accumulation on val)
                    with autocast(enabled=self.use_amp):
                        v_pred = self.net(v_data, v_meta)

                        # compute any auxiliary targets same as train
                        if self.add_class_loss or self.add_alpha_loss or self.add_alphafhc_loss:
                            v_pred_alphas, v_pred_classes, v_target_alphas, v_target_classes = self.class_calculation.get_class_from_output(v_pred, v_target, self.pixel_size)
                        if self.add_alphafhc_loss:
                            v_pred_fhc, v_target_fhc = self.fhc_calc.get_fhc_batches(v_pred, v_target, self.pixel_size)

                        # compute val loss using same branches
                        if self.l2_reg:
                            if self.add_class_loss:
                                v_loss = self.loss_func(v_pred, v_target, self.net, self.gamma, v_pred_alphas, v_target_alphas, v_pred_classes, v_target_classes)
                            elif self.add_alphafhc_loss:
                                v_loss = self.loss_func(v_pred, v_target, self.net, self.gamma, v_pred_alphas, v_target_alphas, v_pred_classes, v_target_classes, v_pred_fhc, v_target_fhc)
                            elif self.add_landmark_loss:
                                v_loss = self.loss_func(v_pred, v_target, self.net, self.gamma)
                            elif self.add_alpha_loss:
                                v_loss = self.loss_func(v_pred, v_target, self.net, self.gamma, v_pred_alphas, v_target_alphas)
                            elif self.add_gumbel:
                                v_loss = self.loss_func(v_pred, v_target, self.net, self.gamma, self.cfg)
                            else:
                                v_loss = self.loss_func(v_pred, v_target, self.net, self.gamma)
                        else:
                            if self.add_class_loss:
                                v_loss = self.loss_func(v_pred, v_target, self.net, v_pred_alphas, v_target_alphas, v_pred_classes, v_target_classes, self.gamma)
                            elif self.add_alpha_loss:
                                v_loss = self.loss_func(v_pred, v_target, self.net, v_pred_alphas, v_target_alphas, self.gamma)
                            else:
                                v_loss = self.loss_func(v_pred, v_target, self.net)

                    total_val_loss += float(v_loss.item())
                    val_batches += 1

            val_loss = total_val_loss / max(1, val_batches)
            print(f"Validation Set Average loss: {val_loss:.6f}", flush=True)
            # restore mode
            if was_training:
                self.net.train()

        if not hasattr(self, "train_losses"):
            self.train_losses = []
        if not hasattr(self, "val_losses"):
            self.val_losses = []

        self.train_losses.append(av_loss)
        # append val loss or nan if no val provided
        self.val_losses.append(val_loss if val_loss is not None else float("nan"))
        self.last_val_loss = val_loss

        # === EARLY STOPPING CHECK ===
        # initialize if first time
        if self._es_best is None:
            self._es_best = val_loss if (self.early_stopping_enabled and val_dataloader is not None) else av_loss
            self._es_wait = 0
            self._es_should_stop = False

        if self.early_stopping_enabled:
            # choose metric depending if val available else train
            metric = val_loss if val_dataloader is not None else av_loss
            if metric is None:
                # can't evaluate early stopping without a metric
                self._debug_print("Early stopping enabled but no metric available this epoch (val missing). Skipping ES update.")
                self._es_should_stop = False
            else:
                improved = False
                if self.early_stopping_mode == 'min':
                    if metric < (self._es_best - self.early_stopping_min_delta):
                        improved = True
                else:  # 'max'
                    if metric > (self._es_best + self.early_stopping_min_delta):
                        improved = True

                if improved:
                    self._es_best = metric
                    self._es_wait = 0
                    self._debug_print(f"EarlyStopping: improvement detected (best -> {self._es_best:.6f}). Reset wait.")
                else:
                    self._es_wait += 1
                    self._debug_print(f"EarlyStopping: no improvement. Wait = {self._es_wait}/{self.early_stopping_patience}")
                    if self._es_wait >= self.early_stopping_patience:
                        self._debug_print("EarlyStopping: patience exceeded. Will request stop.")
                        self._es_should_stop = True

        else:
            # early stopping disabled
            self._es_should_stop = False

        if epoch % 10 == 0:
            try:
                epochs = list(range(1, len(self.train_losses) + 1))

                plt.figure(figsize=(8, 5))

                # ---- Training loss (filter NaNs just in case) ----
                train_plot_x = []
                train_plot_y = []
                for i, v in enumerate(self.train_losses):
                    if v is None:
                        continue
                    if isinstance(v, float) and (v != v):  # NaN check
                        continue
                    train_plot_x.append(i + 1)
                    train_plot_y.append(v)

                if len(train_plot_x) > 0:
                    plt.plot(train_plot_x, train_plot_y, label="Train Loss")

                # ---- Validation loss ----
                val_plot_x = []
                val_plot_y = []
                for i, v in enumerate(self.val_losses):
                    if v is None:
                        continue
                    if isinstance(v, float) and (v != v):
                        continue
                    val_plot_x.append(i + 1)
                    val_plot_y.append(v)

                if len(val_plot_x) > 0:
                    plt.plot(val_plot_x, val_plot_y, label="Val Loss")

                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Train/Val Loss (epoch {epoch})")
                plt.legend()
                plt.grid(True)

                fname = self.cfg.OUTPUT_PATH + f"/train_val_loss_epoch{epoch}.png"
                plt.savefig(fname, bbox_inches="tight")
                plt.close()

                self._debug_print(f"Saved train/val loss plot to {fname}")

            except Exception as e:
                self._debug_print("Plotting failed:", e)

        return av_loss
