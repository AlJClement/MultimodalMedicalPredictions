import torch.nn as nn
import torch
from torch.autograd import Variable
import datetime
from .model_init import model_init
from .loss import *
torch.cuda.empty_cache() 
from .comparison_metrics import graf_angle_calc
from .evaluation_helper import evaluation_helper
from .comparison_metrics import fhc
from torch.cuda.amp import autocast, GradScaler

class training():
    def __init__(self, cfg, logger, l2_reg=True):
        self.plot_target = False
        self.cfg = cfg
        self.model_init = model_init(cfg)
        self.logger = logger
        #get specific models/feature loaders
        self.net, self.net_param = self.model_init.get_net_from_conf()
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
        self.device = torch.device(cfg.MODEL.DEVICE)
        # if cfg.MODEL.DEVICE == 'cuda':
        #     torch.cuda.empty_cache()
        self.pixel_size = torch.tensor(cfg.DATASET.PIXEL_SIZE).to(cfg.MODEL.DEVICE)

        self.grad_accumulation_steps = cfg.TRAIN.GRAD_ACCUMULATION_STEPS
        self.grad_accumulation_steps_min = cfg.TRAIN.GRAD_ACCUMULATION_STEPS_MIN ## min value to stop
        self.grad_accumulation_steps_reduce = cfg.TRAIN.GRAD_ACCUMULATION_STEPS_REDUCE
        
        # Add a config flag in your cfg e.g. cfg.TRAIN.USE_AMP (default True)
        self.use_amp = getattr(cfg.TRAIN, "USE_AMP", True)
        self.scaler = GradScaler() if self.use_amp else None
        pass

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
    #def train_meta(self, dataloader, epoch):
        total_params = sum(p.numel() for p in self.net.parameters())
        trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Total params: {total_params}")
        print(f"Trainable params: {trainable_params}")
        print(f"Frozen params: {total_params - trainable_params}")

        self.net.train()
        total_loss = 0.0
        batches = 0

        # Use accumulation config always (unless it's 1)
        accum_steps = max(1, getattr(self, "grad_accumulation_steps", 1))
        if accum_steps > 1:
            if epoch % 50 == 0 and accum_steps > self.grad_accumulation_steps_min:
                # compute new value but assign to local var and to attribute consistently
                new_acc = max(self.grad_accumulation_steps_min, accum_steps // 2)
                self.grad_accumulation_steps = new_acc
                accum_steps = new_acc
                print(f"Reducing gradient accumulation steps to {accum_steps}")

            print(f"Gradient accumulation enabled: {accum_steps} steps (effective batch size = {accum_steps * self.bs})")

        # zero grads at epoch start
        self.optimizer.zero_grad(set_to_none=True)

        epoch_start = datetime.datetime.now()
        for batch_idx, (data, target, landmarks, meta, id, orig_size, orig_img) in enumerate(dataloader):
            batches += 1

            # Move tensors to device (no Variables)
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            meta = meta.to(self.device, non_blocking=True)

            # ---------- Use autocast for mixed precision around forward+loss ----------
            with autocast(enabled=self.use_amp):  # <<< FIX: ensure autocast is used
                pred = self.net(data, meta)

                # compute any extra needed values (alphas/classes/fhc) as in your original
                if self.add_class_loss or self.add_alpha_loss or self.add_alphafhc_loss:
                    pred_alphas, pred_classes, target_alphas, target_classes = \
                        self.class_calculation.get_class_from_output(pred, target, self.pixel_size)
                if self.add_alphafhc_loss:
                    pred_fhc, target_fhc = self.fhc_calc.get_fhc_batches(pred, target, self.pixel_size)

                # compute loss (kept same branching as your original)
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

            # Make sure loss is a tensor and connected
            if not isinstance(loss, torch.Tensor):
                raise RuntimeError("Loss returned not a torch.Tensor — cannot call backward()")

            total_loss += loss.item()

            # scale loss for accumulation then backward
            loss = loss / accum_steps

            # ---------- backward (scaled when using AMP) ----------
            if self.use_amp:
                # scale and backward using GradScaler
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # ---------- DEBUG: print grad existence and stats after backward ----------
            if batch_idx % 100 == 0:
                grads_exist = [p.grad is not None for p in self.net.parameters() if p.requires_grad]
                any_grad = any(grads_exist)
                print(f"[batch {batch_idx}] any_grad after backward: {any_grad}")

                # compute a few grad stats (max and norm) safely
                max_grad = None
                total_norm = 0.0
                found = False
                for p in self.net.parameters():
                    if p.requires_grad and p.grad is not None:
                        try:
                            g = p.grad.detach()
                            g_abs_max = g.abs().max().item()
                            if max_grad is None or g_abs_max > max_grad:
                                max_grad = g_abs_max
                            total_norm += float(torch.norm(g).item() ** 2)
                            found = True
                        except Exception:
                            pass
                if found:
                    total_norm = total_norm ** 0.5
                    print(f"[batch {batch_idx}] grad max abs = {max_grad:.6e}, grad norm = {total_norm:.6e}")

            # ---------- optimizer step every accum_steps ----------
            if (batch_idx + 1) % accum_steps == 0:
                # OPTIONAL: if you want to clip grads when using AMP, do:
                # if self.use_amp:
                #     self.scaler.unscale_(self.optimizer)  # unscale before clipping
                # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.net.parameters()), max_norm)

                # print gradient stats just BEFORE the step (optional, helpful)
                if (batch_idx + 1) % (accum_steps * 10) == 0:  # less frequent to avoid spam
                    # compute a cheap grad max for logging
                    max_grad_pre = None
                    for p in self.net.parameters():
                        if p.requires_grad and p.grad is not None:
                            try:
                                mg = p.grad.abs().max().item()
                                if max_grad_pre is None or mg > max_grad_pre:
                                    max_grad_pre = mg
                            except Exception:
                                pass
                    print(f"[batch {batch_idx}] max grad pre-step = {max_grad_pre}")

                if self.use_amp:
                    # <<< FIX: use scaler.step + scaler.update with AMP
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # zero grads for next accumulation block
                self.optimizer.zero_grad(set_to_none=True)
                print(f"optimizer.step() at batch {batch_idx}")  # help confirm stepping frequency

            # periodic logging (note: loss is the divided version — multiply back for display)
            if batch_idx % 100 == 0:
                display_loss = (loss.item() * accum_steps)
                print(f"Train Epoch: {epoch} [{(batch_idx + 1) * len(data)}/{len(dataloader.dataset)} ({100. * (batch_idx + 1) / len(dataloader):.0f}%)]\tLoss: {display_loss:.6f}", flush=True)

            # cleanup
            del pred
            del loss
            # del pred_alphas etc. if they exist to free memory earlier
            if 'pred_alphas' in locals():
                del pred_alphas
            if 'pred_classes' in locals():
                del pred_classes
            if 'target_alphas' in locals():
                del target_alphas
            if 'target_classes' in locals():
                del target_classes
            if 'pred_fhc' in locals():
                del pred_fhc
            if 'target_fhc' in locals():
                del target_fhc

        # ---------- final step for leftover gradients (AMP-aware) ----------
        if (batch_idx + 1) % accum_steps != 0:
            if self.use_amp:
                # <<< FIX: use scaler.step + scaler.update for final leftover
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            print("optimizer.step() at final leftover batch")

        av_loss = total_loss / batches
        print(f"\nTraining Set Average loss: {av_loss:.4f}", flush=True)

        epoch_end = datetime.datetime.now()
        print('Time taken for epoch = ', epoch_end - epoch_start)

        return av_loss

    def train_meta(self, dataloader, epoch):
        total_params = sum(p.numel() for p in self.net.parameters())
        trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Total params: {total_params}")
        print(f"Trainable params: {trainable_params}")
        print(f"Frozen params: {total_params - trainable_params}")

        self.net.train()
        total_loss = 0.0
        batches = 0

        accum_steps = max(1, getattr(self, "grad_accumulation_steps", 1))
        if accum_steps > 1:
            if epoch % 50 == 0 and accum_steps > self.grad_accumulation_steps_min:
                new_acc = max(self.grad_accumulation_steps_min, accum_steps // 2)
                self.grad_accumulation_steps = new_acc
                accum_steps = new_acc
                print(f"Reducing gradient accumulation steps to {accum_steps}")

            print(f"Gradient accumulation enabled: {accum_steps} steps (effective batch size = {accum_steps * self.bs})")

        # zero grads at epoch start
        self.optimizer.zero_grad(set_to_none=True)

        epoch_start = datetime.datetime.now()
        for batch_idx, (data, target, landmarks, meta, id, orig_size, orig_img) in enumerate(dataloader):
            batches += 1

            # Move tensors to device (no Variables)
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            meta = meta.to(self.device, non_blocking=True)

            # ---------- Debug: print micro-batch info ----------
            if batch_idx % 10 == 0:
                print(f"\n--- batch_idx={batch_idx} ---")
                print(f"data.shape={tuple(data.shape)}, target.shape={tuple(target.shape)}")
                if self.use_amp:
                    try:
                        print("AMP enabled. GradScaler scale =", self.scaler.get_scale())
                    except Exception:
                        print("Could not read scaler scale")

            # ---------- Use autocast for mixed precision around forward+loss ----------
            with autocast(enabled=self.use_amp):
                pred = self.net(data, meta)
                # Debug: check forward outputs
                if batch_idx % 10 == 0:
                    try:
                        print(f"pred.shape={getattr(pred,'shape', 'unknown')}")
                        # If pred is tensor, print a tiny summary
                        if isinstance(pred, torch.Tensor):
                            print(f"pred min={float(pred.detach().min()):.6e}, max={float(pred.detach().max()):.6e}, mean={float(pred.detach().mean()):.6e}")
                    except Exception as e:
                        print("Error printing pred stats:", e)

                # compute any extra needed values
                if self.add_class_loss or self.add_alpha_loss or self.add_alphafhc_loss:
                    pred_alphas, pred_classes, target_alphas, target_classes = \
                        self.class_calculation.get_class_from_output(pred, target, self.pixel_size)
                if self.add_alphafhc_loss:
                    pred_fhc, target_fhc = self.fhc_calc.get_fhc_batches(pred, target, self.pixel_size)

                # compute loss (kept same branching as your original)
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

            # Make sure loss is a tensor and connected
            if not isinstance(loss, torch.Tensor):
                raise RuntimeError("Loss returned not a torch.Tensor — cannot call backward()")

            # Debug: loss checks
            if batch_idx % 10 == 0:
                try:
                    print(f"raw loss item = {loss.item():.6e}")
                except Exception:
                    print("Could not read loss.item()")

                if not torch.isfinite(loss).all():
                    print(">>>> Loss is not finite (NaN or Inf) at batch", batch_idx)
                    # Optional: raise or continue depending on whether you want to stop
                    # raise RuntimeError("Non-finite loss")

            total_loss += loss.item()

            # scale loss for accumulation then backward
            loss = loss / accum_steps

            # ---------- backward (scaled when using AMP) ----------
            if self.use_amp:
                # Debug: print scale before backward
                if batch_idx % 50 == 0:
                    try:
                        print("scaler.get_scale() before backward =", self.scaler.get_scale())
                    except Exception:
                        pass
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # ---------- DEBUG: print grad existence and stats after backward ----------
            if batch_idx % 10 == 0:
                grads_exist = [p.grad is not None for p in self.net.parameters() if p.requires_grad]
                any_grad = any(grads_exist)
                print(f"[batch {batch_idx}] any_grad after backward: {any_grad}")

                # compute a few grad stats safely (sample first few params)
                max_grad = None
                total_norm = 0.0
                found = False
                sample_count = 0
                for p in self.net.parameters():
                    if p.requires_grad and p.grad is not None:
                        try:
                            g = p.grad.detach()
                            g_abs_max = g.abs().max().item()
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
                    print(f"[batch {batch_idx}] grad sample max abs = {max_grad:.6e}, grad sample norm = {total_norm:.6e}")
                else:
                    print(f"[batch {batch_idx}] No gradients found on sampled params after backward")

            # ---------- optimizer step every accum_steps ----------
            if (batch_idx + 1) % accum_steps == 0:
                print(f"[batch {batch_idx}] Performing optimizer step (accum count reached).")
                if self.use_amp:
                    # Optional: unscale if you want to clip grads
                    try:
                        # debug: scaler scale before step
                        print("scaler.get_scale() before step =", self.scaler.get_scale())
                    except Exception:
                        pass
                    # If clipping is used: self.scaler.unscale_(self.optimizer) -> clip -> scaler.step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # zero grads for next accumulation block
                self.optimizer.zero_grad(set_to_none=True)
                print(f"optimizer.step() at batch {batch_idx}")

            # periodic logging (loss is already scaled)
            if batch_idx % 50 == 0:
                display_loss = (loss.item() * accum_steps)
                print(f"Train Epoch: {epoch} [{(batch_idx + 1) * len(data)}/{len(dataloader.dataset)} ({100. * (batch_idx + 1) / len(dataloader):.0f}%)]\tLoss: {display_loss:.6f}", flush=True)

            # cleanup
            del pred
            del loss
            if 'pred_alphas' in locals():
                del pred_alphas
            if 'pred_classes' in locals():
                del pred_classes
            if 'target_alphas' in locals():
                del target_alphas
            if 'target_classes' in locals():
                del target_classes
            if 'pred_fhc' in locals():
                del pred_fhc
            if 'target_fhc' in locals():
                del target_fhc

        # ---------- final step for leftover gradients (AMP-aware) ----------
        if (batch_idx + 1) % accum_steps != 0:
            print("Final leftover optimizer step (not aligned to accum_steps).")
            if self.use_amp:
                try:
                    print("scaler.get_scale() before final step =", self.scaler.get_scale())
                except Exception:
                    pass
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            print("optimizer.step() at final leftover batch")

        av_loss = total_loss / batches
        print(f"\nTraining Set Average loss: {av_loss:.4f}", flush=True)

        epoch_end = datetime.datetime.now()
        print('Time taken for epoch = ', epoch_end - epoch_start)

        return av_loss