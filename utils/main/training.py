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
        if cfg.MODEL.DEVICE == 'cuda':
            torch.cuda.empty_cache()
        self.pixel_size = torch.tensor(cfg.DATASET.PIXEL_SIZE).to(cfg.MODEL.DEVICE)

        self.grad_accumulation_steps = cfg.TRAIN.GRAD_ACCUMULATION_STEPS
        
        
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
    
    def train_meta(self, dataloader, epoch):
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
            print(f"Gradient accumulation enabled: {accum_steps} steps (effective batch size = {accum_steps * self.bs})")

        # zero grads at epoch start
        self.optimizer.zero_grad()

        epoch_start = datetime.datetime.now()
        for batch_idx, (data, target, landmarks, meta, id, orig_size, orig_img) in enumerate(dataloader):
            batches += 1

            # Move tensors to device (no Variables)
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            meta = meta.to(self.device, non_blocking=True)

            # forward
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
            loss.backward()

            # Debug: verify gradient existence (optional)
            if batch_idx % 100 == 0:
                any_grad = any((p.grad is not None) for p in self.net.parameters() if p.requires_grad)
                print(f"[batch {batch_idx}] any_grad after backward: {any_grad}")

            # optimizer step every accum_steps
            if (batch_idx + 1) % accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(f"optimizer.step() at batch {batch_idx}")

            # periodic logging
            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} [{(batch_idx + 1) * len(data)}/{len(dataloader.dataset)} ({100. * (batch_idx + 1) / len(dataloader):.0f}%)]\tLoss: {loss.item() * accum_steps:.6f}", flush=True)

            # lightweight cleanup (let python free references)
            del pred

        # final step for leftover gradients
        if (batch_idx + 1) % accum_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            print("optimizer.step() at final leftover batch")

        av_loss = total_loss / batches
        print(f"\nTraining Set Average loss: {av_loss:.4f}", flush=True)

        epoch_end = datetime.datetime.now()
        print('Time taken for epoch = ', epoch_end - epoch_start)

        return av_loss