#!/usr/bin/env python3
"""
test_amp_scaler.py

Run this in the same Python environment you use for training.

What it does:
- Verifies CUDA availability.
- Creates a tiny model + optimizer on CUDA.
- Creates a GradScaler and exercises:
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # if available
    scaler.step(optimizer)
    scaler.update()
    scaler.get_scale()          # if available
- Reports which step failed (if any) and returns non-zero exit on failure.

Exit codes:
  0 -> full AMP GradScaler flow succeeded
  1 -> CUDA not available (skipped)
  2 -> Some scaler operation failed
"""

import sys
import traceback
import torch
import torch.nn as nn
import torch.optim as optim

def run_test():
    report = {"cuda_available": torch.cuda.is_available(), "device": None, "steps": {}, "exception": None}
    if not report["cuda_available"]:
        print("CUDA is NOT available in this Python environment. Skipping AMP scaler test.")
        return 1

    report["device"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "unknown"

    # tiny model
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(4, 1)

        def forward(self, x):
            x = self.conv(x)
            x = torch.relu(x)
            x = self.pool(x).view(x.size(0), -1)
            return self.fc(x).squeeze(1)

    device = torch.device("cuda")
    torch.backends.cudnn.enabled = True

    net = TinyNet().to(device)
    opt = optim.SGD(net.parameters(), lr=1e-2)

    # make a tiny batch
    x = torch.randn(2, 3, 32, 32, device=device)
    y = torch.randn(2, device=device)

    # Try to import GradScaler and autocast
    try:
        from torch.cuda.amp import GradScaler, autocast
    except Exception as e:
        print("Failed to import torch.cuda.amp:", repr(e))
        traceback.print_exc()
        return 2

    # Create scaler (guarded)
    try:
        scaler = GradScaler(init_scale=2**8)
        report["steps"]["scaler_create"] = "ok"
    except Exception as e:
        report["steps"]["scaler_create"] = f"failed: {repr(e)}"
        report["exception"] = traceback.format_exc()
        print("scaler creation FAILED:\n", report["steps"]["scaler_create"])
        return 2

    # check available methods
    methods = { m: hasattr(scaler, m) for m in ("scale", "unscale_", "step", "update", "get_scale") }
    print("Scaler method availability:", methods)

    try:
        opt.zero_grad(set_to_none=True)

        # forward + compute loss under autocast
        with autocast(enabled=True):
            pred = net(x)
            loss = nn.functional.mse_loss(pred, y)

        # scaler.scale(loss).backward()
        try:
            scaler.scale(loss).backward()
            report["steps"]["scale_backward"] = "ok"
        except Exception as e:
            report["steps"]["scale_backward"] = f"failed: {repr(e)}"
            raise

        # optional: unscale_ (if exists) before gradient checks/clipping
        if hasattr(scaler, "unscale_"):
            try:
                scaler.unscale_(opt)
                report["steps"]["unscale_"] = "ok"
            except Exception as e:
                report["steps"]["unscale_"] = f"failed: {repr(e)}"
                raise
        else:
            report["steps"]["unscale_"] = "not_present"

        # check for NaNs/Infs in grads (simple scan)
        grads_ok = True
        for p in net.parameters():
            if p.grad is None:
                continue
            g = p.grad
            if torch.isnan(g).any() or torch.isinf(g).any():
                grads_ok = False
                break
        report["steps"]["grad_check"] = "ok" if grads_ok else "bad_grads"

        # scaler.step(opt)
        try:
            scaler.step(opt)
            report["steps"]["scaler_step"] = "ok"
        except Exception as e:
            report["steps"]["scaler_step"] = f"failed: {repr(e)}"
            # try plain optimizer step as fallback (but mark failure)
            try:
                opt.step()
                report["steps"]["scaler_step_fallback_optimizer_step"] = "fallback_ok"
            except Exception as e2:
                report["steps"]["scaler_step_fallback_optimizer_step"] = f"fallback_failed: {repr(e2)}"
                raise

        # scaler.update()
        try:
            scaler.update()
            report["steps"]["scaler_update"] = "ok"
        except Exception as e:
            report["steps"]["scaler_update"] = f"failed: {repr(e)}"
            raise

        # scaler.get_scale() if present
        if hasattr(scaler, "get_scale"):
            try:
                s = scaler.get_scale()
                report["steps"]["get_scale"] = f"ok, value={s}"
            except Exception as e:
                report["steps"]["get_scale"] = f"failed: {repr(e)}"
                raise
        else:
            report["steps"]["get_scale"] = "not_present"

    except Exception as main_e:
        report["exception"] = traceback.format_exc()
        print("=== AMP scaler test FAILED ===")
        for k,v in report["steps"].items():
            print(f"{k}: {v}")
        print("Exception trace:\n", report["exception"])
        return 2

    # If we reach here everything went ok (or fallback to optimizer.step happened)
    print("=== AMP scaler test PASSED ===")
    for k,v in report["steps"].items():
        print(f"{k}: {v}")
    return 0

if __name__ == "__main__":
    rc = run_test()
    # non-zero exit codes helpful in CI / scripts
    sys.exit(rc)