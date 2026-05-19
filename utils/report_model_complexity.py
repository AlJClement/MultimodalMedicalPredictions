#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.support.default_config import get_cfg_defaults


DEFAULT_CFGS = [
    "ddh_0.01466_dpt_mm_channels",
    "ddh_0.01466_dpt",
    "ddh_0.01466_hrnet",
]


def load_cfg(cfg_name: str, device: str) -> Any:
    cfg_path = REPO_ROOT / "experiments" / f"{cfg_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(cfg_path))
    cfg.defrost()
    cfg.MODEL.DEVICE = device
    # Parameter count and FLOPs do not depend on pretrained weights.
    if hasattr(cfg.MODEL, "ENCODER_WEIGHTS"):
        cfg.MODEL.ENCODER_WEIGHTS = None
    cfg.freeze()
    return cfg


def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(cfg: Any) -> torch.nn.Module:
    model_name = str(cfg.MODEL.NAME).strip().lower()
    module_path = REPO_ROOT / "utils" / "main" / "models" / f"{model_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Model module not found for {model_name}: {module_path}")

    module = load_module_from_path(f"complexity_{model_name}", module_path)
    model_cls = getattr(module, model_name)

    if model_name == "hrnet":
        original_get_hrnet_backbone = model_cls.get_hrnet_backbone

        def _get_hrnet_backbone_without_pretrained(self, pretrained=True):
            return original_get_hrnet_backbone(self, pretrained=False)

        model_cls.get_hrnet_backbone = _get_hrnet_backbone_without_pretrained

    model = model_cls(cfg)
    model.eval()
    return model.to(cfg.MODEL.DEVICE)


def build_inputs(cfg: Any, batch_size: int, device: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    image_h, image_w = [int(x) for x in cfg.DATASET.CACHED_IMAGE_SIZE]
    in_channels = int(cfg.MODEL.IN_CHANNELS)
    image = torch.randn(batch_size, in_channels, image_h, image_w, device=device)

    meta_widths = list(getattr(cfg.MODEL, "META_FEATURES", []))
    meta_dim = int(sum(int(width) for width in meta_widths if int(width) > 0))
    meta = None
    if meta_dim > 0:
        meta = torch.randn(batch_size, meta_dim, device=device)

    return image, meta


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def _count_flops_with_flop_counter(
    model: torch.nn.Module, image: torch.Tensor, meta: Optional[torch.Tensor]
) -> Optional[int]:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None

    with torch.no_grad():
        with FlopCounterMode(display=False) as flop_counter:
            model(image, meta)
    return int(flop_counter.get_total_flops())


def _count_flops_with_profiler(
    model: torch.nn.Module, image: torch.Tensor, meta: Optional[torch.Tensor]
) -> Optional[int]:
    try:
        from torch.profiler import ProfilerActivity, profile
    except ImportError:
        return None

    activities = [ProfilerActivity.CPU]
    if image.is_cuda:
        activities.append(ProfilerActivity.CUDA)

    with torch.no_grad():
        with profile(activities=activities, record_shapes=False, with_flops=True) as prof:
            model(image, meta)

    total_flops = 0
    for event in prof.key_averages():
        event_flops = getattr(event, "flops", 0)
        if event_flops:
            total_flops += int(event_flops)

    return total_flops or None


def count_flops(model: torch.nn.Module, image: torch.Tensor, meta: Optional[torch.Tensor]) -> Tuple[Optional[int], str]:
    flops = _count_flops_with_flop_counter(model, image, meta)
    if flops is not None:
        return flops, "torch.utils.flop_counter"

    flops = _count_flops_with_profiler(model, image, meta)
    if flops is not None:
        return flops, "torch.profiler"

    return None, "unavailable"


def summarize_config(cfg_name: str, batch_size: int, device: str) -> Dict[str, Any]:
    cfg = load_cfg(cfg_name, device=device)
    model = build_model(cfg)
    image, meta = build_inputs(cfg, batch_size=batch_size, device=device)
    total_params, trainable_params = count_parameters(model)
    flops, flops_backend = count_flops(model, image, meta)

    return {
        "cfg": cfg_name,
        "model": str(cfg.MODEL.NAME),
        "device": device,
        "batch_size": batch_size,
        "input_shape": list(image.shape),
        "meta_shape": list(meta.shape) if meta is not None else None,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops": flops,
        "gflops": (flops / 1e9) if flops is not None else None,
        "flops_backend": flops_backend,
    }


def format_int(value: Optional[int]) -> str:
    if value is None:
        return "n/a"
    return f"{value:,}"


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def summarize_metric(values: List[float]) -> Dict[str, float]:
    count = len(values)
    mean = sum(values) / count
    if count > 1:
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        std = math.sqrt(variance)
        margin = 1.96 * (std / math.sqrt(count))
    else:
        std = 0.0
        margin = 0.0

    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
    }


def format_summary_line(label: str, values: List[float]) -> str:
    summary = summarize_metric(values)
    return f"  {label}: mean={summary['mean']:.2f}, std={summary['std']:.2f}"


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Report parameter counts and GFLOPs for experiment configs.")
    parser.add_argument("--cfgs", nargs="+", default=DEFAULT_CFGS, help="Config names inside experiments/ without .yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size used for FLOP counting")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used to instantiate the model and run the synthetic forward pass",
    )
    parser.add_argument(
        "--summary-style",
        action="store_true",
        help="Print each config using mean/std/95%% CI summary lines like benchmark output",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table")
    args = parser.parse_args()

    device = resolve_device(args.device)
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for cfg_name in args.cfgs:
        try:
            results.append(summarize_config(cfg_name, batch_size=args.batch_size, device=device))
        except Exception as exc:
            failures.append({"cfg": cfg_name, "error": f"{type(exc).__name__}: {exc}"})

    if args.json:
        print(json.dumps({"results": results, "failures": failures}, indent=2))
    elif args.summary_style:
        for row in results:
            print(f"{row['cfg']}:")
            print(format_summary_line("Parameters Total", [float(row["total_params"])]))
            print(format_summary_line("Parameters Trainable", [float(row["trainable_params"])]))
            if row["flops"] is None:
                print("  Forward FLOPs: mean=n/a, std=n/a")
            else:
                print(format_summary_line("Forward FLOPs", [float(row["flops"])]))
            print()

        if failures:
            print("Failures:")
            for failure in failures:
                print(f"- {failure['cfg']}: {failure['error']}")
    else:
        header = f"{'cfg':34} {'model':10} {'params':>15} {'trainable':>15} {'GFLOPs':>10} {'device':>8}"
        print(header)
        print("-" * len(header))
        for row in results:
            print(
                f"{row['cfg'][:34]:34} "
                f"{row['model'][:10]:10} "
                f"{format_int(row['total_params']):>15} "
                f"{format_int(row['trainable_params']):>15} "
                f"{format_float(row['gflops']):>10} "
                f"{row['device']:>8}"
            )

        if failures:
            print("\nFailures:")
            for failure in failures:
                print(f"- {failure['cfg']}: {failure['error']}")

        if results:
            print(f"\nFLOPs are counted for a single forward pass with batch_size={args.batch_size}.")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
