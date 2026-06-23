"""Strict MambaRaw checkpoint restoration."""

from __future__ import annotations

from typing import Any, Dict

import torch

from .mambaraw_registry import LOCAL_MODEL_BUILDERS, inspect_checkpoint


def checkpoint_model_kwargs(checkpoint: Dict[str, Any], info: Dict[str, Any]):
    if info["model"] == "mambaraw":
        return {
            "N": info.get("N", 192),
            "M": info.get("M", 320),
            "raw_channel": info.get("raw_channel", 3),
            "tile_size": info.get("tile_size", 64),
            "tile_keep_ratio": info.get("tile_keep_ratio", 0.5),
            "stride": info.get("stride", 2),
            "reduce_c": info.get("reduce_c", 8),
            "down_num": info.get("down_num", 2),
            "sampling_num": info.get("sampling_num", 4),
            "use_deconv": info.get("use_deconv", True),
            "rounding": info.get("rounding", "noise"),
            "rounding_aux": info.get("rounding_aux", "forward"),
            "adaptive_quant": False,
        }

    kwargs = {
        "N": info.get("N", 192),
        "stride": info.get("stride", 2),
        "reduce_c": info.get("reduce_c", 8),
        "down_num": info.get("down_num", 2),
        "sampling_num": info.get("sampling_num", 4),
        "use_deconv": info.get("use_deconv", True),
        "adaptive_quant": info.get("adaptive_quant", True),
        "rounding": info.get("rounding", "noise"),
        "rounding_aux": info.get("rounding_aux", "noise"),
    }
    saved_args = checkpoint.get("args")
    if saved_args is not None:
        for name in tuple(kwargs):
            if hasattr(saved_args, name):
                kwargs[name] = getattr(saved_args, name)
    return kwargs


def parse_checkpoint_spec(spec: str):
    """Parse MODEL:[LABEL=]PATH without inferring architecture from weights."""
    if ":" not in spec:
        raise ValueError(
            "Checkpoint must use MODEL:[LABEL=]PATH, where MODEL is "
            "'mambaraw' or 'baseline'"
        )
    model_name, value = spec.split(":", 1)
    if model_name not in LOCAL_MODEL_BUILDERS:
        raise ValueError(f"Unsupported checkpoint model: {model_name}")
    if "=" in value:
        label, path = value.split("=", 1)
    else:
        label, path = model_name, value
    return model_name, label, path


def load_checkpoint_model(
    path: str,
    model_name: str,
    device: str = "cpu",
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    info, state_dict = inspect_checkpoint(checkpoint, model_name=model_name)
    if (
        info["model"] == "mambaraw"
        and not info.get("paper_architecture", False)
    ):
        raise RuntimeError(
            "The checkpoint does not contain the Beyond-R2LCM Level-1 "
            "TileMamba/EAR entropy model."
        )
    if (
        info["model"] == "mambaraw"
        and info.get("implementation_version") != 4
    ):
        raise RuntimeError(
            "This checkpoint uses the legacy channel/JPEG-context layout and "
            "is incompatible with the manuscript-aligned implementation."
        )
    if info["model"] not in LOCAL_MODEL_BUILDERS:
        raise ValueError(f"Unsupported local model: {info['model']}")

    model = LOCAL_MODEL_BUILDERS[info["model"]](
        **checkpoint_model_kwargs(checkpoint, info)
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model, info, checkpoint


def checkpoint_lambda(checkpoint: Dict[str, Any], requested: float = None) -> float:
    if requested is not None:
        return float(requested)
    saved_args = checkpoint.get("args")
    return float(getattr(saved_args, "lmbda", 0.8))
