"""MambaRaw model registration and checkpoint inspection."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, MutableMapping, Tuple

from .mambaraw import MambaRaw
from .mambaraw_baseline import build_baseline


def _strip_zoo_arguments(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = dict(kwargs)
    for name in ("quality", "metric", "pretrained", "progress"):
        kwargs.pop(name, None)
    return kwargs


def mambaraw(**kwargs):
    """MambaRaw architecture."""
    return MambaRaw(**_strip_zoo_arguments(kwargs))


LOCAL_MODEL_BUILDERS = {
    "baseline": build_baseline,
    "mambaraw": mambaraw,
}


def register_local_models(registry: MutableMapping[str, Any]) -> None:
    """Register MambaRaw in the CompressAI model registry."""
    registry.update(LOCAL_MODEL_BUILDERS)


def inspect_checkpoint(
    checkpoint: Dict[str, Any],
    model_name: str | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return model metadata and the contained state dictionary."""
    state_dict = checkpoint.get("state_dict", checkpoint)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = OrderedDict(
            (key[len("module."):], value) for key, value in state_dict.items()
        )
    keys = tuple(state_dict.keys())

    paper_architecture = any(
        key.startswith("entropy_parameters_list.0.tile_mamba.")
        for key in keys
    )
    has_tile = paper_architecture
    has_ear = any(
        key.startswith("entropy_parameters_list.0.ear.") for key in keys
    )

    n_channels = 192
    if "entropy_bottleneck._matrix0" in state_dict:
        n_channels = state_dict["entropy_bottleneck._matrix0"].shape[0]

    m_channels = 320

    args = checkpoint.get("args")
    if model_name is None:
        raise ValueError(
            "model_name must be specified explicitly as 'mambaraw' or 'baseline'"
        )
    if model_name not in LOCAL_MODEL_BUILDERS:
        raise ValueError(f"Unsupported local model: {model_name}")
    metadata = {
        "model": model_name,
        "N": n_channels,
        "M": m_channels,
        "has_tile_mamba": has_tile,
        "has_ear": has_ear,
        "paper_architecture": paper_architecture,
        "implementation_version": 4 if paper_architecture else 1,
    }
    metadata.update(checkpoint.get("model_config", {}))
    if args is not None:
        for name in (
            "stride",
            "reduce_c",
            "down_num",
            "sampling_num",
            "use_deconv",
            "adaptive_quant",
            "rounding",
            "rounding_aux",
            "tile_size",
            "tile_keep_ratio",
            "N",
            "M",
        ):
            if hasattr(args, name):
                metadata[name] = getattr(args, name)

    return metadata, state_dict
