"""Lazy public exports for MambaRaw models."""

__all__ = [
    "MambaRaw",
    "build_baseline",
    "LOCAL_MODEL_BUILDERS",
    "register_local_models",
]


def __getattr__(name):
    if name == "MambaRaw":
        from .mambaraw import MambaRaw

        return MambaRaw
    if name == "build_baseline":
        from .mambaraw_baseline import build_baseline

        return build_baseline
    if name in {"LOCAL_MODEL_BUILDERS", "register_local_models"}:
        from .mambaraw_registry import (
            LOCAL_MODEL_BUILDERS,
            register_local_models,
        )

        return {
            "LOCAL_MODEL_BUILDERS": LOCAL_MODEL_BUILDERS,
            "register_local_models": register_local_models,
        }[name]
    raise AttributeError(name)
