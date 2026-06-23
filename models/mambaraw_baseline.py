"""Beyond-R2LCM baseline used in the MambaRaw experiments.

The implementation is provided by the installed R2LCM/CompressAI package.
Keeping this adapter avoids maintaining a modified copy of the baseline while
preserving the exact model entry used for comparison in the MambaRaw paper.
"""

from __future__ import annotations

from compressai.zoo import learned_context_journal4


def build_baseline(**kwargs):
    """Build the official baseline model used in the paper."""
    quality = int(kwargs.pop("quality", 1))
    metric = kwargs.pop("metric", "mse")
    pretrained = bool(kwargs.pop("pretrained", False))
    progress = bool(kwargs.pop("progress", True))
    return learned_context_journal4(
        quality=quality,
        metric=metric,
        pretrained=pretrained,
        progress=progress,
        **kwargs,
    )
