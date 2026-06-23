"""Spatial-energy coupled context modules proposed in MambaRaw.

This file implements TileMambaBlock and Energy-Aware Refinement (EAR).
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock2d(nn.Module):
    """Thin NCHW wrapper around the official MambaIC VSS block."""

    def __init__(
        self,
        channels: int,
        block_factory: Callable[..., nn.Module],
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if block_factory is None:
            raise RuntimeError(
                "MambaRaw requires the official MambaIC VSSBlock and its "
                "selective-scan dependencies; no identity fallback is allowed."
            )
        self.block = block_factory(hidden_dim=channels, drop_path=drop_path)
        self.no_srgb = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TileMambaBlock2d(nn.Module):
    """Apply VSS to the highest-energy tiles of each image independently.

    The score is the mean squared activation over valid (unpadded) pixels and
    channels.  Small feature maps use dense VSS, matching patch-based training
    in the paper. Large feature maps use per-image top-k tile selection, and
    each selected tile follows the MambaBlock(t_i) branch in Algorithm 1.
    """

    def __init__(
        self,
        channels: int,
        block_factory: Optional[Callable[..., nn.Module]] = None,
        tile_size: int = 64,
        keep_ratio: float = 0.5,
        drop_path: float = 0.0,
        inner: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")

        self.inner = inner or MambaBlock2d(channels, block_factory, drop_path)
        self.tile_size = int(tile_size)
        self.keep_ratio = float(keep_ratio)
        self.no_srgb = True

        self.last_tile_scores: Optional[torch.Tensor] = None
        self.last_selection_mask: Optional[torch.Tensor] = None

    def _partition(self, x: torch.Tensor):
        b, c, h, w = x.shape
        tile_h = math.ceil(h / self.tile_size)
        tile_w = math.ceil(w / self.tile_size)
        pad_h = tile_h * self.tile_size - h
        pad_w = tile_w * self.tile_size - w

        x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        valid = F.pad(
            x.new_ones((b, 1, h, w)),
            (0, pad_w, 0, pad_h),
        )

        def to_tiles(tensor: torch.Tensor) -> torch.Tensor:
            channels = tensor.shape[1]
            return (
                tensor.view(
                    b,
                    channels,
                    tile_h,
                    self.tile_size,
                    tile_w,
                    self.tile_size,
                )
                .permute(0, 2, 4, 1, 3, 5)
                .contiguous()
            )

        return to_tiles(x_pad), to_tiles(valid), (h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b, c, h, w = x.shape
        if (h <= self.tile_size and w <= self.tile_size) or self.keep_ratio >= 1.0:
            self.last_tile_scores = x.pow(2).mean(dim=(1, 2, 3), keepdim=False).view(b, 1)
            self.last_selection_mask = torch.ones(b, 1, dtype=torch.bool, device=x.device)
            return self.inner(x)

        tiles, valid_tiles, original_size = self._partition(x)
        tile_h, tile_w = tiles.shape[1:3]
        num_tiles = tile_h * tile_w

        energy_sum = (tiles.pow(2) * valid_tiles).sum(dim=(3, 4, 5))


        tile_area = self.tile_size * self.tile_size
        scores = energy_sum / (c * tile_area)
        scores = scores.view(b, num_tiles)

        k = max(1, min(num_tiles, int(math.floor(self.keep_ratio * num_tiles))))
        selected_indices = scores.topk(
            k=k, dim=1, largest=True, sorted=False
        ).indices
        selection_mask = torch.zeros(
            b, num_tiles, dtype=torch.bool, device=x.device
        )
        selection_mask.scatter_(1, selected_indices, True)

        flat_tiles = tiles.view(
            b, num_tiles, c, self.tile_size, self.tile_size
        )
        selected = flat_tiles[selection_mask]
        processed = self.inner(selected)
        output_tiles = flat_tiles.clone()
        output_tiles[selection_mask] = processed

        output = (
            output_tiles.view(
                b,
                tile_h,
                tile_w,
                c,
                self.tile_size,
                self.tile_size,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(b, c, tile_h * self.tile_size, tile_w * self.tile_size)
        )

        self.last_tile_scores = scores.detach()
        self.last_selection_mask = selection_mask.detach()
        return output[:, :, : original_size[0], : original_size[1]]


class EnergyAwareRefinement2d(nn.Module):
    """Identity-initialized local energy refinement from the MambaRaw paper."""

    def __init__(self, channels: int, hidden_channels: Optional[int] = None) -> None:
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.energy_gate = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        self.no_srgb = True


        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = x.pow(2).mean(dim=1, keepdim=True)
        gate = self.energy_gate(energy)
        return x + gate * self.residual(x)
