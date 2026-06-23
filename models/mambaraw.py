"""MambaRaw built directly on the Beyond-R2LCM codec."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models.learned_raw_journal4 import RawLearnedJournal4

from .spatial_energy_context import (
    EnergyAwareRefinement2d,
    TileMambaBlock2d,
)
from .vmamba import VSSBlock


def resize_jpeg(x_jpg: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        x_jpg,
        size=feature.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


class SpatialEnergyEntropyParameters(nn.Module):
    """R2LCM entropy head with TileMambaBlock and EAR."""

    def __init__(
        self,
        input_channels: int,
        context_channels: int,
        output_channels: int,
        tile_size: int,
        keep_ratio: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels + 3, context_channels, 1),
            nn.LeakyReLU(0.1, inplace=False),
        )
        self.tile_mamba = TileMambaBlock2d(
            channels=context_channels,
            block_factory=VSSBlock,
            tile_size=tile_size,
            keep_ratio=keep_ratio,
            drop_path=0.0,
        )
        self.ear = EnergyAwareRefinement2d(context_channels)
        self.output_projection = nn.Sequential(
            nn.Conv2d(context_channels + 3, context_channels, 1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(context_channels + 3, output_channels, 1),
        )
        self.no_srgb = True

    def forward(
        self,
        feature: torch.Tensor,
        x_jpg: torch.Tensor,
        lam_embedding=None,
    ) -> torch.Tensor:
        del lam_embedding
        jpeg = resize_jpeg(x_jpg, feature)
        context = self.input_projection(torch.cat([feature, jpeg], dim=1))
        context = self.tile_mamba(context)
        context = self.ear(context)
        for layer in self.output_projection:
            if isinstance(layer, nn.Conv2d):
                jpeg = resize_jpeg(x_jpg, context)
                context = layer(torch.cat([context, jpeg], dim=1))
            else:
                context = layer(context)
        return context


class MambaRaw(RawLearnedJournal4):
    """Beyond-R2LCM with a spatial-energy Level-1 entropy model."""

    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        raw_channel: int = 3,
        tile_size: int = 64,
        tile_keep_ratio: float = 0.5,
        **kwargs,
    ) -> None:
        if kwargs.get("lambda_list") is not None:
            raise ValueError(
                "MambaRaw trains one independent model per lambda."
            )
        if kwargs.get("gmm_num") is not None:
            raise ValueError("MambaRaw uses an independent single Gaussian.")
        if kwargs.get("adaptive_quant", False):
            raise ValueError(
                "MambaRaw predicts Gaussian mean and scale only; "
                "adaptive quantization must be disabled."
            )

        kwargs["raw_channel"] = raw_channel
        kwargs["lambda_list"] = None
        kwargs["gmm_num"] = None
        kwargs["adaptive_quant"] = False
        super().__init__(N=N, **kwargs)

        self.N = int(N)
        self.M = int(M)
        self.raw_channel = int(raw_channel)
        self.tile_size = int(tile_size)
        self.tile_keep_ratio = float(tile_keep_ratio)
        self.lambda_list = None

        latent_channels = self.N // self.reduce_c
        input_channels = 4 * latent_channels + 1
        output_channels = 2 * latent_channels
        context_channels = 2 * self.N
        level_one_entropy = SpatialEnergyEntropyParameters(
            input_channels=input_channels,
            context_channels=context_channels,
            output_channels=output_channels,
            tile_size=self.tile_size,
            keep_ratio=self.tile_keep_ratio,
        )
        self.entropy_parameters_list[0] = level_one_entropy
