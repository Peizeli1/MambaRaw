"""VMamba VSS block used by TileMambaBlock."""

from functools import partial
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath

from .vmamba_ss2d import SS2D


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias: bool = True,
        ssm_drop_rate: float = 0,
        ssm_init: str = "v0",
        forward_type: str = "v2",
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type=forward_type,
        )
        self.drop_path = DropPath(drop_path)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.post_norm:
            return x + self.drop_path(self.norm(self.op(x)))
        return x + self.drop_path(self.op(self.norm(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x)
        x = x.permute(0, 2, 3, 1)
        return self._forward(x).permute(0, 3, 1, 2)
