# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import math
import random
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from compressai.datasets.raw_image import RawImageDataset
from compressai.datasets.dataset_raw import DatasetRAW
from compressai.datasets.lmdb_dataset import LMDBDataset

from compressai.zoo import image_models
from compressai.datasets import random_crop, center_crop
import torch.cuda.amp as amp
import os
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

try:
    from models.mambaraw_registry import (
        inspect_checkpoint,
        register_local_models,
    )

    register_local_models(image_models)
except Exception as exc:
    print(f"[WARN] local RAW models not registered: {exc}")

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, beta=1, **kwargs):
        super().__init__()
        self.lmbda = lmbda
        self.beta = beta

        self.noquant = kwargs.get("noquant", False)
        self.l1 = kwargs.get("l1", False)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target, eval=False, lam=None):
        N, C, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        rate_terms = {
            name: (
                torch.log(likelihoods) / (-math.log(2) * num_pixels)
            ).sum()
            for name, likelihoods in output["likelihoods"].items()
        }
        out["bpp_loss"] = sum(rate_terms.values())
        for name, value in rate_terms.items():
            out[f"bpp_{name}"] = value

        x_hat = output["x_hat"]

        output = x_hat
        out["mse_loss"] = self.mse_loss(output, target)
        out["l1_loss"] = self.l1_loss(output, target)
        def calc_psnr(img1, img2):
            return torch.mean(20*torch.log10(torch.rsqrt(torch.mean((img1 - img2) ** 2, dim=[1, 2, 3]))))
        out["PSNR"] =  calc_psnr(output, target)
        if lam is None:
            lam = self.lmbda
        if self.l1:
            out["loss"] = (lam * 255**2 * out["l1_loss"]) + \
                (self.beta * out["bpp_loss"] if not self.noquant else 0)
        else:
            out["loss"] = (lam * 255**2 * out["mse_loss"]) + \
                (self.beta * out["bpp_loss"] if not self.noquant else 0)
        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers

    Supports an optional learning rate for TileMamba and EAR when fine-tuning.
    """

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    new_module_types = {"TileMambaBlock2d", "EnergyAwareRefinement2d"}
    new_parameter_ids = {
        id(parameter)
        for module in net.modules()
        if type(module).__name__ in new_module_types
        for parameter in module.parameters()
    }
    old_params = []
    new_params = []

    for n in sorted(parameters):
        if id(params_dict[n]) in new_parameter_ids:
            new_params.append(params_dict[n])
        else:
            old_params.append(params_dict[n])

    if hasattr(args, 'new_module_lr') and args.new_module_lr is not None:
        new_module_lr = args.new_module_lr
        old_module_lr = args.learning_rate
        print(f"[INFO] Using differential learning rates:")
        print(f"  Old modules: {old_module_lr:.2e} ({len(old_params)} params)")
        print(f"  New modules: {new_module_lr:.2e} ({len(new_params)} params)")

        parameter_groups = [{"params": old_params, "lr": old_module_lr}]
        if new_params:
            parameter_groups.append({"params": new_params, "lr": new_module_lr})

        if args.optimizer == "adam":
            optimizer = optim.Adam(
                parameter_groups,
                betas=(0.9, 0.999),
                eps=args.eps,
            )
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(parameter_groups, momentum=0.9)
    else:
        if args.optimizer == "adam":
            optimizer = optim.Adam(
                (params_dict[n] for n in sorted(parameters)),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=args.eps,
            )
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(
                (params_dict[n] for n in sorted(parameters)),
                lr=args.learning_rate,
                momentum = 0.9
            )

    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    ) if args.optimizer == "adam" else optim.SGD(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
        weight_decay=1e-4,
        momentum=0.9
    )

    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, scaler, aux_scaler, logger: SummaryWriter, args
):
    model.train()
    device = next(model.parameters()).device
    print_freq = args.print_freq
    accu_train_time = 0
    for i, data in enumerate(train_dataloader):
        rgb_jpeg, raw_visible, metadata = data
        rgb_jpeg, raw_visible = rgb_jpeg.to(device), raw_visible.to(device)
        if args.model == "mambaraw":
            if random.random() < 0.5:
                rgb_jpeg = torch.flip(rgb_jpeg, dims=(-1,))
                raw_visible = torch.flip(raw_visible, dims=(-1,))
            if random.random() < 0.5:
                rgb_jpeg = torch.flip(rgb_jpeg, dims=(-2,))
                raw_visible = torch.flip(raw_visible, dims=(-2,))
            rotation = random.randint(0, 3)
            if rotation:
                rgb_jpeg = torch.rot90(
                    rgb_jpeg, rotation, dims=(-2, -1)
                )
                raw_visible = torch.rot90(
                    raw_visible, rotation, dims=(-2, -1)
                )
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        current_time = time.time()
        if args.lambda_list is not None:
            lam = random.sample(args.lambda_list, 1)[0]
        else:
            lam = None
        with amp.autocast(enabled=not args.no_amp):
            out_net = model(raw_visible, rgb_jpeg, lam=lam)
            out_criterion = criterion(out_net, raw_visible, lam=lam)
            total_loss = out_criterion["loss"]
            if args.dual:
                out_net_soft = model(raw_visible, rgb_jpeg, lam=lam, soft=True)
                out_criterion_soft = criterion(out_net_soft, raw_visible, lam=lam)
                total_loss += out_criterion_soft["loss"]
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad.zero_()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        scaler.step(optimizer)
        scaler.update()
        aux_loss = model.aux_loss()
        aux_scaler.scale(aux_loss * args.wAux).backward()
        aux_scaler.step(aux_optimizer)
        aux_scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        accu_train_time += (time.time() - current_time)
        if i % print_freq == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(rgb_jpeg)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f' Loss: {out_criterion["loss"].item():.4f} |'
                f' MSE: {out_criterion["mse_loss"].item():.3e} |'
                f' L1: {out_criterion["l1_loss"].item():.3e} |'
                f' L_Bpp: {out_criterion["bpp_loss"].item():.6f} |'
                f" L_Aux: {aux_loss.item():.2f}"
                f' trian_iter: {accu_train_time/print_freq:.6f}'
            )
            if args.dual:
                print(
                    f' \t\t Aux Dual net: '
                    f' Loss: {out_criterion_soft["loss"].item():.4f} |'
                    f' MSE: {out_criterion_soft["mse_loss"].item():.3e} |'
                    f' L1: {out_criterion_soft["l1_loss"].item():.3e} |'
                    f' L_Bpp: {out_criterion_soft["bpp_loss"].item():.6f} |'
                )
            accu_train_time = 0
            logger.add_scalar("Loss", out_criterion["loss"].item(
            ), epoch * len(train_dataloader) + i)
            logger.add_scalar("bpp", out_criterion["bpp_loss"].item(
            ), epoch * len(train_dataloader) + i)
            logger.add_scalar("MSE", out_criterion["mse_loss"].item(
            ), epoch * len(train_dataloader) + i)
            logger.add_scalar("L1", out_criterion["l1_loss"].item(
            ), epoch * len(train_dataloader) + i)

def test_epoch(epoch, test_dataloader, model, criterion, args):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    l1_loss = AverageMeter()
    psnr = AverageMeter()

    with torch.no_grad():
        for rgb_jpeg, raw_visible, metadata in test_dataloader:
            raw_visible, rgb_jpeg = raw_visible.to(device), rgb_jpeg.to(device)
            with amp.autocast(enabled=not args.no_amp):
                out_net = model(raw_visible, rgb_jpeg)
            out_criterion = criterion(out_net, raw_visible, eval=True)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            l1_loss.update(out_criterion["l1_loss"])
            psnr.update(out_criterion["PSNR"])
    print(
        f"Test epoch {epoch}: Average losses:"
        f" Loss: {loss.avg:.4f} |"
        f" MSE loss: {mse_loss.avg:.3e} |"
        f" L1 loss: {l1_loss.avg:.3e} |"
        f" Bpp loss: {bpp_loss.avg:.3e} |"
        f" PSNR: {psnr.avg:.3e} |"
        f" Aux loss: {aux_loss.avg:.2f}\n"
    )
    if "resid_bpp_loss" in out_criterion.keys():
        f' L_Bpp_R: {out_criterion["resid_bpp_loss"].item():.3e} |'
    return loss.avg


def save_checkpoint(state, baseame="checkpoint"):
    """Save a single checkpoint file baseame.pth.tar"""
    torch.save(state, baseame + ".pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "--adaptive_quant",
        action="store_true"
    )
    parser.add_argument(
        "--freeze_delta",
        action="store_true"
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Dual net for training. One is discrete values (hard gumebl softmax and forward rounding), another is soft (soft gumbel softmax and noise rounding)"
    )
    parser.add_argument(
        "--drop_pixel",
        action="store_true"
    )
    parser.add_argument(
        "--soft_gumbel",
        action="store_true"
    )
    parser.add_argument(
        "--gmm_num",
        default=None,
        type=int
    )
    parser.add_argument(
        "--norm",
        type=str,
        default=None,
        choices=["layer", "half_ins"]
    )
    parser.add_argument(
        "--act",
        type=str,
        default="GDN",
        choices=["lrelu", "GDN"]
    )

    parser.add_argument(
        "--lmdb_dataset",
        action="store_true",
        help="Use the generated lmdb dataset. Need to specficy the file which includes the raw and srgb lmdb path by specifying --train. "
    )
    parser.add_argument(
        "--ablation_no_raw",
        action="store_true"
    )

    parser.add_argument(
        "--rounding",
        type=str,
        choices=["noise", "forward", "gumbel", "none"],
        default="noise",
        help="The policy of rounding the continuous feature to discrete ones"
    )
    parser.add_argument(
        "--rounding_aux",
        type=str,
        choices=["noise", "forward", "gumbel", "none"],
        default="noise",
        help="The policy of rounding the continuous feature to discrete ones"
    )
    parser.add_argument(
        "--raw_space",
        type=str,
        choices=["raw_linear", "xyz_gamma", "cvpr2022", "raw_gamma", "sid"],
        default="raw_linear",
    )

    parser.add_argument(
        "-q",
        "--quality",
        default=3,
        type=int,
        help="The quality of compressed images, the larger the higher"
    )
    parser.add_argument(
        "--z-scale",
        default=1,
        type=int,
        help="The quality of compressed images, the larger the higher"
    )
    parser.add_argument(
        "--use-deconv",
        action="store_true"
    )
    parser.add_argument(
        "--qlist",
        default=[90],
        nargs='+', type=int,
        help="The quality of compressed JPEG images, the larger the higher"
    )
    parser.add_argument(
        "--info",
        default="",
        type=str,
        help="The additional information to the basename"
    )
    parser.add_argument(
        "--train",
        default="train",
        type=str,
        help="The basename of the train split file"
    )
    parser.add_argument(
        "--val",
        default="val",
        type=str,
        help="The basename of the val split file"
    )
    parser.add_argument(
        "--print_freq",
        default=50,
        type=int
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1
    )
    parser.add_argument(
        "--test_freq",
        default=1,
        type=int
    )
    parser.add_argument(
        "--patience",
        default=10,
        type=float
    )
    parser.add_argument(
        "--resid-path",
        action="store_true",
        help=""
    )
    parser.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Weighted loss to highlight the loss of the area with high brightness"
    )
    parser.add_argument(
        "--l1",
        action="store_true",
        help="Wheteher to use L1 loss instead of L2 loss"
    )
    parser.add_argument(
        "--relu",
        action="store_true",
        help="Wheteher to use Relu instead of GDN"
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
    )
    parser.add_argument(
        "--hyper-mean",
        action="store_true",
    )
    parser.add_argument(
        "--channel-mean",
        help="Encoder the channel mean and variance",
        action="store_true",
    )
    parser.add_argument(
        "--resid",
        action="store_true",
    )
    parser.add_argument(
        "--xyz",
        action="store_true",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
    )
    parser.add_argument(
        "--reduce-c",
        default=1,
        type=int,
        help="reduce the number of latent channels using a factor (default: %(default)s)"
    )
    parser.add_argument(
        "--noquant",
        action="store_true",
    )
    parser.add_argument(
        "--inmem",
        action="store_true"
    )
    parser.add_argument(
        "--stride",
        default=2,
        type=int
    )
    parser.add_argument(
        "--nocompress",
        action="store_true"
    )
    parser.add_argument(
        "--demosaic",
        action="store_true",
    )
    parser.add_argument(
        "--context-z",
        action="store_true",
    )
    parser.add_argument(
        "--down_num",
        default="4",
        type=int
    )
    parser.add_argument(
        "-m",
        "--model",
        default="raw_hyperprior",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=False, help="Training dataset", default="./datasets/SID"
    )
    parser.add_argument(
        "--sampling-num", type=int, default=2, help="The number of sampling for context model"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=1000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "sgd"],
        type=str
    )
    parser.add_argument(
        "--eps",
        default=1e-6,
        type=float
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--beta",
        default=1,
        type=float,
        help="The weight of the bpp loss (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s), the higher the better image quality.",
    )
    parser.add_argument(
        "--lambda_list",
        default=None,
        nargs='+',
        type=float,
        help="List of lambdas for variable bit-rate training (variable BPP)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--new-module-lr",
        default=None,
        type=float,
        help="Learning rate for TileMamba/EAR when fine-tuning. "
             "If None, use same lr as old modules. Recommended: 1e-4 for fine-tuning.",
    )
    parser.add_argument(
        "--tile-size",
        default=64,
        type=int,
        help="Tile size on the Level-1 latent feature map (default: 64).",
    )
    parser.add_argument(
        "--tile-keep-ratio",
        default=0.5,
        type=float,
        help="Per-image fraction of Level-1 tiles processed by VSS (default: 0.5).",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=192,
        help="Backbone and hyper-latent channels for MambaRaw.",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=320,
        help="Main latent channels for MambaRaw.",
    )
    parser.add_argument(
        "--wAux",
        default=1,
        type=float,
        help="Auxiliary loss weight (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda", default=True)
    parser.add_argument("--debug_nan", action="store_true", help="")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="load the dataset to memory"
    )
    parser.add_argument(
        "--file_type",
        type=str,
        help="load the dataset to memory",
        default="tif"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument(
        "--save-root",
        type=str,
        default="./checkpoints",
        help="Root directory to save checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (overrides auto-generated name)"
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    if args.model == "mambaraw":
        if "--reduce-c" not in argv:
            args.reduce_c = 8
        if "--down_num" not in argv:
            args.down_num = 2
        if "--sampling-num" not in argv:
            args.sampling_num = 4
        args.use_deconv = True
        if "--rounding_aux" not in argv:
            args.rounding_aux = "forward"
        if args.lambda_list is not None:
            raise ValueError(
                "The paper configuration trains one MambaRaw checkpoint per "
                "lambda. Remove --lambda_list and use --lambda instead."
            )
        if args.dual:
            raise ValueError(
                "The paper MambaRaw configuration does not use dual/soft "
                "variable-rate training."
            )
        if not args.l1:
            args.l1 = True
            print("[INFO] MambaRaw uses the paper L1 distortion objective.")
    print(args)
    level_one_downsample = args.stride ** min(args.down_num, 4)
    if (
        args.model == "mambaraw"
        and args.patch_size <= level_one_downsample * args.tile_size
    ):
        warnings.warn(
            "The Level-1 feature map will contain at most one tile at "
            f"patch_size={args.patch_size} (downsample="
            f"{level_one_downsample}, tile_size={args.tile_size}); "
            "TileMambaBlock will run in dense fallback during training, "
            "as described in the paper's training protocol. Increase "
            "patch_size beyond the Level-1 downsample factor times "
            "tile_size to also train the sparse tile selector.",
            RuntimeWarning,
            stacklevel=2,
        )
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    def train_transforms(x): return random_crop(x, args.patch_size)
    def test_transforms(x): return center_crop(x, args.patch_size)
    if args.lmdb_dataset:
        with open(os.path.join(args.dataset, args.train+".txt"), "r") as f:
            db_path_raw, db_path_srgb = f.readlines()
        train_dataset = LMDBDataset(db_path_raw=db_path_raw.rstrip(), db_path_srgb=db_path_srgb.rstrip(), nocompress=args.nocompress, qlist=args.qlist, repeat=args.repeat)
    else:
        if args.cache:
            train_dataset = DatasetRAW(os.path.join(args.dataset, "train"), args.batch_size, args.patch_size, args.patch_size//2, to_gpu=False, ftype=args.file_type, debug=args.debug_nan)
        else:
            train_dataset = RawImageDataset(
                args.dataset, split=args.train, transform=train_transforms, **vars(args))

    if args.cache:
        test_dataset = DatasetRAW(os.path.join(args.dataset, "test"), args.test_batch_size, 512, 512, to_gpu=False, ftype=args.file_type)
    else:
        test_dataset = RawImageDataset(
            args.dataset, split=args.val, transform=test_transforms, **vars(args))

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    if args.model == "mambaraw":
        model_kwargs = dict(
            N=args.N,
            M=args.M,
            tile_size=args.tile_size,
            tile_keep_ratio=args.tile_keep_ratio,
            stride=args.stride,
            reduce_c=args.reduce_c,
            down_num=args.down_num,
            sampling_num=args.sampling_num,
            use_deconv=args.use_deconv,
            adaptive_quant=False,
            rounding=args.rounding,
            rounding_aux=args.rounding_aux,
        )
        print(
            "[INFO] MambaRaw: "
            f"TileMamba -> EAR, T={args.tile_size}, "
            f"rho={args.tile_keep_ratio}"
        )
    else:
        model_kwargs = dict(
            quality=args.quality,
            stride=args.stride,
            demosaic=args.demosaic,
            noquant=args.noquant,
            reduce_c=args.reduce_c,
            hyper_mean=args.hyper_mean,
            resid=args.resid,
            relu=args.relu,
            context_z=args.context_z,
            down_num=args.down_num,
            channel_mean=args.channel_mean,
            resid_path=args.resid_path,
            sampling_num=args.sampling_num,
            use_deconv=args.use_deconv,
            ablation_no_raw=args.ablation_no_raw,
            z_scale=args.z_scale,
            rounding=args.rounding,
            adaptive_quant=args.adaptive_quant,
            lambda_list=args.lambda_list,
            norm=args.norm,
            act=args.act,
            gmm_num=args.gmm_num,
            drop_pixel=args.drop_pixel,
            soft_gumbel=args.soft_gumbel,
            rounding_aux=args.rounding_aux,
        )

    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            checkpoint_temp = torch.load(args.checkpoint, map_location='cpu')
            checkpoint_info, _ = inspect_checkpoint(
                checkpoint_temp, model_name=args.model
            )
            if (
                args.model == "mambaraw"
                and checkpoint_info.get("implementation_version") != 4
            ):
                raise RuntimeError(
                    "Legacy MambaRaw checkpoints are incompatible with the "
                    "manuscript-aligned implementation."
                )
            if args.model == "mambaraw":
                model_kwargs.update(
                    N=checkpoint_info["N"],
                    M=checkpoint_info["M"],
                    tile_size=checkpoint_info.get(
                        "tile_size", args.tile_size
                    ),
                    tile_keep_ratio=checkpoint_info.get(
                        "tile_keep_ratio", args.tile_keep_ratio
                    ),
                )
            else:
                model_kwargs["N"] = checkpoint_info["N"]
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[WARN] Could not infer N from checkpoint: {e}")

    net = image_models[args.model](**model_kwargs)
    net = net.to(device)

    def nan_hook(self, inp, output):
        if isinstance(output, dict):
            return None
        if not isinstance(output, tuple):
            outputs = [output]
        else:
            outputs = output

        for i, out in enumerate(outputs):
            nan_mask = torch.isnan(out)
            if nan_mask.any():
                print("In", self.__class__.__name__)
                raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
    if args.debug_nan:
        for submodule in net.modules():
            submodule.register_forward_hook(nan_hook)

    basename = "%s%s_qual_%d_lamd_%.2e_stride_%d_%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s_%s_%s%s%s%s%s%s%s%s%s%s%s%s%s" % (args.model, "_resd" if args.resid else "", args.quality, args.lmbda, args.stride, f"down{args.down_num}", f"_{args.raw_space}", "_hyperMean" if args.hyper_mean else "", "_l1" if args.l1 else "", "_relu" if args.relu else "", "_context-z" if args.context_z else "", "_sample%d" % args.sampling_num if args.sampling_num != 2 else "", "_reduce-c%d" % args.reduce_c if args.reduce_c != 1 else "", "_%s" % os.path.basename(args.dataset), "_xyz" if args.xyz else "", "_nocompress" if args.nocompress else "", "_channelMean" if args.channel_mean else "", "" if args.patch_size == 512 else "_p%d" % args.patch_size, "" if not args.weighted_loss else "_weightedLoss", "_use_deconv" if args.use_deconv else "", args.info, args.train, args.rounding, "" if not args.adaptive_quant else "_adaptQuant", f"_bs{args.batch_size}" if args.batch_size!=1 else "", "_varaibleBPP" if args.lambda_list is not None else "", f"_{args.norm}" if args.norm else "", f"_{args.act}" if args.act!="GDN" else "", f"_ntrain{args.ntrain}" if args.ntrain > 0 else "", f"_{args.optimizer}" if args.optimizer!="adam" else "", f"_eps{args.eps:.1e}" if args.eps!=1e-6 else "", f"_GMM{args.gmm_num}" if args.gmm_num is not None else "", "_dropPixel" if args.drop_pixel else "", "_Dual" if args.dual else "", "_SoftGumbel" if args.soft_gumbel else "")

    logger = SummaryWriter(log_dir=f"experiments/{basename}_{int(time.time())}")
    last_epoch = 0
    if args.checkpoint:
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        checkpoint_info, state_dict = inspect_checkpoint(
            checkpoint, model_name=args.model
        )
        source_model = checkpoint_info["model"]
        if source_model == args.model:
            net.load_state_dict(state_dict, strict=True)
        else:
            raise RuntimeError(
                f"Cannot initialize '{args.model}' from '{source_model}' checkpoint"
            )
        if "epoch" in checkpoint:
            print(f"Loaded weights from epoch {checkpoint['epoch']}, "
                  f"starting fine-tuning from epoch 0 with lr={args.learning_rate}")

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    scaler = amp.GradScaler(enabled=not args.no_amp)
    aux_scaler = amp.GradScaler(enabled=not args.no_amp)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    criterion = RateDistortionLoss(**vars(args))

    best_loss = float("inf")
    save_dir = args.save_dir if args.save_dir else args.save_root
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Checkpoints will be saved to: {save_dir}")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            scaler,
            aux_scaler,
            logger,
            args
        )
        if epoch % args.test_freq == 0:
            torch.cuda.empty_cache()
            loss = test_epoch(epoch, test_dataloader, net, criterion, args)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                state = {
                    "epoch": epoch,
                    "state_dict": (
                        net.module.state_dict()
                        if isinstance(net, CustomDataParallel)
                        else net.state_dict()
                    ),
                    "model_name": args.model,
                    "model_config": {
                        "N": model_kwargs.get("N", 192),
                        "M": model_kwargs.get("M", 320),
                        "raw_channel": 3,
                        "stride": args.stride,
                        "reduce_c": args.reduce_c,
                        "down_num": args.down_num,
                        "sampling_num": args.sampling_num,
                        "use_deconv": args.use_deconv,
                        "adaptive_quant": (
                            False
                            if args.model == "mambaraw"
                            else args.adaptive_quant
                        ),
                        "rounding": args.rounding,
                        "rounding_aux": args.rounding_aux,
                        "tile_size": args.tile_size,
                        "tile_keep_ratio": args.tile_keep_ratio,
                        "paper_architecture": args.model == "mambaraw",
                        "implementation_version": 4,
                    },
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args": args,
                }
                lambda_tag = f"lamda{args.lmbda}".replace(".", "_")
                latest_base = os.path.join(save_dir, f"{lambda_tag}_latest")
                best_base = os.path.join(save_dir, f"{lambda_tag}_best")

                save_checkpoint(state, baseame=latest_base)
                if is_best:
                    save_checkpoint(state, baseame=best_base)
        lr_scheduler.step()


if __name__ == "__main__":
    main(sys.argv[1:])
