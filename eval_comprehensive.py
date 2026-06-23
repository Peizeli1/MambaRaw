#!/usr/bin/env python3
"""Real-dataset evaluation for MambaRaw checkpoints."""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from compressai.datasets import RawImageDataset
from skimage.metrics import structural_similarity
from tqdm import tqdm

from models.mambaraw_checkpoint import (
    checkpoint_lambda,
    load_checkpoint_model,
    parse_checkpoint_spec,
)


def pad_to_multiple(x, multiple=128):
    height, width = x.shape[-2:]
    return F.pad(
        x,
        (0, (-width) % multiple, 0, (-height) % multiple),
        mode="reflect",
    )


def byte_length(value):
    if isinstance(value, (bytes, str)):
        return len(value)
    return sum(byte_length(item) for item in value)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, dataset, device, lmbda, limit, warmup):
    model.update(force=True)
    variable_rate = model.lambda_list is not None
    codec_lambda = lmbda if variable_rate else None
    image_rows = []
    count = min(len(dataset), limit) if limit > 0 else len(dataset)

    for index in tqdm(range(count)):
        x_jpg, target, _ = dataset[index]
        height, width = target.shape[-2:]
        x_jpg = pad_to_multiple(x_jpg.unsqueeze(0).to(device))
        target_pad = pad_to_multiple(target.unsqueeze(0).to(device))

        if index == 0:
            for _ in range(warmup):
                warmup_result = model.compress(
                    target_pad, x_jpg, lmbda=codec_lambda
                )
                model.decompress(
                    warmup_result["strings"],
                    warmup_result["shape"],
                    x_jpg,
                    compressed_result=warmup_result,
                    lmbda=codec_lambda,
                )
            sync(device)

        sync(device)
        start = time.perf_counter()
        compressed = model.compress(target_pad, x_jpg, lmbda=codec_lambda)
        sync(device)
        encode_ms = (time.perf_counter() - start) * 1000.0

        sync(device)
        start = time.perf_counter()
        reconstruction = model.decompress(
            compressed["strings"], compressed["shape"], x_jpg,
            compressed_result=compressed, lmbda=codec_lambda,
        )["x_hat"]
        sync(device)
        decode_ms = (time.perf_counter() - start) * 1000.0

        reconstruction = reconstruction[..., :height, :width].float().clamp(0, 1)
        target = target.unsqueeze(0).to(device).float().clamp(0, 1)
        if not torch.isfinite(reconstruction).all():
            raise RuntimeError(f"Non-finite reconstruction at image {index}")
        mse = torch.mean((reconstruction - target) ** 2).item()
        psnr = -10.0 * math.log10(max(mse, 1e-12))
        target_np = target[0].permute(1, 2, 0).cpu().numpy()
        reconstruction_np = reconstruction[0].permute(1, 2, 0).cpu().numpy()
        ssim = structural_similarity(
            target_np, reconstruction_np, channel_axis=2, data_range=1.0
        )
        image_rows.append(
            {
                "index": index,
                "psnr": psnr,
                "ssim": float(ssim),
                "bpp": 8.0 * byte_length(compressed["strings"]) / (height * width),
                "encode_ms": encode_ms,
                "decode_ms": decode_ms,
            }
        )

    summary = {"images": count}
    for key in ("psnr", "ssim", "bpp", "encode_ms", "decode_ms"):
        values = [row[key] for row in image_rows]
        summary[key] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary, image_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="MODEL:[LABEL=]PATH",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--val", default="test")
    parser.add_argument("--raw-space", default="raw_linear")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-dir", default="results/evaluation")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset = RawImageDataset(
        args.dataset, split=args.val, transform=None, nocompress=True,
        ntest=args.limit, qlist=[90], raw_space=args.raw_space,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = []
    for spec in args.checkpoint:
        model_name, label, path = parse_checkpoint_spec(spec)
        model, info, checkpoint = load_checkpoint_model(
            path, model_name, args.device
        )
        summary, image_rows = evaluate(
            model, dataset, args.device,
            checkpoint_lambda(checkpoint, args.lmbda), args.limit,
            args.warmup,
        )
        summary.update(label=label, model=info["model"])
        summary_rows.append(summary)
        print(
            f"{label}: PSNR={summary['psnr']:.4f} dB, "
            f"SSIM={summary['ssim']:.6f}, bpp={summary['bpp']:.6f}, "
            f"enc={summary['encode_ms']:.2f} ms, dec={summary['decode_ms']:.2f} ms"
        )
        with open(os.path.join(args.output_dir, f"{label}_images.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=image_rows[0].keys())
            writer.writeheader()
            writer.writerows(image_rows)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
