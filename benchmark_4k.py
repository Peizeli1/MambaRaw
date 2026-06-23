#!/usr/bin/env python3
"""Synthetic-resolution latency benchmark using strictly matched checkpoints."""

import argparse
import csv
import math
import os
import time

import torch
import torch.nn.functional as F

from models.mambaraw_checkpoint import (
    checkpoint_lambda,
    load_checkpoint_model,
    parse_checkpoint_spec,
)


def pad_to_multiple(x, multiple=128):
    height, width = x.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def byte_length(value):
    if isinstance(value, (bytes, str)):
        return len(value)
    return sum(byte_length(item) for item in value)


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark(model, device, height, width, iterations, warmup, lmbda):
    x_raw = pad_to_multiple(torch.rand(1, 3, height, width, device=device))
    x_jpg = pad_to_multiple(torch.rand_like(x_raw))
    variable_rate = model.lambda_list is not None
    codec_lambda = lmbda if variable_rate else None

    model.update(force=True)
    for _ in range(warmup):
        compressed = model.compress(x_raw, x_jpg, lmbda=codec_lambda)
        model.decompress(
            compressed["strings"], compressed["shape"], x_jpg,
            compressed_result=compressed, lmbda=codec_lambda,
        )

    encode_times = []
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter()
        compressed = model.compress(x_raw, x_jpg, lmbda=codec_lambda)
        synchronize(device)
        encode_times.append((time.perf_counter() - start) * 1000.0)

    decode_times = []
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter()
        decoded = model.decompress(
            compressed["strings"], compressed["shape"], x_jpg,
            compressed_result=compressed, lmbda=codec_lambda,
        )["x_hat"]
        synchronize(device)
        decode_times.append((time.perf_counter() - start) * 1000.0)

    if not torch.isfinite(decoded).all():
        raise RuntimeError("Decoder produced non-finite values")
    pixels = height * width
    return {
        "encode_ms": sum(encode_times) / len(encode_times),
        "decode_ms": sum(decode_times) / len(decode_times),
        "bpp": 8.0 * byte_length(compressed["strings"]) / pixels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="MODEL:[LABEL=]PATH",
        help="Repeat for fair comparisons; every model uses its own checkpoint.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.8)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    rows = []
    for spec in args.checkpoint:
        model_name, label, path = parse_checkpoint_spec(spec)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        model, info, checkpoint = load_checkpoint_model(
            path, model_name, args.device
        )
        lmbda = checkpoint_lambda(checkpoint, args.lmbda)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        result = benchmark(
            model, args.device, args.height, args.width,
            args.iters, args.warmup, lmbda,
        )
        peak_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if args.device == "cuda" else math.nan
        )
        row = {
            "label": label,
            "model": info["model"],
            "height": args.height,
            "width": args.width,
            "encode_ms": result["encode_ms"],
            "decode_ms": result["decode_ms"],
            "total_ms": result["encode_ms"] + result["decode_ms"],
            "bpp": result["bpp"],
            "peak_memory_mb": peak_mb,
        }
        rows.append(row)
        print(
            f"{label} ({info['model']}): encode={row['encode_ms']:.2f} ms, "
            f"decode={row['decode_ms']:.2f} ms, bpp={row['bpp']:.5f}, "
            f"peak={row['peak_memory_mb']:.1f} MB"
        )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
