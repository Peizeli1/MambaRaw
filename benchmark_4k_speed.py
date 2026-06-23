#!/usr/bin/env python3
"""Real-image codec benchmark with one strictly matched checkpoint per model."""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from compressai.datasets import RawImageDataset
from tqdm import tqdm

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


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_model(model, dataset, device, lmbda, limit, warmup):
    model.update(force=True)
    variable_rate = model.lambda_list is not None
    codec_lambda = lmbda if variable_rate else None
    encode_ms, decode_ms, bpps = [], [], []
    count = min(len(dataset), limit) if limit > 0 else len(dataset)

    for index in tqdm(range(count)):
        x_jpg, x_raw, _ = dataset[index]
        height, width = x_raw.shape[-2:]
        x_jpg = pad_to_multiple(x_jpg.unsqueeze(0).to(device))
        x_raw = pad_to_multiple(x_raw.unsqueeze(0).to(device))

        if index == 0:
            for _ in range(warmup):
                warmup_result = model.compress(
                    x_raw, x_jpg, lmbda=codec_lambda
                )
                model.decompress(
                    warmup_result["strings"],
                    warmup_result["shape"],
                    x_jpg,
                    compressed_result=warmup_result,
                    lmbda=codec_lambda,
                )
            sync()

        sync()
        start = time.perf_counter()
        compressed = model.compress(x_raw, x_jpg, lmbda=codec_lambda)
        sync()
        encode_ms.append((time.perf_counter() - start) * 1000.0)

        sync()
        start = time.perf_counter()
        decoded = model.decompress(
            compressed["strings"], compressed["shape"], x_jpg,
            compressed_result=compressed, lmbda=codec_lambda,
        )["x_hat"]
        sync()
        decode_ms.append((time.perf_counter() - start) * 1000.0)
        if not torch.isfinite(decoded).all():
            raise RuntimeError(f"Non-finite decoder output at image {index}")
        bpps.append(
            8.0 * byte_length(compressed["strings"])
            / (height * width)
        )

    return {
        "images": count,
        "encode_ms": float(np.mean(encode_ms)),
        "encode_std_ms": float(np.std(encode_ms)),
        "decode_ms": float(np.mean(decode_ms)),
        "decode_std_ms": float(np.std(decode_ms)),
        "bpp": float(np.mean(bpps)),
    }


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
    parser.add_argument("--csv", default="results/benchmark_real.csv")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset = RawImageDataset(
        args.dataset, split=args.val, transform=None, nocompress=True,
        ntest=args.limit, qlist=[90], raw_space=args.raw_space,
    )
    rows = []
    for spec in args.checkpoint:
        model_name, label, path = parse_checkpoint_spec(spec)
        model, info, checkpoint = load_checkpoint_model(
            path, model_name, args.device
        )
        result = benchmark_model(
            model, dataset, args.device,
            checkpoint_lambda(checkpoint, args.lmbda), args.limit,
            args.warmup,
        )
        result.update(label=label, model=info["model"])
        rows.append(result)
        print(
            f"{label}: encode={result['encode_ms']:.2f}+-{result['encode_std_ms']:.2f} ms, "
            f"decode={result['decode_ms']:.2f}+-{result['decode_std_ms']:.2f} ms, "
            f"bpp={result['bpp']:.5f}"
        )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
