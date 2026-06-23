# MambaRaw: Selective State Space Modeling for Efficient 4K RAW Image Reconstruction (ECCV 2026)

Official implementation of the ECCV 2026 paper:

> **MambaRaw: Selective State Space Modeling for Efficient 4K RAW Image Reconstruction**
>
> Peize Li*, Fanhu Zeng*, Tongda Xu, Xinjie Zhang, Xingtong Ge, Haotian Zhang, Xingguo Xu, Yan Wang

## Overview

MambaRaw is a JPEG-guided metadata-based RAW reconstruction framework for efficient high-resolution inference. Its core contribution is a **spatial-energy coupled context model** for entropy-parameter estimation. The method introduces:

- **TileMambaBlock**, an energy-guided selective SSM module that ranks latent tiles by L2 energy and applies VSS processing to information-dense regions;
- **Energy-Aware Refinement (EAR)**, a local energy-guided refinement module with an identity-initialized residual branch.

These modules are integrated into the Level-1 entropy-parameter network of a two-level JPEG-conditioned RAW codec. The default paper configuration uses `N=192`, `T=64`, and `rho=0.5`. The implementation trains one independent checkpoint for each rate-distortion multiplier `lambda`.

## Code Provenance

This repository combines the proposed MambaRaw modules with established open-source components for reproducible training and evaluation:

- **MambaRaw contribution:** spatial-energy coupled entropy modeling with TileMambaBlock and EAR, integrated into the Level-1 entropy-parameter network.
- **R2LCM / Beyond-R2LCM** ([wyf0912/R2LCM](https://github.com/wyf0912/R2LCM)): base two-level RAW codec, learned sampling masks, entropy coding API, and `compress` / `decompress` evaluation path.
- **MambaIC / VMamba-style selective scan** ([AuroraZengfh/MambaIC](https://github.com/AuroraZengfh/MambaIC)): VSS / SS2D implementation used as the SSM operator inside TileMambaBlock.

Following the original codec path ensures that bitrate, encoding time, decoding time, PSNR, and SSIM are measured with real entropy coding rather than a forward-only proxy.

## Environment

Install the R2LCM environment:

```bash
git clone https://github.com/wyf0912/R2LCM.git
cd R2LCM
git checkout 429c7da21e1060ea901652524a23d7697acf3fb4
pip install -e .
```

Install the selective-scan dependencies following [MambaIC](https://github.com/AuroraZengfh/MambaIC) commit `160c43f51cfd22f91b9782cee55b545fa8939324`.

## Dataset

- For NUS evaluation, use the same post-processed downsampled setting as prior metadata-based RAW reconstruction work.
- The processed NUS dataset is available [here](https://ln5.sync.com/dl/9bf21ed40/z6bn94xs-uiaeij3m-rc3izeje-3epv2uz7/view/default/13870041630008).
- MIT-Adobe FiveK can be downloaded from the [official website](https://data.csail.mit.edu/graphics/fivek/).

## Training

Train one checkpoint for each `lambda` in `{0.02, 0.24, 0.8, 1.5, 2.0, 5.0, 10.0, 20.0}`.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model mambaraw \
  --dataset /path/to/dataset \
  --train train \
  --val test \
  --lambda 0.8 \
  --epochs 1000 \
  --batch-size 8 \
  --patch-size 256 \
  --num-workers 8 \
  --raw_space raw_linear \
  --file_type tif \
  --nocompress \
  --N 192 \
  --M 320 \
  --reduce-c 8 \
  --down_num 2 \
  --sampling-num 4 \
  --use-deconv \
  --rounding_aux forward \
  --tile-size 64 \
  --tile-keep-ratio 0.5 \
  --save-root ./checkpoints
```

The `mambaraw` entry automatically uses the paper L1 distortion objective and rejects variable-rate `--lambda_list` training.

## Evaluation

Evaluate a MambaRaw checkpoint:

```bash
python eval_comprehensive.py \
  --checkpoint mambaraw:MambaRaw=/path/to/mambaraw.pth.tar \
  --dataset /path/to/dataset \
  --val test \
  --raw-space raw_linear \
  --lambda 0.8 \
  --device cuda \
  --output-dir results/sony
```

Evaluate a Beyond-R2LCM baseline checkpoint:

```bash
python eval_comprehensive.py \
  --checkpoint baseline:Baseline=/path/to/baseline.pth.tar \
  --dataset /path/to/dataset \
  --val test \
  --raw-space raw_linear \
  --lambda 0.8 \
  --device cuda \
  --output-dir results/baseline
```

The checkpoint prefix must be explicit: use `mambaraw:` for MambaRaw checkpoints and `baseline:` for Beyond-R2LCM checkpoints. The evaluation script performs real entropy coding through `model.compress` and `model.decompress`, and reports PSNR, SSIM, metadata bpp, encoding time, and decoding time.

## 4K Benchmark

Synthetic 4K benchmark:

```bash
python benchmark_4k.py \
  --checkpoint mambaraw:MambaRaw=/path/to/mambaraw.pth.tar \
  --height 2160 \
  --width 3840 \
  --lambda 0.8 \
  --device cuda \
  --csv results/benchmark_4k.csv
```

Real-image benchmark:

```bash
python benchmark_4k_speed.py \
  --checkpoint mambaraw:MambaRaw=/path/to/mambaraw.pth.tar \
  --dataset /path/to/dataset \
  --val test \
  --lambda 0.8 \
  --device cuda \
  --csv results/benchmark_real.csv
```

## Acknowledgement

This project is built upon [R2LCM](https://github.com/wyf0912/R2LCM), [MambaIC](https://github.com/AuroraZengfh/MambaIC), [CompressAI](https://github.com/InterDigitalInc/CompressAI), and [VMamba](https://github.com/MzeroMiko/VMamba). We thank the authors for releasing their code.
