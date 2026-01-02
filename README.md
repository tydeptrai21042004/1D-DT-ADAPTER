
# 1D-DT-Adapter (1D Dilated-Tap Adapter)

Parameter-efficient fine-tuning (PEFT) for convolutional backbones using a lightweight **1D-DT Adapter** module (a.k.a. **HCC-style** adapter).
This repo builds on the **Conv-Adapter** idea and adds:

* A **1D dilated-tap, even-symmetric** spatial adapter (low distortion / linear-phase behavior),
* Robust training/runtime guards (CPU fallback, AMP auto-disable on CPU, safe CUDA sync),
* A simpler backbone loader supporting **torchvision** and common **CIFAR TorchHub** backbones.

> If you use this code, please cite **Conv-Adapter** and **this repository**.

---

## Method (short analysis)

### What the 1D-DT Adapter does

Given a feature map `x ∈ R^{B×C×H×W}`, the adapter builds a small residual update:

1. **Depthwise 1D “dilated taps” along height and/or width**
   It applies a **per-channel** 1D convolution with kernel length `K = 2M+1` and dilation `h`, either:

* vertical (`axis='h'`), horizontal (`axis='w'`), or both (`axis='hw'`).

2. **Even symmetry (cosine-only response / linear phase)**
   The taps are symmetric around the center (same coefficient for `+m` and `−m`).
   This makes the spatial operator “low-distortion” in the sense that it avoids odd/phase-skew components (intuitively: it behaves like a smooth cosine response rather than a phase-warping filter).

3. **Group-shared coefficients (parameter-efficient)**
   Instead of learning a separate tap vector per channel, channels are partitioned into groups and **share** the same tap coefficients within each group.
   So α parameters scale like:

* `#α ≈ (C / alpha_group) × (M+1)` (center + M side taps)

4. **Optional tiny channel mixing (1×1 bottleneck, grouped)**
   After spatial aggregation, an optional **grouped 1×1 bottleneck** mixes channels:
   `C → (C/pw_ratio) → C` (with optional BN + ReLU).

5. **Residual gate for safe initialization**
   Output is:
   `out = x + adapt_scale * gate * adapter(x)`
   with small `gate_init` (e.g., 0.0 or 0.1), so training starts close to identity and remains stable.

**Why it’s PEFT-friendly:** you freeze the backbone and train only these small adapter parameters.

---

## What is “1D-DT” here?

**1D-DT = 1D Dilated-Tap**: a separable, depthwise 1D filter (with dilation) applied along H/W directions to add cheap spatial context.

---

## What changed vs. the original Conv-Adapter repo

### New

* **1D-DT Adapter tuning method** (`--tuning_method dt` or `hcc`, depending on your CLI naming)

* 1D-DT flags (typical):

  * `--dt_h`, `--dt_M`, `--dt_axis`
  * `--dt_per_channel` (legacy) / `--dt_alpha_group` (new)
  * `--dt_pw_ratio`, `--dt_pw_groups`, `--dt_no_pw`
  * `--dt_gate_init`, `--dt_padding` (`reflect|replicate|zeros`)

* Global adapter residual scale:

  * `--adapt_scale`

### Training/runtime quality-of-life

* CPU fallback + AMP auto-disable on CPU
* CUDA-guarded synchronizations (avoid “no NVIDIA driver” crashes)
* BatchNorm properly frozen when backbone is frozen (eval mode + affine grads off)
* Modern torchvision weights (multi-weight API, e.g., `ResNet50_Weights.IMAGENET1K_V2`)

---

## Installation

```bash
pip install torch torchvision timm
```

---

## Data layout

Place datasets under `./data/<name>`:

```
./data/cifar10
./data/cifar100
./data/pets
./data/flowers
```

---

## Backbones

Supported backbones:

* **torchvision** models via `--backbone` + `--weights`

  * Example: `resnet50` with `ResNet50_Weights.IMAGENET1K_V2`
* **CIFAR TorchHub** models via `--backbone cifar10_resnet56` (or `cifar100_resnet56`)

  * with `--cifar_hub chenyaofo` (or `akamaster`)

List torchvision model names:

```bash
python main.py --list_backbones
```

> For CIFAR TorchHub backbones, the script auto-sets `--input_size 32`.

---

## Quick start (CIFAR-10 / CIFAR-100)

> Tip: For single GPU, add `--dist_eval False`.
> For CPU: `--device cpu` (AMP auto-disables).

### CIFAR-10 + ResNet-56 (TorchHub)

```bash
python main.py \
  --dataset cifar10 --data_path ./data/cifar10 --nb_classes 10 \
  --backbone cifar10_resnet56 --cifar_hub chenyaofo \
  --tuning_method dt \
  --dt_h 1 --dt_M 2 --dt_axis hw \
  --dt_per_channel True \
  --dt_pw_ratio 8 --dt_gate_init 0.0 \
  --dt_padding reflect --adapt_scale 1.0 \
  --batch_size 128 --epochs 200 --lr 1e-3 \
  --use_amp True --dist_eval False
```

**Notes**

* `dt_per_channel True` usually means “each channel gets its own α” (legacy).
* In the newer adapter API, this is equivalent to setting `alpha_group=1`.

---

## Torchvision (ImageNet) backbone example

Example with `resnet50` + ImageNet weights:

```bash
python main.py \
  --dataset cifar100 --data_path ./data/cifar100 --nb_classes 100 \
  --backbone resnet50 --weights ResNet50_Weights.IMAGENET1K_V2 \
  --tuning_method dt \
  --input_size 224 --imagenet_default_mean_and_std True \
  --batch_size 64 --epochs 50 --lr 1e-4
```

---

## Tips & gotchas

* **Backbone freezing**: when using adapters (`--tuning_method conv|dt|residual`), the script freezes backbone params and BN affine terms; only adapter params train.
* **Input size**:

  * CIFAR backbones: `32×32`
  * ImageNet backbones: typically `224×224` + ImageNet mean/std
* **EMA**: optional via `--model_ema`; can eval EMA weights during training with `--model_ema_eval`.

---

## Repository structure

```
1D-DT-Adapter/
├── main.py                # CLI, backbones, adapters, device/AMP guards
├── engine.py              # Train/eval loops (safe CUDA sync)
├── models/                # Backbones + adapters (1D-DT / Conv / Residual)
├── datasets/              # Dataset builders + transforms
├── utils.py               # Logging, EMA, schedulers, etc.
└── ...
```

---

## Why 1D-DT Adapter?

* **Low distortion**: even-symmetric spatial aggregation behaves like a cosine-only response (linear phase intuition).
* **Tiny parameter footprint**: group-shared taps reduce α parameters significantly.
* **Drop-in**: works like Conv-Adapter—same training loop, just switch `--tuning_method dt`.

---

## Citation

Please cite:

* **Conv-Adapter**: arXiv:2208.07463 (also appeared in a CVPR 2024 Workshop context)
* **This repository**: 1D-DT-Adapter (codebase)

(If you want, tell me your preferred citation format: BibTeX, IEEE, or ACM, and I’ll generate the exact entries.)

---

## Acknowledgements & Credits

* Built on the ideas of **Conv-Adapter** (thanks to the original authors).
* Pretrained backbones via **torchvision**; CLIP baselines (if used) via OpenCLIP / OpenAI CLIP.

---

## Maintainers

* Tran Kim Huong
* Dang Ba Ty



