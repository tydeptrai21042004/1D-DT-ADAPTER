# 1D-DT-Adapter (1D Dilated-Tap Adapter)

Parameter-efficient fine-tuning (PEFT) for convolutional backbones using a lightweight **1D-DT Adapter** (implemented in `models/hcc_adapter.py`).
The project keeps the Conv-Adapter training pipeline, but adds a **1D dilated-tap, even-symmetric** spatial adapter with a safe residual gate and optional 1×1 bottleneck mixing.

> If you use this code, please cite **Conv-Adapter** and **this repository**.

---

## Directory structure

```
tydeptrai21042004-1d-dt-adapter/
├── main.py                 # entry point (CLI)
├── engine.py               # training/evaluation loops
├── utils.py                # logging, schedulers, misc utilities
├── memory_utils.py         # memory helpers
├── datasets/               # dataset builders & transforms
│   ├── build.py
│   ├── fewshot.py
│   ├── fgvc.py
│   ├── vdd.py
│   └── vtab.py
└── models/
    ├── hcc_adapter.py      # 1D-DT adapter module (depthwise 1D dilated taps + optional PW)
    ├── backbones/          # model definitions (resnet/convnext/swin/efficientnet/clip)
    ├── heads/              # linear/mlp classifier heads
    ├── layers/             # extra layers (e.g., ws_conv)
    └── tuning_modules/     # conv/residual adapters + side-tuning modules
```

---

## Method (short analysis)

Given a feature map `x ∈ R^{B×C×H×W}`, the **1D-DT adapter** builds a small residual update:

* **Depthwise 1D dilated taps** along height/width (`axis ∈ {h,w,hw}`), kernel `K=2M+1`, dilation `h`.
* **Even symmetry**: taps at `±m` share the same coefficient ⇒ cosine-only / linear-phase intuition (low distortion).
* **Group-shared coefficients**: channels share α within groups (`alpha_group`) ⇒ parameter-efficient.
* **Optional grouped 1×1 bottleneck** (`pw_ratio`, `pw_groups`) for lightweight channel mixing.
* **Residual gate** (`gate_init`) keeps the adapter near-identity at initialization:
  `out = x + residual_scale * gate * adapter(x)`.

---

## Installation

```bash
pip install torch torchvision timm
```

(Optional) If your repo uses distributed training utilities:

```bash
pip install tensorboard
```

---

## Prepare datasets

This repo expects datasets under `./data/<name>` (or whatever you pass to `--data_path`).

Examples:

```
./data/cifar10
./data/cifar100
./data/pets
./data/flowers
```

---

## How to run

All experiments run through **`main.py`**. The most important flags are:

* **Dataset**

  * `--dataset <name>`
  * `--data_path <path>`
  * `--nb_classes <int>`

* **Backbone**

  * `--backbone <model_name>`
  * (optional) `--weights <torchvision_weight_enum_or_none>`
  * (optional) ImageNet preprocessing: `--input_size 224 --imagenet_default_mean_and_std True`

* **Tuning method**

  * `--tuning_method dt` (your 1D-DT adapter)
  * (or other supported methods in `models/tuning_modules/`: `conv`, `residual`, `side`, …)

* **1D-DT adapter params** (mapped to `models/hcc_adapter.py`)

  * `--dt_h`, `--dt_M`, `--dt_axis`
  * `--dt_per_channel` (legacy) / `--dt_alpha_group`
  * `--dt_no_pw`, `--dt_pw_ratio`, `--dt_pw_groups`
  * `--dt_gate_init`, `--dt_padding`
  * global: `--adapt_scale`

> If your current CLI still uses `--hcc_*` names, keep using them.
> Only the adapter file has been renamed conceptually—CLI flag names depend on `main.py`.

---

## Quick start examples

### 1) CIFAR-10 (32×32) + ResNet-style backbone

```bash
python main.py \
  --dataset cifar10 --data_path ./data/cifar10 --nb_classes 10 \
  --backbone resnet18 \
  --tuning_method dt \
  --dt_h 1 --dt_M 2 --dt_axis hw \
  --dt_per_channel True \
  --dt_pw_ratio 8 --dt_gate_init 0.0 \
  --dt_padding reflect --adapt_scale 1.0 \
  --batch_size 128 --epochs 200 --lr 1e-3 \
  --dist_eval False
```

### 2) CIFAR-100 + torchvision ResNet-50 (ImageNet weights)

```bash
python main.py \
  --dataset cifar100 --data_path ./data/cifar100 --nb_classes 100 \
  --backbone resnet50 --weights ResNet50_Weights.IMAGENET1K_V2 \
  --tuning_method dt \
  --input_size 224 --imagenet_default_mean_and_std True \
  --dt_h 1 --dt_M 2 --dt_axis hw \
  --dt_alpha_group 16 --dt_pw_ratio 32 --dt_gate_init 0.1 \
  --dt_padding reflect --adapt_scale 1.0 \
  --batch_size 64 --epochs 50 --lr 1e-4 \
  --dist_eval False
```

### 3) CPU run (AMP auto-disable is recommended)

```bash
python main.py \
  --device cpu \
  --dataset cifar10 --data_path ./data/cifar10 --nb_classes 10 \
  --backbone resnet18 \
  --tuning_method dt \
  --dt_h 1 --dt_M 2 --dt_axis hw \
  --batch_size 64 --epochs 5 --lr 1e-3 \
  --dist_eval False
```

---

## Outputs / logs

Typical runs create:

* a results/log directory (configured inside `main.py` / `utils.py`)
* checkpoints (if enabled)
* training logs (stdout + optional tensorboard)

If you want, paste the part in `main.py` where it defines `output_dir/results_dir`, and I’ll write the exact “where results go” section for your README.

---

## Tips & gotchas

* **Backbone freezing**: with adapters, the backbone is typically frozen and only adapter/head params train.
* **Input size matters**:

  * CIFAR: `32×32`
  * ImageNet backbones: usually `224×224` + ImageNet mean/std
* **BN freezing**: when backbone is frozen, BN should be in `eval()` (this repo includes that fix).

---

## Citation

* **Conv-Adapter**: arXiv:2208.07463
* **This repository**: 1D-DT-Adapter

---

## Maintainers

* Tran Kim Huong
* Dang Ba Ty
