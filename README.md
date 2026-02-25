
# 1D-DT-Adapter (1D Dilated-Tap Adapter)

Parameter-efficient fine-tuning (PEFT) for vision backbones using a lightweight **1D-DT Adapter** (implemented in `models/hcc_adapter.py`).

This project keeps the Conv-Adapter training pipeline and adds a **1D dilated-tap, even-symmetric axial spatial adapter** with:
- group-shared coefficients,
- optional grouped `1×1` bottleneck channel mixing,
- and a safe residual gate for stable initialization.

> **Manuscript-linked repository**  
> This code is associated with:  
> **A Lightweight Adapter for Efficient Fine-Tuning in Computer Vision** (February 2026)  
> DOI: **10.21203/rs.3.rs-8843187/v1**

---

## Highlights

- ✅ Lightweight PEFT adapter for vision backbones
- ✅ Axial depthwise 1D filtering (`h`, `w`, or `hw`) with dilation
- ✅ Even-symmetric taps (parameter-efficient, low-distortion intuition)
- ✅ Group-shared coefficients for low parameter count
- ✅ Optional grouped `1×1` bottleneck for channel mixing
- ✅ Residual gate initialization for stable training
- ✅ Compatible with existing `main.py` training pipeline

---

## Repository structure

```text
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
    ├── backbones/          # resnet / convnext / swin / efficientnet / clip
    ├── heads/              # linear / mlp classifier heads
    ├── layers/             # extra layers
    └── tuning_modules/     # conv/residual adapters + side-tuning modules
````

> **Note:** The file is currently named `hcc_adapter.py` (legacy naming), while the method is referred to as **1D-DT Adapter / DT1D-Adapter** in the manuscript.

---

## Method overview (short)

Given a feature map `x ∈ R^{B×C×H×W}`, the **1D-DT adapter** builds a residual update:

* **Axial depthwise 1D dilated taps** along height/width (`axis ∈ {h,w,hw}`), kernel `K = 2M + 1`
* **Even symmetry**: taps at `±m` share the same coefficient
* **Group-shared coefficients**: channels share α within groups (`alpha_group`)
* **Optional grouped 1×1 bottleneck** (`pw_ratio`, `pw_groups`) for lightweight channel mixing
* **Residual gate** (`gate_init`) keeps the adapter near identity at initialization

Update rule:

```text
out = x + residual_scale * gate * adapter(x)
```



## Installation

```bash
pip install torch torchvision timm
```

Optional (if using TensorBoard / distributed utilities):

```bash
pip install tensorboard
```

---

## Dataset preparation

This repo expects datasets under:

```text
./data/<dataset_name>
```

Examples:

```text
./data/cifar10
./data/cifar100
./data/pets
./data/flowers
```

You can also pass a custom path via `--data_path`.

---

## How to run

All experiments are launched through **`main.py`**.

### Main flags

#### Dataset

* `--dataset <name>`
* `--data_path <path>`
* `--nb_classes <int>`

#### Backbone

* `--backbone <model_name>`
* `--weights <torchvision_weight_enum_or_none>` (optional)
* `--input_size <int>` (e.g., 224)
* `--imagenet_default_mean_and_std True|False`

#### Tuning method

* `--tuning_method dt`  (1D-DT adapter)
* Other methods may be available in `models/tuning_modules/` (e.g., `conv`, `residual`, `side`, ...)

#### 1D-DT adapter hyperparameters

* `--dt_h`
* `--dt_M`
* `--dt_axis` (`h`, `w`, `hw`)
* `--dt_per_channel` (legacy) / `--dt_alpha_group`
* `--dt_no_pw`
* `--dt_pw_ratio`
* `--dt_pw_groups`
* `--dt_gate_init`
* `--dt_padding`
* global: `--adapt_scale`

> If your CLI still uses `--hcc_*` flag names, keep using them (legacy naming is supported in some versions).

---

## Quick start examples

### 1) CIFAR-10 + ResNet-18

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

### 2) CIFAR-100 + ResNet-50 (ImageNet pretrained)

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

### 3) CPU sanity run

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

* a results/log directory (configured in `main.py` / `utils.py`)
* checkpoints (if enabled)
* training logs (stdout and optional TensorBoard files)

---

## Reproducibility notes

For reproducible experiments, please record:

* exact commit hash
* PyTorch / torchvision / timm versions
* GPU/CPU info
* random seed(s)
* dataset split protocol
* full CLI command

Recommended additions for the repo:

* `requirements.txt` or `environment.yml`
* `scripts/` with exact commands used in the paper
* `CITATION.cff`
* Zenodo archive release (optional, for permanent software DOI)

---

## Tips & gotchas

* **Backbone freezing:** In PEFT, backbone parameters are typically frozen while only adapter/head parameters are trained.
* **BatchNorm freezing:** When the backbone is frozen, BatchNorm should usually be kept in `eval()` mode.
* **Input size matters:**

  * CIFAR: often `32×32`
  * ImageNet-pretrained backbones: usually `224×224` + ImageNet mean/std
* **Hyperparameter sensitivity:** `dt_M`, `dt_h`, `dt_alpha_group`, and `dt_pw_ratio` affect the accuracy/parameter trade-off.

---

## Citation

If you use this repository, please cite the associated manuscript:

### Paper (February 2026)

**A Lightweight Adapter for Efficient Fine-Tuning in Computer Vision** ( preprint )
DOI: **10.21203/rs.3.rs-8843187/v1**

```bibtex
@article{tran2026lightweightadapter,
  title   = {A Lightweight Adapter for Efficient Fine-Tuning in Computer Vision},
  author  = {Tran, Kim Huong and Dang, Ba Ty},
  year    = {2026},
  month   = feb,
  doi     = {10.21203/rs.3.rs-8843187/v1},
  url     = {https://doi.org/10.21203/rs.3.rs-8843187/v1}
}
```

### Baseline / training pipeline inspiration (Conv-Adapter)

```bibtex
@inproceedings{chen2024convadapter,
  title={Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets},
  author={Chen, Wei and Gao, Peng and Zhang, Xiaoyu and others},
  booktitle={CVPR Workshops},
  year={2024}
}
```

---

## Maintainers

* **Tran Kim Huong**
* **Dang Ba Ty**


