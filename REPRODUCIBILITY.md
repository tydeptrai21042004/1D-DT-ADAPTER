# CNN Three-Seed Reproducibility Protocol

## Scope

This release standardizes the **CNN classification** results in the manuscript. Transformer/ViT experiments and dense-prediction experiments are not dispatched by `tools/run_cnn_paper.py`. The runner fails when a target requests a ViT, Swin, Transformer, DeiT, BEiT, or CLIP backbone.

## Independent seeds

The official seed set is:

```text
0, 1, 2
```

Each seed controls:

- Python `random`;
- NumPy;
- PyTorch CPU and CUDA RNGs;
- random samplers and distributed samplers;
- DataLoader worker RNGs;
- model and task-head initialization;
- augmentation randomness;
- generated train/validation partitions for datasets without an official validation split.

For fair paired comparison, all methods under the same target and seed use the same dataset partition. Results are aggregated across seeds only after every required run completes.

## Full fine-tuning

`--tuning_method full` replaces the classifier for the downstream class count and enables gradients for every model parameter. It is not a PEFT method; it is the upper-capacity reference.

## Linear probing

`--tuning_method linear` replaces the classifier and trains only classifier parameters. All backbone parameters are frozen and BatchNorm modules remain in evaluation mode, preventing running-statistic updates from leaking adaptation into the frozen feature extractor.

## Manifest-driven experiments

The authoritative matrix is:

```text
configs/paper/cnn_three_seed_manifest.yaml
```

It stores:

- manuscript target identifiers;
- dataset and CNN backbone;
- epoch and batch-size protocol;
- method list;
- baseline-specific arguments;
- DT1D hyperparameters;
- the default three-seed set.

All 408 per-method/per-seed YAML files are committed under `configs/paper/generated/<target>/`. They use portable repository-relative data and split paths. The runner regenerates the same files from the authoritative manifest before execution.

## Split policy

- DTD and Flowers102 use official torchvision partitions.
- Food-101, SVHN, Oxford-IIIT Pet, and FGVC-Aircraft use their official test split and a deterministic seed-specific validation split from the training partition.
- Caltech101 and EuroSAT have no official torchvision test partition. They use disjoint seed-specific 80% train / 10% validation / 10% test partitions. Caltech101 manifests are committed; EuroSAT manifests are generated and recorded per run.
- Caltech101 split manifests for seeds 0, 1, and 2 are committed in `splits/caltech101/`.
- Any generated split is written into the run directory as `split_manifest_used.json`.

## Result aggregation

Run:

```bash
python tools/aggregate_cnn_paper.py \
  --root outputs/cnn_paper_three_seed \
  --target table_14_15 \
  --require-seeds 0,1,2
```

The aggregation tool refuses to describe an incomplete group as a complete three-seed result. It writes a separate `seed_completeness.json` report.

Report stochastic metrics as arithmetic mean ± sample standard deviation (`ddof=1`). Parameter counts and FLOPs are expected to be seed invariant. Hardware-sensitive latency, FPS, memory, epoch time, and total training time must be measured on the same GPU/software environment and reported with the environment metadata.

## Figure policy

- Figure 1: mean train/validation curves across seeds with ±1 standard-deviation bands.
- Figure 2: deterministic mathematical visualization; no seed is applicable.
- Figure 3: deterministic architecture diagram; no seed is applicable.
- Figure 4: mean test accuracy across seeds with vertical ±1 standard-deviation error bars; parameter count is the seed-invariant horizontal coordinate.

## Run metadata

Before training, every run stores:

- resolved configuration and source-config checksum;
- executable command;
- branch, commit, and dirty status;
- Python, package, CUDA, cuDNN, GPU, and driver information;
- resolved torchvision pretrained-weight enum and source URL;
- independent seed and manuscript target mapping.

During/after training it stores stdout, run status, epoch history, convergence summary, test metrics, efficiency metrics, split manifest, and checkpoint files when enabled.

## Smoke validation

The FakeData smoke mode is only for code-path validation. It must never be used for manuscript numbers.

## Source/manuscript alignment preflight

Run `python tools/preflight_cnn_matrix.py --target all` before GPU training. It instantiates 136 unique CNN model configurations without downloading datasets or weights and validates the trainability policy for Full fine-tuning, Linear probing, and every implemented PEFT baseline. Historical parameter-count differences are reported diagnostically; see `MANUSCRIPT_ALIGNMENT_NOTES.md`.
