# Dataset and split instructions — CNN package

Set one writable dataset root for every script:

```bash
export DATA_DIR=/absolute/path/to/data
```

The torchvision datasets download automatically when their licenses and upstream URLs permit it. Every run stores the exact split information in `split_manifest_used.json` or records the official split name in its resolved configuration and log.

## Official train/validation/test partitions

- **DTD**: official `train`, `val`, and `test` partitions.
- **Flowers102**: official `train`, `val`, and `test` partitions.

No generated index split is used for these datasets.

## Official test partition with seed-specific validation split

- **SVHN**: the official training partition is divided into train/validation; the official test partition remains untouched.
- **Food-101**: the official training partition is divided into train/validation; the official test partition remains untouched.
- **Oxford-IIIT Pet**: `trainval` is divided into train/validation; the official `test` partition remains untouched.
- **FGVC-Aircraft**: `trainval` is divided into train/validation; the official `test` partition remains untouched.

For these datasets, `--seed` controls the deterministic `torch.randperm` partition of the training data. The exact generated indices are written to each run directory.

## Datasets without an official torchvision test partition

### Caltech101

The package commits a disjoint **80% train / 10% validation / 10% test** partition for each seed:

```text
splits/caltech101/seed0_holdout20.json
splits/caltech101/seed1_holdout20.json
splits/caltech101/seed2_holdout20.json
```

For each seed the split contains 6,942 training images, 868 validation images, and 867 test images. The loader rejects the split if the loaded dataset length is not 8,677 or if indices are missing, duplicated, overlapping, or out of range.

### EuroSAT

EuroSAT is partitioned at runtime into disjoint 80% train / 10% validation / 10% test subsets for each seed. The generated manifest is stored with the run output. This prevents validation and test evaluation from reusing the same images.

## Run examples

```bash
DATA_DIR=/absolute/path/to/data bash scripts/tables/table_14_15_three_seed.sh
DATA_DIR=/absolute/path/to/data bash scripts/tables/table_18_19_three_seed.sh
```

To generate an index manifest manually:

```bash
python tools/generate_split.py \
  --dataset caltech101 \
  --length 8677 \
  --seed 0 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --output /tmp/caltech101_seed0.json
```
