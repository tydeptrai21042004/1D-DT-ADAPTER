# Manuscript alignment notes for the v0.8.0 CNN rerun

## Results must be regenerated

The tables in the current manuscript are historical single-run references. The v0.8.0 package changes the reporting protocol to three independent seeds (0, 1, and 2), deterministic execution where supported, and mean ± sample standard deviation. New table values must therefore be generated from this release rather than copied from the previous PDF.

## Full fine-tuning and Linear probing

Full fine-tuning and Linear probing are added to every CNN **comparison** target. They are not inserted as rows into Table 2 because Table 2 is an internal DT1D hyperparameter ablation, not a cross-method comparison. Figure 1 is a DT1D convergence figure. Figure 4 includes both reference controls.

## CNN-only baseline scope

Prompt/VPT is not dispatched in this CNN-only package because its token-prompt mechanism is defined for tokenized Transformer backbones and is not a faithful CNN baseline. The CNN comparison matrix keeps the manuscript's implemented CNN-compatible methods and adds Full fine-tuning and Linear probing.

## Independent validation and test partitions

Caltech101 and EuroSAT do not expose official train/validation/test partitions through torchvision. The previous code path could reuse a held-out partition for both checkpoint selection and final evaluation. The v0.8.0 runner corrects this by using disjoint 80% train / 10% validation / 10% test subsets. All affected Caltech101 and EuroSAT values must be regenerated.

## Table 9 parameter-count inconsistency

The current PDF reports 66,349 trainable parameters for DT1D-Adapter on Oxford-IIIT Pet with ResNet-50. The canonical v0.8.0 implementation, with the target's 37-class classifier and the documented DT1D settings, produces 401,621 trainable parameters. The release does **not** alter the implementation merely to reproduce the old count. Table 9 must be regenerated from the canonical source, and the manuscript value should be replaced after the three-seed run. Use:

```bash
python tools/preflight_cnn_matrix.py --target table_09
bash scripts/tables/table_09_three_seed.sh
```

This note prevents an old manuscript number from being mistaken for a source-code invariant.
