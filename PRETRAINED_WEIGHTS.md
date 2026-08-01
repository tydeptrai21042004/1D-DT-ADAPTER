# Pretrained-weight protocol

## CNN paper setting

All publication targets request torchvision `weights: DEFAULT` from their committed YAML configuration. With the pinned torchvision version, this resolves to the library's named default ImageNet pretrained-weight enum for the selected CNN backbone.

Supported paper backbones are:

- ResNet-18;
- ResNet-50;
- EfficientNet-B0;
- MobileNetV3-Small.

Random initialization is used only by the explicitly marked FakeData smoke mode and must not be reported as a paper experiment.

## Recorded metadata

Before training, `tools/run_from_config.py` records:

- requested weight string;
- resolved torchvision enum class and member;
- official URL exposed by torchvision;
- installed torch and torchvision versions;
- source branch and commit;
- exact resolved command and output directory.

The metadata is stored in `resolved_config.json` and `environment.json`.

## Cache behavior

Torchvision downloads weights to the normal torch hub cache when absent. Do not rename or substitute cached files. For offline execution, populate the cache with the same pinned torchvision version and preserve the official filename.

## Dry-run verification

```bash
python tools/run_from_config.py \
  configs/paper/generated/table_14_15/dt1d_seed0.yaml \
  --output-dir /tmp/dt1d_weight_check \
  --data-path /path/to/data \
  --device cuda \
  --dry-run
cat /tmp/dt1d_weight_check/resolved_config.json
```
