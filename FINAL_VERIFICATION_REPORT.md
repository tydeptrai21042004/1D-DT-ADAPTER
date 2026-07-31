# DT1D Exact Realization — Final Independent Verification

## Final verdict

The revised DT1D path is an **exact mathematical realization of the same static axis–scale operator**. It does not change the learned parameters, checkpoint schema, pointwise block, residual gate, router, padding rules, datasets, optimizer, or baselines.

Within every executable check available in this environment, the non-time metrics were preserved to floating-point tolerance. The final paired timing test supports a reliable **inference** improvement. End-to-end training-time improvement is positive in the median but is not yet statistically conclusive, so the paper should not claim a guaranteed epoch-time reduction until it is rerun on the original GPU.

## Final tests rerun

### Automated tests

- 226 focused mathematical and regression tests passed.
- 720 exhaustive configurations passed.
- 240 additional independently sampled configurations passed.
- No modeled tap-count or convolution-count regression was observed.

The broader independent audit varied axis mode, channel count, kernel radius, dilation set, padding, feature-map size, pointwise block, BatchNorm, shared/separate axis kernels, coefficient-sharing group size, routing temperature, and train/eval mode.

### Numerical preservation

Across the final 240-case independent audit:

| Quantity | Maximum absolute difference |
|---|---:|
| Adapter output | 4.7684e-7 |
| Loss | 4.7684e-7 |
| Input gradient | 5.9605e-8 |
| Trainable-parameter gradient | 1.1921e-7 |
| BatchNorm running statistics | 4.6566e-10 |
| Minimum prediction agreement | 100% |

Across a 300-step AdamW trajectory with the pointwise block and BatchNorm enabled:

| Quantity | Result |
|---|---:|
| Maximum loss difference | 2.3842e-7 |
| Maximum logit difference | 5.9605e-8 |
| Minimum top-1 agreement | 100% |
| Maximum parameter difference | 1.1921e-7 |
| Maximum optimizer-state difference | 4.1910e-9 |
| Maximum BatchNorm-buffer difference | 5.9605e-8 |

CPU bfloat16 autocast also retained 100% top-1 proxy agreement. CUDA fp16/AMP could not be tested because this environment has no GPU.

## Metrics that are exactly unchanged

- Trainable parameter count
- Buffer count
- State-dictionary keys and tensor shapes
- Serialized adapter checkpoint size: 7,928 bytes before and after in the representative test
- Backbone, head, pointwise branch, routing logits, residual gate, normalization, and padding policy
- Dataset and baseline code

## Computation and memory profiling

Representative adapter, batch 4, 64 channels, 56x56:

| Profiler quantity | Before | After | Change |
|---|---:|---:|---:|
| Reported FLOPs | 60,211,544 | 55,394,688 | -8.00% |
| `conv2d` calls | 8 | 6 | -25.0% |
| Largest key-level CPU allocation | 22,887,776 B | 16,469,216 B | -28.0% |

Controlled ResNet-18, batch 2, 224x224:

| Profiler quantity | Before | After | Change |
|---|---:|---:|---:|
| Reported FLOPs | 7,363,157,920 | 7,354,126,560 | -0.123% |
| `conv2d` calls | 84 | 68 | -19.0% |
| Largest key-level CPU allocation | 82,135,296 B | 69,635,552 B | -15.2% |

These are CPU profiler measurements, not GPU peak-memory results.

## Randomized interleaved timing

The final timing test randomizes whether the before or after variant is timed first in each round. Both variants use the same class, weights, input, kernel builder, pointwise block, and operators. No cache, compiler, mixed precision, or pipeline modification is used.

### Adapter level

| Measurement | Median speedup | 95% bootstrap interval |
|---|---:|---:|
| Inference | 1.302x | [1.276x, 1.314x] |
| Forward/backward | 1.342x | [1.280x, 1.406x] |

### Controlled ResNet-18

| Measurement | Median speedup | 95% bootstrap interval |
|---|---:|---:|
| Inference | 1.095x | [1.055x, 1.218x] |
| Forward/backward | 1.072x | [0.898x, 1.150x] |

The ResNet-18 inference gain is supported by the paired interval. The full-model training interval includes 1.0, so a strong training-time claim is not justified from the current CPU evidence.

## Publication-safe conclusion

The supported claim is:

> The exact minimum-cost realization preserves the DT1D mapping, trainable parameters, checkpoints, stability bound, and spectral multiplier while reducing the number of axial convolution calls and improving measured inference latency.

Do not yet claim:

- guaranteed faster epoch or total training time on every device;
- unchanged manuscript dataset scores without evaluating the actual saved checkpoints;
- GPU latency or GPU memory improvement from the CPU results.

For the strongest no-regression evidence, load each existing paper checkpoint once and evaluate it with `exact_cost_realization=false` and `true` on the same test loader. The checkpoint is identical, so this isolates only the exact mathematical realization. Re-profile latency/FPS using the same warm-up, batch size, precision, and hardware used in the manuscript.
