# HOSQ-DT1D: implemented final specification

## Method

HOSQ-DT1D retains the DT1D shifted-symmetric axial origin but replaces the
redundant dyadic branch coordinates with a hierarchical observable quotient.
For each coarse channel group and axis, it learns an MLQ8 kernel on offsets

\[
\{0,\pm1,\pm2,\pm4,\pm8\}.
\]

A Group-32 coarse kernel is refined through Group-8 subgroup contrasts. The
final setting activates one orthogonal contrast at offset 4 and two at offset
8. Fine corrections use the zero-DC atoms

\[
\psi_r=\delta_{-r}+\delta_r-2\delta_0,\qquad r\in\{4,8\}.
\]

The final residual adapter is

\[
y=x+\gamma T_{\mathrm{HOSQ}}x.
\]

No pointwise block is used in the final preset.

## Final configuration

```yaml
dt_M: 1
dt_dilations: "1,2,4"
dt_scale_adaptive: true
dt_axis: hw
dt_alpha_group: 32
dt_minimal_quotient_realization: false
dt_quotient_support_cap: 8
dt_hosq_realization: true
dt_hosq_subgroup_size: 8
dt_hosq_rank4: 1
dt_hosq_rank8: 2
dt_no_pw: true
dt_gate_init: 0.01
dt_padding: reflect
```

## Mathematical invariants implemented

1. For four subgroups, the fixed channel basis is zero mean and orthonormal.
2. With all detail coefficients zero, HOSQ exactly reduces to MLQ8 Group-32.
3. Every fine spatial atom has zero DC response.
4. The coarse and detail coordinates are separated by zero-mean channel
   contrasts, preventing detail coordinates from changing the coarse group
   average.
5. Joint height/width L1 projection guarantees

\[
\sum_{a\in\{h,w\}}\|k_{a,c}\|_1\le 1
\]

for every channel.
6. HOSQ still evaluates one ordinary depthwise convolution per enabled axis.

## Parameter budget on Caltech101/ResNet-18

The eight ResNet-18 adapters use channel dimensions
`64,64,128,128,256,256,512,512`.

- Coarse MLQ8 parameters: 600
- Orthogonal detail parameters: 360
- Residual gates: 8
- Total adapter parameters: 968
- Classifier parameters: 51,813
- Total trainable parameters: **52,781**

## Validation completed

```text
316 passed
1 skipped
10 subtests passed
```

The deterministic HOSQ validator confirms:

- Haar zero-mean error: 0
- Haar Gram error: below 1e-6
- MLQ8-to-HOSQ forward error with zero detail: 0
- MLQ8-to-HOSQ input-gradient error: 0
- zero-DC detail error: below 1e-10
- maximum joint-axis L1 norm: 1.0
- convolution calls: height 17-tap + width 17-tap
- ResNet-18 adapter budget: 968

A complete one-epoch FakeData training, validation, and final-test run also
completed successfully on CPU. This validates execution and gradients; it does
not establish the final Caltech101 accuracy.

## Experimental status

The method is fully runnable, but the proposed higher accuracy remains an
experimental hypothesis until the three-seed GPU runs finish. The code does not
claim an unmeasured improvement. The main comparison and HOSQ ablation manifests
are ready for Kaggle execution.
