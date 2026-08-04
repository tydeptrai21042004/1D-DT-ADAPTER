# HOSQ-Lite-C1-Orth final implementation report

## Scope

This package contains one revised proposal, **HOSQ-Lite-C1-Orth**, and retains the submitted **DT1D-Adapter** only as the original-method baseline. All other proposal branches have been removed. The remaining component switches are ablations of HOSQ-Lite-C1-Orth.

## Final operator

For axis `a` and original Group-16 channel group `g`, the coarse symmetric kernel is

\[
K^{(0)}_{a,g}=\beta_{a,g,0}\delta_0+\sum_{r\in\{1,2,4,8\}}\beta_{a,g,r}(\delta_{-r}+\delta_r).
\]

One weighted Helmert contrast `h_g(c)` is used inside each original channel group, with

\[
\sum_{c\in g}h_g(c)=0,\qquad \sum_{c\in g}h_g(c)^2=1.
\]

The spatial detail atoms are generated from

\[
\psi_r=\delta_{-r}+\delta_r-2\delta_0,\qquad r\in\{4,8\},
\]

and orthonormalized into `q4,q8`. The channel-dependent kernel is

\[
K_{a,c}=K^{(0)}_{a,g(c)}+h_{g(c)}(c)\left(\eta_{a,g(c),4}q_4+\eta_{a,g(c),8}q_8\right).
\]

A joint two-axis projection enforces

\[
\sum_{a\in\{h,w\}}\|K_{a,c}\|_1\le 1,
\]

and the residual output is

\[
y=x+\gamma(T_hx+T_wx).
\]

The implementation executes one 17-tap depthwise convolution per enabled axis.

## Why this remains close to DT1D

The revised method preserves the original Group-16 sharing, shifted symmetric axial filtering, dyadic long-range support, height/width separation, scalar residual gate, insertion locations, and frozen-backbone PEFT protocol. The changes are limited to an identifiable coarse coordinate system, one channel-orthogonal contrast per original group, and fused axial evaluation.

Setting both detail coordinates to zero produces the **coarse-only ablation**. Initializing the coarse coordinates from original DT1D under a common `replicate` boundary gives numerical forward equality at every ResNet stage size.

## Focused ablations

| Ablation | Purpose |
|---|---|
| Original DT1D + pointwise | Submitted-method baseline |
| Original DT1D core | Remove pointwise mixing only |
| Without orthogonal detail | Isolate the fused coarse quotient |
| Offset-4 only | Test short/mid-range detail |
| Offset-8 only | Test long-range detail |
| Raw zero-DC atoms | Test the effect of orthogonal conditioning |
| Final orthogonal two-coordinate method | Complete proposal |
| Height-only / width-only | Axis contribution |
| Group-8 / Group-32 | Sharing sensitivity |

## Validation results

The full retained repository test suite passes. Dedicated invariants verify:

- exact original-DT1D warm-start equivalence under `replicate` padding;
- zero-DC spatial details;
- zero group-mean channel details;
- orthonormal direct-sum coordinates;
- joint two-axis non-expansiveness;
- nonzero gradients from zero detail initialization;
- mixed-precision forward/backward;
- exact state restoration;
- one convolution per enabled axis.

The mathematical validator reports an orthogonal-basis Gram condition number near 1, compared with 5 for the raw atoms. The complete two-coordinate basis fits the synthetic target detail to numerical precision, while removing either coordinate leaves nonzero residual error.

## Structural latency

Across eight ResNet-18 adapter positions:

| Variant | Adapter convolution calls |
|---|---:|
| Original DT1D + pointwise | 64 |
| Original DT1D core | 48 |
| HOSQ-Lite-C1-Orth | 16 |

The included CPU benchmark confirms the expected latency direction. CPU values are diagnostic only; publication results must use synchronized GPU timing with identical batch size, precision, warm-up, iterations, input size, and hardware.

## Boundary rule

Use `replicate` padding for both original DT1D and HOSQ-Lite-C1-Orth in equivalence and latency comparisons. Under `reflect`, the separate dilation branches and the fused radius-8 kernel can select different finite-boundary extensions at the final `7×7` stage.

## Publication decision

HOSQ-Lite-C1-Orth is the sole revised proposal. Report original DT1D as the submitted-method baseline and use the focused ablations to attribute accuracy, parameter, and latency changes. Three-seed classification experiments remain necessary before claiming an accuracy improvement.
