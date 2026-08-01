# DT1D-Adapter V9: Minimal Laurent Quotient and Spectrally Closed Dyadic Quotient

## Final recommendation

Keep the core contribution unchanged: **reduced axial filtering with shifted symmetric group-shared kernels**. Replace the over-parameterized axis-scale branch coordinates by their minimal Laurent quotient, and use the spectrally closed support set `{0, ±1, ±2, ±4}` as the final model.

Final name used in the code and ablation configs:

**SCDQ-DT1D — Spectrally Closed Dyadic Quotient DT1D-Adapter**

The final settings are:

```yaml
dt_M: 1
dt_dilations: 1,2,4
dt_scale_adaptive: true
dt_axis: hw
dt_alpha_group: 16
dt_minimal_quotient_realization: true
dt_quotient_support_cap: 4
dt_no_pw: true
dt_padding: reflect
dt_gate_init: 0.01
```

This does not replace axial convolution with attention, pruning, quantization, compilation, a custom CUDA kernel, or a hardware-specific shortcut.

## 1. Mathematical breakthrough: the old scale coordinates are not identifiable

For `M=1`, one normalized shifted-symmetric branch at dilation `d` has taps

\[
[v_d,\;u_d,\;2v_d,\;u_d,\;v_d]
\]

at offsets

\[
-2d,-d,0,d,2d.
\]

After absorbing the residual gate and routing weight into

\[
p_d=\gamma\pi_d u_d,\qquad q_d=\gamma\pi_d v_d,
\]

the Laurent symbol of the three dyadic branches `d ∈ {1,2,4}` is

\[
P(z)=\sum_{d\in\{1,2,4\}}
\left[q_d(z^{2d}+2+z^{-2d})+p_d(z^d+z^{-d})\right].
\]

Collecting equal powers gives

\[
P(z)=\beta_0+\beta_1(z+z^{-1})+\beta_2(z^2+z^{-2})
+\beta_4(z^4+z^{-4})+\beta_8(z^8+z^{-8}),
\]

where

\[
\begin{aligned}
\beta_0 &= 2(q_1+q_2+q_4),\\
\beta_1 &= p_1,\\
\beta_2 &= q_1+p_2,\\
\beta_4 &= q_2+p_4,\\
\beta_8 &= q_4.
\end{aligned}
\]

Therefore six branch coordinates map to only five observable convolution coefficients. The linear map has rank five and the exact null direction

\[
(0,1,-1,-1,1,0),
\]

because this direction changes the scale parameters but leaves the applied convolution unchanged. This is a genuine non-identifiability in the old parameterization.

The V9 MLQ parameterization learns the observable coefficients directly. It removes the flat scale-cancellation direction rather than merely implementing the same branch loop more efficiently.

## 2. Spectral closure: why the final support ends at ±4

The chosen dilation basis is

\[
D=\{1,2,4\}.
\]

The shifted construction generates first and second harmonics, so the actual harmonic set is

\[
D\cup 2D=\{1,2,4,8\}.
\]

The terms at frequencies 2 and 4 overlap with first harmonics already present at larger scales. The frequency 8 term is the only component outside the declared dyadic basis. SCDQ imposes closure in

\[
\mathcal V_D=\operatorname{span}\{1,\cos\omega,\cos2\omega,\cos4\omega\}
\]

by setting the unmatched coefficient `β8` to zero. The resulting kernel has support

\[
\{0,\pm1,\pm2,\pm4\}
\]

and length 9 rather than 17.

This is also compatible with the final `7×7` ResNet-18 feature stage. Under the periodic spectral model,

\[
\cos(8\omega_k)=\cos(\omega_k),\qquad \omega_k=2\pi k/7,
\]

so the outer harmonic is exactly aliased with the first harmonic on that stage. Under finite nonperiodic boundaries it is strongly boundary-dependent rather than a clean independent long-range mode.

Every SCDQ kernel still has a shifted-symmetric dyadic decomposition. For example, given `(β0,β1,β2,β4)`, one valid decomposition is

\[
q_1=q_2=\beta_0/4,\quad p_1=\beta_1,\quad
p_2=\beta_2-q_1,\quad p_4=\beta_4-q_2,\quad q_4=0.
\]

Thus the final method remains inside the shifted-symmetric axial-filter family; it is not a different convolutional mechanism.

## 3. Stability theorem

For sharing group `g`, define the joint two-axis magnitude

\[
s_g=\sum_{u\in\{h,w\}}
\left(|\beta_{u,0,g}|+2\sum_{r\in\{1,2,4\}}|\beta_{u,r,g}|\right).
\]

Use the projection

\[
\bar\beta_{u,r,g}=\frac{\beta_{u,r,g}}{\max(1,s_g)}.
\]

Then the combined height-width convolution has `l1` kernel norm at most one for every group. Young's inequality gives

\[
\|T_{\mathrm{SCDQ}}x\|_p\leq\|x\|_p,
\qquad 1\leq p\leq\infty.
\]

For the residual adapter

\[
y=x+\gamma T_{\mathrm{SCDQ}}x,
\]

we obtain

\[
\|y-x\|_p\leq |\gamma|\|x\|_p.
\]

The code implements this joint projection exactly.

## 4. Why this can improve optimization as well as speed

The old dyadic branch map has an exact null direction. Hence the loss is constant along at least one scale-coordinate direction whenever only the composite convolution matters. This creates:

- redundant scale updates;
- a singular local curvature direction;
- unnecessary competition between dilation-1 side taps and dilation-2 center-neighbour taps, and between dilation-2 side taps and dilation-4 center-neighbour taps.

The quotient coordinates remove this ambiguity. The direct observable coordinates have no scale-cancellation nullspace. This does not mathematically guarantee a higher test score, but it gives a concrete mechanism for more stable fitting and faster convergence rather than a post-hoc efficiency explanation.

## 5. Runtime complexity

For the V8 paper setting, the closed-form implementation evaluates two axial convolutions per axis: one fused `(d=1,d=2)` convolution and one `d=4` convolution. With both axes, that is four axial convolution calls per adapter. The optional pointwise bottleneck adds two more convolution calls.

SCDQ evaluates one length-9 axial convolution per axis and removes the non-core pointwise bottleneck:

| Method | Axial calls / adapter | Pointwise calls / adapter | Total calls / adapter |
|---|---:|---:|---:|
| V8 closed-form | 4 | 2 | 6 |
| MLQ support 8 + pointwise | 2 | 2 | 4 |
| SCDQ support 4 + pointwise | 2 | 2 | 4 |
| **SCDQ final** | **2** | **0** | **2** |

For ResNet-18 with eight inserted adapters, this changes 48 adapter convolution calls to 16.

Trainable parameter counts for Caltech101/ResNet-18:

| Variant | Trainable parameters |
|---|---:|
| V8 current | 64,317 |
| MLQ support 8 + pointwise | 64,029 |
| SCDQ support 4 + pointwise | 63,789 |
| MLQ support 8 core | 53,021 |
| **SCDQ final** | **52,781** |

## 6. Tests completed

Full repository test result:

```text
304 passed, 1 skipped, 10 subtests passed
```

Deterministic theory validation:

| Check | Result |
|---|---:|
| Dyadic quotient rank | 5 |
| Null-direction residual | 0.0 |
| Legacy → MLQ8 forward max error | 2.3842e-7 |
| Legacy → MLQ8 input-gradient max error | 2.9104e-11 |
| Maximum projected joint axis `l1` norm | 1.0 |
| Legacy core parameters in 32-channel unit test | 55 |
| MLQ8 core parameters | 41 |
| SCDQ4 core parameters | 33 |

The exact legacy-to-MLQ8 equivalence was tested with a common replicate boundary operator. SCDQ4 is the deliberate spectral-closure constraint and is therefore not claimed to reproduce an arbitrary nonzero `β8` legacy model.

## 7. Structural latency test completed

A CPU structural benchmark was run with PyTorch 2.10, five threads, batch size 4, input `224×224`, and ResNet-18. These are not publication GPU numbers; they test whether the mathematical reduction actually lowers executed model latency.

| Variant | Median ms/image | FPS | Latency reduction vs V8 | Throughput gain |
|---|---:|---:|---:|---:|
| V8 closed-form + pointwise | 26.429 | 37.84 | – | – |
| MLQ8 + pointwise | 24.102 | 41.49 | 8.80% | 9.65% |
| SCDQ4 + pointwise | 22.882 | 43.70 | 13.42% | 15.50% |
| MLQ8 core | 21.932 | 45.59 | 17.01% | 20.50% |
| **SCDQ4 core final** | **21.255** | **47.05** | **19.58%** | **24.34%** |

This validates the direction of the runtime improvement without using compilation, quantization, pruning, TensorRT, or a custom kernel. The final paper must still report the canonical GPU profiler results before making a numerical speed claim against SSF, Conv-Adapter, or LoRA-Conv.

## 8. Required three-seed ablations

Thirty ready-to-run YAML files are provided in:

```text
configs/experiments/table_14_15_mlq_ablation/
```

Run the following variants over seeds 0, 1, and 2:

1. `legacy_v8`: current method and current pointwise branch.
2. `mlq8_pointwise`: quotient only; tests removal of non-identifiability.
3. `scdq4_pointwise`: adds spectral closure while retaining pointwise mixing.
4. `mlq8_core`: tests removal of the optional pointwise block.
5. `scdq4_core_final`: complete final proposal.
6. `scdq4_core_reflect`: boundary-rule ablation.
7. `scdq4_core_h`: height-only ablation.
8. `scdq4_core_w`: width-only ablation.
9. `scdq4_core_group8`: finer coefficient sharing.
10. `scdq4_core_group32`: stronger coefficient sharing.

Report for every variant:

- test Acc1 and Acc5;
- test loss;
- best validation accuracy and epoch;
- trainable and total parameters;
- FLOPs;
- canonical latency and FPS;
- total and per-epoch training time;
- peak training and inference memory;
- mean ± standard deviation over three independent seeds.

## 9. Acceptance criteria

Do not replace the manuscript method solely because the CPU structural benchmark is positive. Accept SCDQ as the final paper method only when the three-seed GPU run satisfies all of the following:

1. Mean test Acc1 is not lower than the current `90.465%` by more than 0.20 percentage points.
2. Mean latency is lower than the PDF value `2.099 ms/image` and the current rerun value `3.047 ms/image`.
3. Throughput exceeds the PDF value `476.31 FPS`.
4. The same profiler, batch size, warm-up, iteration count, precision, and GPU are used for every method.
5. SCDQ is compared particularly against SSF, Conv-Adapter, and LoRA-Conv, not only against the older DT1D realization.

A stronger publication result would require SCDQ to approach or exceed SSF accuracy while materially reducing the gap in latency. That outcome cannot be guaranteed without running the supplied three-seed GPU experiments.
