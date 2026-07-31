# DT1D-Adapter: Math-First Exact Cost Realization Audit

## Final conclusion

The speed revision is **not** a dataset, optimizer, baseline, batching, caching, compiler, or measurement-pipeline trick. The final V3 package changes only the mathematical realization of the static axis--scale DT1D operator.

The learned operator, trainable parameters, checkpoint tensors, kernel normalization, routing weights, pointwise block, residual gate, padding rule, optimizer trajectory, predictions, and losses are preserved up to floating-point summation order.

The key limitation must be stated honestly: this is a genuine **operator-realization optimization theorem**, not a new hypothesis class. That distinction is what allows a strong no-regression guarantee. A non-equivalent model change could not guarantee unchanged metrics without retraining every benchmark.

---

## 1. Scope audit: what was and was not changed

Comparison against the uploaded repository shows:

- Existing source files changed: `models/hcc_adapter.py`, `main.py`.
- Existing source files unchanged: 50.
- Baseline implementations changed: 0.
- Dataset files changed: 0.
- Training-engine files changed: 0.
- Files removed: 0.

`main.py` only exposes an ablation switch and passes it to DT1D:

```text
--hcc_exact_cost_realization true   # final mathematical realization
--hcc_exact_cost_realization false  # uploaded branch realization
```

No result-producing setting is changed automatically.

### Explicitly removed from the earlier V2 idea

The final V3 does **not** use:

- inference-kernel caching;
- vectorized replacement of the uploaded kernel builder;
- vectorized replacement of group-to-channel expansion;
- TorchScript, compilation, quantization, pruning, distillation, or mixed precision;
- altered batch size, input resolution, data loader, optimizer, epochs, or baseline code.

Both before and after timing paths call the same class, the same kernel builder, the same channel expansion, the same pointwise block, and the same PyTorch convolution primitive. The only switch is the exact mathematical branch partition.

---

## 2. Before: uploaded branch-form algorithm

Let the feature tensor be `x`, the enabled axes be \(u\in\mathcal A\), the dilation scales be \(d_s\), and the separately normalized shifted kernels be \(\widetilde k_{u,s}\). The static router gives

\[
\pi_{u,s}\ge 0,
\qquad
\sum_{u,s}\pi_{u,s}=1.
\]

Define the axial shift operator

\[
(\mathcal T_{u,d,k}x)[n]
=
\sum_{r=-R}^{R} k[r]x[n+d r e_u],
\qquad R=M+1.
\]

The uploaded scale-adaptive implementation evaluates

\[
\boxed{
\mathcal F_{\mathrm{before}}x
=
\sum_{u\in\mathcal A}\sum_{s=1}^{S}
\pi_{u,s}\mathcal T_{u,d_s,\widetilde k_{u,s}}x.
}
\]

### Before algorithm

```text
weights = softmax(axis_scale_logits / temperature)
y = 0
for axis u:
    for scale s:
        k = build_and_normalize_shifted_kernel(alpha[u,s])
        response = axial_conv(x, k, dilation=d_s)
        y += weights[u,s] * response
out = x + residual_scale * gate * pointwise(y)
```

For two axes and three scales, the branch realization invokes six axial convolutions per inserted adapter.

---

## 3. After: exact minimum-cost Laurent realization

### 3.1 Exact group aggregation

For a boundary-compatible group of scales \(G\), define

\[
K_{u,G}[t]
=
\sum_{s\in G}\sum_{r=-R}^{R}
\pi_{u,s}\widetilde k_{u,s}[r]\,
\mathbf 1_{\{t=d_s r\}}.
\]

Then

\[
\begin{aligned}
(\mathcal T_{u,1,K_{u,G}}x)[n]
&=
\sum_t K_{u,G}[t]x[n+t e_u]\\
&=
\sum_{s\in G}\sum_{r=-R}^{R}
\pi_{u,s}\widetilde k_{u,s}[r]x[n+d_s r e_u]\\
&=
\sum_{s\in G}
\pi_{u,s}\mathcal T_{u,d_s,\widetilde k_{u,s}}x[n].
\end{aligned}
\]

Therefore, branch aggregation is an exact operator identity, not an approximation.

### 3.2 Cost minimization

Let \(\mathcal P\) be a partition of the scale indices. A singleton retains the uploaded compact dilated kernel, with cost

\[
c(\{s\})=2M+3.
\]

A multi-scale group is represented by a unit-dilation dense kernel extending to the largest offset, with cost

\[
c(G)=2(M+1)\max_{s\in G}d_s+1,
\qquad |G|>1.
\]

The final realization solves

\[
\boxed{
\mathcal P^*
=
\arg\min_{\mathcal P}^{\mathrm{lex}}
\left(
\sum_{G\in\mathcal P}c(G),
|\mathcal P|
\right),
}
\]

where the first objective is dense axial tap count and the second is convolution-launch count.

After sorting scales by dilation, the dynamic-programming recurrence is

\[
J(i)=
\min_{0\le j<i}
\left[J(j)+c(\{j,\ldots,i-1\})\right].
\]

The implementation was checked against brute-force enumeration of all set partitions for multiple scale sets and radii; the selected score always matched the global optimum.

### 3.3 After algorithm

```text
weights = softmax(axis_scale_logits / temperature)
y = 0
for axis u:
    P_star = globally_minimum_cost_exact_partition(scales, boundary_class)
    for group G in P_star:
        if G is a singleton:
            k = the same uploaded normalized shifted kernel
            y += weights[u,s] * axial_conv(x, k, dilation=d_s)
        else:
            K = exact Laurent aggregation of the same normalized kernels and weights
            y += axial_conv(x, K, dilation=1)
out = x + residual_scale * gate * pointwise(y)
```

The pointwise block, residual gate, and task pipeline are unchanged.

---

## 4. Exact default before/after calculation

For the manuscript default

\[
M=1,
\qquad R=M+1=2,
\qquad \mathcal D=\{1,2,4\},
\qquad K_{\mathrm{eff}}=2M+3=5,
\]

the candidate costs per axis include:

| Partition | Dense taps | Calls |
|---|---:|---:|
| \(\{1\},\{2\},\{4\}\) | \(5+5+5=15\) | 3 |
| \(\{1,2,4\}\) | \(2\cdot2\cdot4+1=17\) | 1 |
| \(\{1,2\},\{4\}\) | \(9+5=14\) | 2 |

Hence

\[
\boxed{
\mathcal P^*=\{\{1,2\},\{4\}\}.
}
\]

For both axes:

| Quantity | Before | After | Change |
|---|---:|---:|---:|
| Axial convolution calls | 6 | 4 | \(-33.3\%\) |
| Dense axial taps | 30 | 28 | \(-6.67\%\) |
| Trainable parameters | unchanged | unchanged | 0 |
| Checkpoint tensors | unchanged | unchanged | 0 |

Runtime instrumentation confirmed six versus four calls, while both paths built the same six normalized scale kernels. Thus, the speed difference is not caused by a faster kernel-building routine.

---

## 5. Preservation theorems

### Theorem 1: exact output preservation

For every input, parameter value, and compatible boundary class,

\[
\boxed{
\mathcal F_{\mathrm{after}}x
=
\mathcal F_{\mathrm{before}}x.
}
\]

Differences in finite-precision tests arise only from summation order.

### Theorem 2: no cost regression within the declared model

The all-singleton partition is always feasible. Therefore,

\[
\sum_{G\in\mathcal P^*}c(G)
\le
S(2M+3),
\]

and every partition contains at most \(S\) groups, so

\[
|\mathcal P^*|\le S.
\]

The selected exact realization cannot increase the declared dense axial tap count or axial convolution count.

### Theorem 3: stability is preserved

Because kernels are normalized before aggregation,

\[
\|\widetilde k_{u,s}\|_1\le 1.
\]

Then

\[
\begin{aligned}
\sum_{u,G}\|K_{u,G}\|_1
&\le
\sum_{u,s}\pi_{u,s}\|\widetilde k_{u,s}\|_1\\
&\le
\sum_{u,s}\pi_{u,s}=1.
\end{aligned}
\]

Thus, the existing non-expansiveness result remains

\[
\|\mathcal F_{\mathrm{after}}x\|_p
\le
\|x\|_p.
\]

### Theorem 4: spectral response is preserved

For each exact group,

\[
\widehat K_{u,G}(\omega)
=
\sum_{s\in G}
\pi_{u,s}\widehat{\widetilde k}_{u,s}(d_s\omega).
\]

Summing over groups recovers the uploaded global axis--scale multiplier exactly. The cosine-modulated shifted-kernel interpretation and Fredholm-type argument therefore remain unchanged.

### Gradient preservation

The after operator is the same differentiable function of \(x\), \(\alpha\), router logits, pointwise parameters, and the residual gate. Consequently, exact-real arithmetic gives identical gradients. Finite-precision differences only reflect a changed summation tree.

---

## 6. Boundary correctness

The uploaded reflect-padding code falls back to replicate padding when a branch pad is not smaller than the feature dimension. A reflect branch and a replicate-fallback branch are different boundary operators and must not be fused.

V3 first divides scales into boundary-equivalence classes and optimizes only within each class. Tests include very small \(3\times4\) feature maps that activate the fallback behavior.

---

## 7. Numerical regression results

### Automated tests

```text
226 passed
```

This includes the original focused DT1D tests and the new mathematical-realization suite. The broader repository suite could not be collected in this container because the optional `timm` dependency is absent; no claim is made that those unrelated tests ran.

### Exhaustive comparison against the uploaded source

A total of 720 configurations were tested across:

- \(M\in\{0,1,2,3\}\);
- five dilation sets;
- zero, replicate, and reflect padding;
- normal and very small feature maps;
- float32 and float64;
- pointwise block enabled and disabled.

| Quantity | Maximum absolute difference |
|---|---:|
| Output | \(4.7684\times10^{-7}\) |
| Input gradient | \(1.4901\times10^{-8}\) |
| Parameter gradient | \(1.1921\times10^{-7}\) |
| Loss | \(4.7684\times10^{-7}\) |
| Minimum prediction agreement | 100% |
| Cost-regression violations | 0 / 720 |

### Fifty-step AdamW trajectory

| Quantity | Result |
|---|---:|
| Maximum loss difference | 0.0 |
| Maximum logit difference | \(2.9802\times10^{-8}\) |
| Minimum top-1 agreement | 100% |
| Maximum parameter difference after 50 steps | \(1.1921\times10^{-7}\) |

### Synthetic metric regression

| Metric | Before | After |
|---|---:|---:|
| Accuracy | 0.144000 | 0.144000 |
| Cross-entropy | 1.990323 | 1.990323 |
| Prediction agreement | — | 100% |

These synthetic metrics validate functional preservation. They are not substitutes for rerunning the manuscript datasets.

---

## 8. Fair timing results without caching or unrelated code optimization

Both timing variants use the same class and differ only in `exact_cost_realization`.

### Adapter-level CPU benchmark

| Feature shape per sample | Inference speedup | Forward/backward speedup |
|---|---:|---:|
| \(64\times56\times56\) | 1.248x | 1.446x |
| \(128\times28\times28\) | 1.156x | 1.480x |
| \(256\times14\times14\) | 1.229x | 1.407x |
| \(512\times7\times7\) | 1.136x | 1.232x |

### Controlled ResNet-18 CPU benchmark, batch 2, 224x224

| Quantity | Before | After | Speedup |
|---|---:|---:|---:|
| Inference | 69.372 ms | 65.091 ms | 1.066x |
| Forward/backward | 186.711 ms | 173.091 ms | 1.079x |
| Top-1 agreement | — | 100% | — |
| Maximum logit difference | — | \(1.4305\times10^{-6}\) | — |

The end-to-end improvement is smaller because frozen-backbone computation dominates. CPU timings are controlled evidence, not the final manuscript GPU table. GPU latency, peak memory, and dataset-level results must still be rerun in the original environment.

---

## 9. Exact publication claim

Recommended claim:

> We derive an exact minimum-cost realization of the static axis--scale DT1D operator. By representing compatible dilated branches as Laurent-polynomial kernel groups and selecting a globally minimum-cost partition, the method preserves the learned mapping, stability bounds, spectral multiplier, parameters, and checkpoints while reducing the default axial realization from six to four convolution calls and from 30 to 28 dense taps per adapter. Controlled tests against the uploaded branch implementation show numerical equivalence and lower execution time without changing the training or evaluation pipeline.

Do not claim:

- a new accuracy-improving hypothesis class;
- universal fastest inference;
- unchanged final benchmark values without rerunning saved checkpoints;
- GPU speed or memory gains based only on the included CPU tests.

---

## 10. Required final paper experiment

For every saved manuscript checkpoint, evaluate the same checkpoint twice:

```text
Before: --hcc_exact_cost_realization false
After:  --hcc_exact_cost_realization true
```

Keep dataset, split, input resolution, batch size, device, precision, warm-up, repetitions, and synchronization identical. Accuracy and all task metrics should match within numerical tolerance; latency and training-step time are the only intended changes.
