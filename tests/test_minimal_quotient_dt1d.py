import torch

from models.dt1d_adapter import DT1DAdapter


def _legacy(no_pw: bool = True) -> DT1DAdapter:
    return DT1DAdapter(
        C=16,
        M=1,
        axis="hw",
        alpha_group=4,
        no_pw=no_pw,
        pw_ratio=4,
        pw_groups=2,
        use_bn=False,
        gate_init=0.2,
        padding_mode="replicate",
        dilations=(1, 2, 4),
        scale_adaptive=True,
        separate_axis_kernels=True,
        exact_cost_realization=False,
        closed_form_dyadic_realization=False,
    )


def _mlq(cap: int = 8, no_pw: bool = True) -> DT1DAdapter:
    return DT1DAdapter(
        C=16,
        M=1,
        axis="hw",
        alpha_group=4,
        no_pw=no_pw,
        pw_ratio=4,
        pw_groups=2,
        use_bn=False,
        gate_init=0.2,
        padding_mode="replicate",
        dilations=(1, 2, 4),
        scale_adaptive=True,
        separate_axis_kernels=True,
        minimal_quotient_realization=True,
        quotient_support_cap=cap,
    )


def test_dyadic_quotient_has_rank_five_and_exact_null_direction():
    a = DT1DAdapter.dyadic_quotient_matrix(dtype=torch.float64)
    assert torch.linalg.matrix_rank(a).item() == 5
    null = torch.tensor([0.0, 1.0, -1.0, -1.0, 1.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(a @ null, torch.zeros(5, dtype=torch.float64))


def test_mlq8_exactly_reproduces_legacy_forward_and_input_gradient():
    torch.manual_seed(3)
    legacy = _legacy(no_pw=True)
    with torch.no_grad():
        legacy.alpha.normal_(0.0, 0.5)
        legacy.axis_scale_logits.normal_(0.0, 0.4)
        legacy.gate.fill_(0.17)
    mlq = _mlq(cap=8, no_pw=True)
    mlq.initialize_quotient_from_legacy(legacy)

    x1 = torch.randn(2, 16, 19, 21, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1 = legacy(x1)
    y2 = mlq(x2)
    torch.testing.assert_close(y1, y2, rtol=2e-6, atol=2e-6)

    y1.square().mean().backward()
    y2.square().mean().backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=2e-6, atol=2e-6)


def test_mlq8_exactly_reproduces_bias_free_pointwise_path():
    torch.manual_seed(7)
    legacy = _legacy(no_pw=False)
    with torch.no_grad():
        legacy.alpha.normal_(0.0, 0.4)
        legacy.axis_scale_logits.normal_(0.0, 0.3)
        legacy.gate.fill_(0.11)
        for p in legacy.pw.parameters():
            p.normal_(0.0, 0.1)
    mlq = _mlq(cap=8, no_pw=False)
    mlq.initialize_quotient_from_legacy(legacy)
    x = torch.randn(2, 16, 20, 20)
    torch.testing.assert_close(legacy(x), mlq(x), rtol=2e-6, atol=2e-6)


def test_scdq4_kernel_is_symmetric_closed_and_jointly_nonexpansive():
    torch.manual_seed(11)
    model = _mlq(cap=4, no_pw=True)
    with torch.no_grad():
        model.quotient_beta.normal_(0.0, 2.0)
    beta = model._normalized_quotient_beta(torch.device("cpu"), torch.float64)
    per_axis = beta[..., 0].abs() + 2.0 * beta[..., 1:].abs().sum(dim=-1)
    assert torch.all(per_axis.sum(dim=0) <= 1.0 + 1e-12)

    w = model._build_minimal_quotient_kernel_1d(beta, 0, torch.device("cpu"), torch.float64)
    assert w.shape[-1] == 9
    torch.testing.assert_close(w, torch.flip(w, dims=(-1,)))


def test_minimal_quotient_uses_one_convolution_per_axis():
    model = _mlq(cap=4, no_pw=True)
    calls = []
    original = model._conv_axis

    def counted(x, axis_name, w1d, dilation):
        calls.append((axis_name, int(w1d.shape[-1]), dilation))
        return original(x, axis_name, w1d, dilation)

    model._conv_axis = counted
    _ = model(torch.randn(1, 16, 12, 12))
    assert calls == [("h", 9, 1), ("w", 9, 1)]


def test_scdq4_has_fewer_trainable_parameters_than_legacy_core():
    legacy = _legacy(no_pw=True)
    final = _mlq(cap=4, no_pw=True)
    legacy_n = sum(p.numel() for p in legacy.parameters() if p.requires_grad)
    final_n = sum(p.numel() for p in final.parameters() if p.requires_grad)
    assert final_n < legacy_n
