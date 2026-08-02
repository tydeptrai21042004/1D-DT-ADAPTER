from __future__ import annotations

import torch

from models.dt1d_adapter import DT1DAdapter


def _hosq(C: int = 32, *, rank4: int = 1, rank8: int = 2, no_pw: bool = True) -> DT1DAdapter:
    return DT1DAdapter(
        C=C,
        M=1,
        axis="hw",
        alpha_group=32,
        no_pw=no_pw,
        pw_ratio=32,
        pw_groups=4,
        use_bn=False,
        gate_init=0.01,
        padding_mode="reflect",
        dilations=(1, 2, 4),
        scale_adaptive=True,
        separate_axis_kernels=True,
        minimal_quotient_realization=False,
        quotient_support_cap=8,
        hosq_realization=True,
        hosq_subgroup_size=8,
        hosq_rank4=rank4,
        hosq_rank8=rank8,
    )


def _mlq32(C: int = 32) -> DT1DAdapter:
    return DT1DAdapter(
        C=C,
        M=1,
        axis="hw",
        alpha_group=32,
        no_pw=True,
        gate_init=0.01,
        padding_mode="reflect",
        dilations=(1, 2, 4),
        scale_adaptive=True,
        separate_axis_kernels=True,
        minimal_quotient_realization=True,
        quotient_support_cap=8,
    )


def test_hosq_four_subgroup_basis_is_zero_mean_and_orthonormal():
    model = _hosq(C=32)
    basis = model.hosq_basis[0, :4, :3].to(torch.float64)
    torch.testing.assert_close(basis.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-7, rtol=0)
    torch.testing.assert_close(basis.T @ basis, torch.eye(3, dtype=torch.float64), atol=1e-7, rtol=0)


def test_hosq_zero_detail_exactly_reduces_to_mlq8_group32():
    torch.manual_seed(17)
    mlq = _mlq32(C=64)
    hosq = _hosq(C=64, rank4=1, rank8=2)
    with torch.no_grad():
        mlq.quotient_beta.normal_(0.0, 0.2)
        mlq.gate.fill_(0.13)
    hosq.initialize_hosq_from_mlq(mlq)
    x = torch.randn(2, 64, 23, 21)
    torch.testing.assert_close(hosq(x), mlq(x), rtol=2e-6, atol=2e-6)


def test_hosq_detail_atoms_are_zero_dc_before_projection():
    model = _hosq(C=32)
    with torch.no_grad():
        model.quotient_beta.zero_()
        model.hosq_detail4.normal_(0.0, 0.3)
        model.hosq_detail8.normal_(0.0, 0.3)
    kernels = model._build_normalized_hosq_kernels(torch.device("cpu"), torch.float64)
    torch.testing.assert_close(
        kernels.squeeze(2).sum(dim=-1),
        torch.zeros(model.num_axes, model.C, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


def test_hosq_joint_axis_l1_projection_is_nonexpansive():
    torch.manual_seed(19)
    model = _hosq(C=64)
    with torch.no_grad():
        model.quotient_beta.normal_(0.0, 3.0)
        model.hosq_detail4.normal_(0.0, 3.0)
        model.hosq_detail8.normal_(0.0, 3.0)
    kernels = model._build_normalized_hosq_kernels(torch.device("cpu"), torch.float64)
    joint = kernels.squeeze(2).abs().sum(dim=-1).sum(dim=0)
    assert torch.all(joint <= 1.0 + 1e-12)


def test_hosq_uses_one_convolution_per_axis():
    model = _hosq(C=32)
    calls = []
    original = model._conv_axis

    def counted(x, axis_name, w1d, dilation):
        calls.append((axis_name, int(w1d.shape[-1]), int(dilation)))
        return original(x, axis_name, w1d, dilation)

    model._conv_axis = counted
    _ = model(torch.randn(1, 32, 20, 20))
    assert calls == [("h", 17, 1), ("w", 17, 1)]


def test_hosq_forward_backward_has_finite_gradients():
    torch.manual_seed(23)
    model = _hosq(C=64)
    x = torch.randn(2, 64, 16, 16, requires_grad=True)
    loss = model(x).square().mean()
    loss.backward()
    for p in (model.quotient_beta, model.hosq_detail4, model.hosq_detail8, model.gate):
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_hosq_resnet18_adapter_budget_is_968_parameters():
    channels = [64, 64, 128, 128, 256, 256, 512, 512]
    total = sum(sum(p.numel() for p in _hosq(C=c).parameters() if p.requires_grad) for c in channels)
    assert total == 968
    assert 51_813 + total == 52_781


def test_hosq_remainder_channels_run_without_unused_shape_errors():
    # EfficientNet-like channel counts exercise partial coarse groups.
    for channels in (16, 24, 40, 80, 112, 192, 320):
        model = _hosq(C=channels)
        y = model(torch.randn(1, channels, 9, 11))
        assert y.shape == (1, channels, 9, 11)
