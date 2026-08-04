from __future__ import annotations

import math

import torch

from models.dt1d_adapter import DT1DAdapter
from models.hosq_lite_c1_adapter import HOSQLiteC1Adapter


def legacy(C=64, group=16, *, no_pw=True, padding="replicate"):
    return DT1DAdapter(
        C=C, M=1, h=1, axis="hw", alpha_group=group, no_pw=no_pw,
        gate_init=0.01, padding_mode=padding, dilations=(1, 2, 4),
        scale_adaptive=True, separate_axis_kernels=True,
    )


def lite(C=64, group=16, *, basis="orth", components="both", axis="hw"):
    return HOSQLiteC1Adapter(
        C=C, axis=axis, alpha_group=group, gate_init=0.01,
        padding_mode="replicate", detail_basis=basis,
        detail_components=components, contrast_split=max(1, group // 2),
    )


def test_no_detail_is_exact_coarse_submodel():
    torch.manual_seed(1)
    a = lite(components="none")
    b = lite(components="both")
    with torch.no_grad():
        a.quotient_beta.normal_(0, 0.05)
        a.gate.fill_(0.2)
        b.quotient_beta.copy_(a.quotient_beta)
        b.gate.copy_(a.gate)
        b.detail_eta.zero_()
    x = torch.randn(2, 64, 19, 21)
    torch.testing.assert_close(a(x), b(x), rtol=2e-6, atol=2e-6)


def test_exact_legacy_warm_start_at_all_resnet_sizes():
    torch.manual_seed(2)
    a = legacy()
    b = lite()
    with torch.no_grad():
        a.alpha.normal_(0, 0.3)
        a.axis_scale_logits.normal_(0, 0.4)
        a.gate.fill_(0.17)
    b.initialize_from_dt1d(a)
    for h, w in ((7, 7), (8, 9), (14, 14), (28, 31), (56, 56)):
        x = torch.randn(2, 64, h, w)
        torch.testing.assert_close(a(x), b(x), rtol=3e-6, atol=3e-6)


def test_orthogonal_atoms_are_zero_dc_and_orthonormal():
    atoms = lite().spectral_atoms.double()
    torch.testing.assert_close(atoms.sum(-1), torch.zeros(2, dtype=torch.float64), atol=1e-7, rtol=0)
    torch.testing.assert_close(atoms @ atoms.T, torch.eye(2, dtype=torch.float64), atol=1e-7, rtol=0)


def test_raw_atoms_span_same_space_as_orthogonal_atoms():
    raw = lite(basis="raw").spectral_atoms.double().T
    orth = lite(basis="orth").spectral_atoms.double().T
    assert torch.linalg.matrix_rank(raw).item() == 2
    projection = raw @ torch.linalg.lstsq(raw, orth).solution
    torch.testing.assert_close(projection, orth, atol=2e-7, rtol=0)


def test_channel_contrasts_are_zero_mean_and_unit_norm():
    for C in (9, 15, 16, 17, 24, 31, 40, 63, 64):
        m = lite(C)
        values = m.channel_contrast
        for group in range(m.num_alpha_groups):
            start = group * m.alpha_group
            end = min(C, (group + 1) * m.alpha_group)
            if bool(m.valid_contrast_group[group]):
                torch.testing.assert_close(values[start:end].sum(), torch.tensor(0.0), atol=1e-7, rtol=0)
                torch.testing.assert_close(torch.linalg.vector_norm(values[start:end]), torch.tensor(1.0), atol=1e-7, rtol=0)


def test_seven_directions_form_identifiable_direct_sum():
    n = 16
    u = torch.ones(n, dtype=torch.float64) / math.sqrt(n)
    h = lite(16).channel_contrast.double()
    spatial = []
    delta0 = torch.zeros(17, dtype=torch.float64)
    delta0[8] = 1
    spatial.append(delta0)
    for offset in (1, 2, 4, 8):
        atom = torch.zeros(17, dtype=torch.float64)
        atom[8 - offset] = atom[8 + offset] = 1 / math.sqrt(2)
        spatial.append(atom)
    q = lite(16).spectral_atoms.double()
    columns = [torch.outer(u, atom).reshape(-1) for atom in spatial]
    columns += [torch.outer(h, q[index]).reshape(-1) for index in range(2)]
    matrix = torch.stack(columns, dim=1)
    torch.testing.assert_close(matrix.T @ matrix, torch.eye(7, dtype=torch.float64), atol=2e-7, rtol=0)
    assert torch.linalg.matrix_rank(matrix).item() == 7


def test_detail_is_zero_dc_and_zero_group_mean_before_projection():
    m = lite()
    with torch.no_grad():
        m.quotient_beta.zero_()
        m.detail_eta.normal_(0, 0.2)
    kernels = m.build_kernels(torch.device("cpu"), torch.float64, project=False).squeeze(2)
    torch.testing.assert_close(kernels.sum(-1), torch.zeros_like(kernels.sum(-1)), atol=2e-7, rtol=0)
    for group in range(m.num_alpha_groups):
        start = group * m.alpha_group
        end = min(m.C, (group + 1) * m.alpha_group)
        torch.testing.assert_close(
            kernels[:, start:end].mean(1),
            torch.zeros_like(kernels[:, start:end].mean(1)),
            atol=1e-12,
            rtol=0,
        )


def test_joint_axis_l1_projection_is_nonexpansive():
    torch.manual_seed(3)
    m = lite()
    with torch.no_grad():
        m.quotient_beta.normal_(0, 3)
        m.detail_eta.normal_(0, 3)
    kernels = m.build_kernels(torch.device("cpu"), torch.float64).squeeze(2)
    assert torch.all(kernels.abs().sum(-1).sum(0) <= 1 + 1e-12)


def test_detail_annihilates_constant_input():
    m = lite()
    with torch.no_grad():
        m.quotient_beta.zero_()
        m.detail_eta.normal_(0, 0.1)
        m.gate.fill_(1)
    x = torch.ones(1, 64, 13, 11)
    torch.testing.assert_close(m(x), x, atol=2e-7, rtol=0)


def test_zero_initialized_details_receive_gradient():
    torch.manual_seed(4)
    m = lite()
    with torch.no_grad():
        m.quotient_beta.normal_(0, 0.03)
        m.detail_eta.zero_()
        m.gate.fill_(0.01)
    x = torch.randn(2, 64, 15, 17, requires_grad=True)
    m(x).square().mean().backward()
    assert m.detail_eta.grad is not None
    assert torch.isfinite(m.detail_eta.grad).all()
    assert float(m.detail_eta.grad.abs().sum()) > 0


def test_one_convolution_per_enabled_axis():
    m = lite()
    calls = []
    original = m._conv_axis

    def counted(x, axis, weight, dilation):
        calls.append((axis, int(weight.shape[-1]), int(dilation)))
        return original(x, axis, weight, dilation)

    m._conv_axis = counted
    m(torch.randn(1, 64, 20, 20))
    assert calls == [("h", 17, 1), ("w", 17, 1)]


def test_resnet18_adapter_budget_is_1688():
    channels = (64, 64, 128, 128, 256, 256, 512, 512)
    total = sum(sum(p.numel() for p in lite(c).parameters() if p.requires_grad) for c in channels)
    assert total == 1688


def test_all_ablation_modes_forward_and_backward():
    for basis in ("orth", "raw"):
        for components in ("both", "offset4", "offset8", "none"):
            m = lite(16, basis=basis, components=components)
            x = torch.randn(2, 16, 9, 9, requires_grad=True)
            loss = m(x).square().mean()
            loss.backward()
            assert torch.isfinite(loss)
            assert x.grad is not None and torch.isfinite(x.grad).all()


def test_state_dict_round_trip_and_bfloat16():
    torch.manual_seed(5)
    a = lite(16)
    with torch.no_grad():
        a.quotient_beta.normal_()
        a.detail_eta.normal_()
        a.gate.fill_(0.2)
    b = lite(16)
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 16, 11, 12)
    torch.testing.assert_close(a(x), b(x))

    x = torch.randn(2, 16, 9, 9, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = b(x).float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
