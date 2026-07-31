"""Regression tests for the exact minimum-cost mathematical DT1D realization."""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/mnt/data/dt1d_repo/1D-DT-ADAPTER-main/models/hcc_adapter.py')


def load_class(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.DT1DAdapter


OriginalDT1D = load_class(ORIGINAL, 'dt1d_uploaded_original')
OptimizedDT1D = load_class(ROOT / 'models' / 'hcc_adapter.py', 'dt1d_math_optimized')


def kwargs(m=1, dilations=(1, 2, 4), padding='reflect', c=7, no_pw=False):
    return dict(
        C=c,
        M=m,
        dilations=dilations,
        scale_adaptive=True,
        axis='hw',
        alpha_group=3,
        no_pw=no_pw,
        pw_ratio=4,
        pw_groups=1,
        gate_init=0.23,
        padding_mode=padding,
        separate_axis_kernels=True,
    )


def seed_nontrivial(model, seed=7):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.alpha.copy_(torch.randn(model.alpha.shape, generator=generator))
        model.axis_scale_logits.copy_(
            torch.randn(model.axis_scale_logits.shape, generator=generator)
        )
        model.gate.fill_(0.31)
        for parameter in model.pw.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)


def copy_original_state(original, optimized):
    result = optimized.load_state_dict(original.state_dict(), strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys


def test_no_parameter_or_checkpoint_change():
    original = OriginalDT1D(**kwargs())
    optimized = OptimizedDT1D(**kwargs(), exact_cost_realization=True)
    assert tuple(original.state_dict().keys()) == tuple(optimized.state_dict().keys())
    assert sum(p.numel() for p in original.parameters()) == sum(
        p.numel() for p in optimized.parameters()
    )
    for key, value in original.state_dict().items():
        assert value.shape == optimized.state_dict()[key].shape


def test_default_before_after_cost_is_6_to_4_calls_and_30_to_28_taps():
    model = OptimizedDT1D(**kwargs(), exact_cost_realization=True)
    x = torch.randn(2, 7, 56, 56)
    assert model._exact_partition_for_axis(x, 'h') == ((0, 1), (2,))
    per_axis = model.exact_realization_cost(x, 'h')
    assert per_axis == {
        'before_calls': 3,
        'after_calls': 2,
        'before_dense_taps': 15,
        'after_dense_taps': 14,
    }


@pytest.mark.parametrize('m', [0, 1, 2, 3])
@pytest.mark.parametrize(
    'dilations',
    [(1,), (1, 2), (1, 2, 4), (1, 3, 5), (2, 4), (1, 2, 3, 4)],
)
@pytest.mark.parametrize('padding', ['zeros', 'replicate', 'reflect'])
@pytest.mark.parametrize('shape', [(19, 23), (7, 8), (3, 4)])
def test_exact_mapping_and_gradients_against_uploaded_source(
    m, dilations, padding, shape
):
    common = kwargs(m=m, dilations=dilations, padding=padding, c=7, no_pw=False)
    original = OriginalDT1D(**common)
    optimized = OptimizedDT1D(**common, exact_cost_realization=True)
    seed_nontrivial(original, seed=100 + m + sum(dilations))
    copy_original_state(original, optimized)

    generator = torch.Generator().manual_seed(900 + m + shape[0])
    x0 = torch.randn((2, 7, *shape), generator=generator)
    x_original = x0.clone().requires_grad_(True)
    x_optimized = x0.clone().requires_grad_(True)

    y_original = original(x_original)
    y_optimized = optimized(x_optimized)
    torch.testing.assert_close(y_optimized, y_original, rtol=5e-6, atol=8e-7)

    target = torch.randn(y_original.shape, generator=generator)
    loss_original = torch.nn.functional.mse_loss(y_original, target)
    loss_optimized = torch.nn.functional.mse_loss(y_optimized, target)
    loss_original.backward()
    loss_optimized.backward()

    torch.testing.assert_close(
        x_optimized.grad, x_original.grad, rtol=8e-6, atol=1e-7
    )
    for (name_o, p_o), (name_n, p_n) in zip(
        original.named_parameters(), optimized.named_parameters()
    ):
        assert name_o == name_n
        assert (p_o.grad is None) == (p_n.grad is None)
        if p_o.grad is not None:
            torch.testing.assert_close(p_n.grad, p_o.grad, rtol=1e-5, atol=1e-7)

    for axis in ('h', 'w'):
        cost = optimized.exact_realization_cost(x_optimized, axis)
        assert cost['after_calls'] <= cost['before_calls']
        assert cost['after_dense_taps'] <= cost['before_dense_taps']


def all_partitions(items):
    if not items:
        yield []
        return
    first, *rest = items
    for partition in all_partitions(rest):
        yield [[first]] + [group[:] for group in partition]
        for index in range(len(partition)):
            candidate = [group[:] for group in partition]
            candidate[index] = [first] + candidate[index]
            yield candidate


def test_dynamic_program_matches_global_partition_optimum():
    for m in (0, 1, 2, 3):
        for dilations in (
            (1, 2, 4),
            (1, 3, 5),
            (1, 2, 3, 4),
            (2, 3, 7, 8, 9),
        ):
            model = OptimizedDT1D(
                **kwargs(m=m, dilations=dilations, padding='zeros'),
                exact_cost_realization=True,
            )
            selected = model._optimal_exact_partition(range(len(dilations)))

            def score(partition):
                return (
                    sum(model._group_dense_cost(group) for group in partition),
                    len(partition),
                )

            brute = min(
                score(partition)
                for partition in all_partitions(list(range(len(dilations))))
            )
            assert score(selected) == brute


def test_fifty_step_adamw_trajectory_and_metrics_are_preserved():
    torch.manual_seed(123)
    original = OriginalDT1D(**kwargs(c=8, no_pw=False))
    optimized = OptimizedDT1D(
        **kwargs(c=8, no_pw=False), exact_cost_realization=True
    )
    seed_nontrivial(original, seed=321)
    copy_original_state(original, optimized)

    head_original = torch.nn.Linear(8, 5)
    head_optimized = torch.nn.Linear(8, 5)
    head_optimized.load_state_dict(head_original.state_dict())

    opt_original = torch.optim.AdamW(
        list(original.parameters()) + list(head_original.parameters()), lr=1e-3
    )
    opt_optimized = torch.optim.AdamW(
        list(optimized.parameters()) + list(head_optimized.parameters()), lr=1e-3
    )

    max_loss_diff = 0.0
    min_agreement = 1.0
    for step in range(50):
        generator = torch.Generator().manual_seed(2000 + step)
        x = torch.randn((6, 8, 20, 21), generator=generator)
        target = torch.randint(0, 5, (6,), generator=generator)
        records = []
        for model, head, optimizer in (
            (original, head_original, opt_original),
            (optimized, head_optimized, opt_optimized),
        ):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = head(model(x).mean(dim=(2, 3)))
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            records.append((loss.detach(), logits.detach()))
        max_loss_diff = max(
            max_loss_diff, float((records[0][0] - records[1][0]).abs())
        )
        agreement = (records[0][1].argmax(1) == records[1][1].argmax(1)).float().mean()
        min_agreement = min(min_agreement, float(agreement))

    assert max_loss_diff <= 2e-7
    assert min_agreement == 1.0
    for p_o, p_n in zip(
        list(original.parameters()) + list(head_original.parameters()),
        list(optimized.parameters()) + list(head_optimized.parameters()),
    ):
        torch.testing.assert_close(p_n, p_o, rtol=1e-5, atol=3e-6)


def test_only_runtime_graph_change_is_axial_call_partition():
    common = kwargs(c=8, no_pw=False)
    before = OptimizedDT1D(**common, exact_cost_realization=False)
    after = OptimizedDT1D(**common, exact_cost_realization=True)
    after.load_state_dict(before.state_dict(), strict=True)
    x = torch.randn(2, 8, 56, 56)

    counts = {}
    for name, model in (('before', before), ('after', after)):
        count = {'conv_axis': 0, 'kernel_builds': 0}
        original_conv = model._conv_axis
        original_build = model._build_weighted_hcc_kernel_1d

        def counted_conv(*args, _fn=original_conv, _count=count, **kw):
            _count['conv_axis'] += 1
            return _fn(*args, **kw)

        def counted_build(*args, _fn=original_build, _count=count, **kw):
            _count['kernel_builds'] += 1
            return _fn(*args, **kw)

        model._conv_axis = counted_conv
        model._build_weighted_hcc_kernel_1d = counted_build
        model(x)
        counts[name] = count

    assert counts['before'] == {'conv_axis': 6, 'kernel_builds': 6}
    assert counts['after'] == {'conv_axis': 4, 'kernel_builds': 6}
