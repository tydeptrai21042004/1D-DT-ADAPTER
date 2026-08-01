from fractions import Fraction

from tools.verify_pareto_laurent_optimality import (
    branch_poly,
    closed_form_12,
    partition_cost,
    reflect_fusion_threshold,
    unique_partitions,
)


def test_default_partition_pareto_costs():
    rows = {p: partition_cost(p, radius=2) for p in unique_partitions((1, 2, 4))}
    assert rows[((1, 2, 4),)] == (1, 17)
    assert rows[((1, 2), (4,))] == (2, 14)
    assert rows[((1,), (2, 4))] == (2, 14)
    assert rows[((1,), (2,), (4,))] == (3, 15)
    assert min(taps for calls, taps in rows.values() if calls == 2) == 14


def test_boundary_tie_break_prefers_12_group():
    p12 = ((1, 2), (4,))
    p24 = ((1,), (2, 4))
    assert reflect_fusion_threshold(p12, radius=2) == 5
    assert reflect_fusion_threshold(p24, radius=2) == 9


def test_closed_form_laurent_identity_exact_rational():
    a1, b1, pi1 = 3, -2, Fraction(4, 7)
    a2, b2, pi2 = -5, 1, Fraction(3, 8)
    direct = branch_poly(a1, b1, pi1.numerator, pi1.denominator, 1)
    second = branch_poly(a2, b2, pi2.numerator, pi2.denominator, 2)
    for exponent, coeff in second.items():
        direct[exponent] = direct.get(exponent, Fraction(0)) + coeff
    direct = {e: c for e, c in direct.items() if c != 0}
    fused = {e: c for e, c in closed_form_12(a1, b1, pi1, a2, b2, pi2).items() if c != 0}
    assert direct == fused
