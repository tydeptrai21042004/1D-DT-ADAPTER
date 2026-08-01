from __future__ import annotations

import argparse
import json
import math
import random
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ScaleGroup = Tuple[int, ...]
Partition = Tuple[ScaleGroup, ...]
Polynomial = Dict[int, Fraction]


def set_partitions(items: Sequence[int]) -> Iterable[Partition]:
    """Enumerate unordered set partitions in canonical sorted form."""
    items = tuple(sorted(items))
    if not items:
        yield tuple()
        return
    first = items[0]
    for rest in set_partitions(items[1:]):
        # first begins a new block
        yield tuple(sorted(((first,),) + rest))
        # first joins an existing block
        for i in range(len(rest)):
            block = tuple(sorted((first,) + rest[i]))
            candidate = list(rest)
            candidate[i] = block
            canonical = tuple(sorted(candidate))
            yield canonical


def unique_partitions(items: Sequence[int]) -> List[Partition]:
    return sorted(set(set_partitions(items)), key=lambda p: (len(p), p))


def group_dense_taps(group: ScaleGroup, radius: int) -> int:
    """Minimum dense taps for one exact dilated convolution realization.

    The fused branch uses dilation gcd(group), so its dense coefficient radius is
    radius * max(group) / gcd(group). A singleton retains its native dilation and
    therefore has 2*radius+1 taps.
    """
    if not group:
        raise ValueError("group must be non-empty")
    g = math.gcd(*group)
    return 2 * radius * (max(group) // g) + 1


def partition_cost(partition: Partition, radius: int) -> Tuple[int, int]:
    return len(partition), sum(group_dense_taps(g, radius) for g in partition)


def reflect_fusion_threshold(partition: Partition, radius: int) -> int:
    """Smallest feature length for which every multi-scale fused group is reflect-valid.

    PyTorch reflection padding requires pad < feature length. The largest physical
    radius inside a fused group is radius*max(d). Singletons need no cross-scale
    boundary compatibility and are excluded from this tie-break.
    """
    fused = [g for g in partition if len(g) > 1]
    if not fused:
        return 1
    return 1 + radius * max(max(g) for g in fused)


def add_poly(dst: Polynomial, src: Polynomial) -> None:
    for e, c in src.items():
        dst[e] = dst.get(e, Fraction(0)) + c
        if dst[e] == 0:
            del dst[e]


def branch_poly(a: int, b: int, pi_num: int, pi_den: int, dilation: int) -> Polynomial:
    den = 2 * abs(a) + 4 * abs(b)
    if den == 0:
        raise ValueError("a and b cannot both be zero")
    pi = Fraction(pi_num, pi_den)
    p = pi * Fraction(a, den)
    q = pi * Fraction(b, den)
    return {
        -2 * dilation: q,
        -1 * dilation: p,
        0: 2 * q,
        1 * dilation: p,
        2 * dilation: q,
    }


def closed_form_12(a1: int, b1: int, pi1: Fraction,
                   a2: int, b2: int, pi2: Fraction) -> Polynomial:
    d1 = 2 * abs(a1) + 4 * abs(b1)
    d2 = 2 * abs(a2) + 4 * abs(b2)
    p1, q1 = pi1 * Fraction(a1, d1), pi1 * Fraction(b1, d1)
    p2, q2 = pi2 * Fraction(a2, d2), pi2 * Fraction(b2, d2)
    # Length-nine dense kernel indexed by physical Laurent exponent -4,...,4.
    return {
        -4: q2,
        -2: q1 + p2,
        -1: p1,
        0: 2 * (q1 + q2),
        1: p1,
        2: q1 + p2,
        4: q2,
    }


def verify_exact_closed_form(trials: int, seed: int) -> Tuple[bool, int]:
    rng = random.Random(seed)
    for _ in range(trials):
        vals = []
        for _scale in (1, 2):
            while True:
                a = rng.randint(-9, 9)
                b = rng.randint(-9, 9)
                if a != 0 or b != 0:
                    break
            num = rng.randint(1, 9)
            den = rng.randint(1, 9)
            vals.append((a, b, Fraction(num, den)))
        (a1, b1, pi1), (a2, b2, pi2) = vals
        direct: Polynomial = {}
        add_poly(direct, branch_poly(a1, b1, pi1.numerator, pi1.denominator, 1))
        add_poly(direct, branch_poly(a2, b2, pi2.numerator, pi2.denominator, 2))
        fused = closed_form_12(a1, b1, pi1, a2, b2, pi2)
        direct = {e: c for e, c in direct.items() if c != 0}
        fused = {e: c for e, c in fused.items() if c != 0}
        if direct != fused:
            return False, _
    return True, trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("pareto_laurent_verification.json"))
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    scales = (1, 2, 4)
    radius = 2  # M=1 => effective radius M+1=2
    rows = []
    for p in unique_partitions(scales):
        calls, taps = partition_cost(p, radius)
        rows.append({
            "partition": [list(g) for g in p],
            "calls_per_axis": calls,
            "dense_taps_per_axis": taps,
            "reflect_all_fused_valid_from_length": reflect_fusion_threshold(p, radius),
        })

    min_two_call_taps = min(r["dense_taps_per_axis"] for r in rows if r["calls_per_axis"] == 2)
    two_call_optima = [r for r in rows if r["calls_per_axis"] == 2 and r["dense_taps_per_axis"] == min_two_call_taps]
    selected = min(two_call_optima, key=lambda r: r["reflect_all_fused_valid_from_length"])
    one_call = next(r for r in rows if r["calls_per_axis"] == 1)
    three_call = next(r for r in rows if r["calls_per_axis"] == 3)

    exact_ok, checked = verify_exact_closed_form(args.trials, args.seed)
    report = {
        "configuration": {"M": 1, "effective_radius": radius, "dilations": list(scales)},
        "all_partitions": rows,
        "pareto_points_calls_taps": sorted(set((r["calls_per_axis"], r["dense_taps_per_axis"]) for r in rows)),
        "minimum_two_call_taps": min_two_call_taps,
        "two_call_tap_optima": two_call_optima,
        "selected_partition": selected,
        "selection_rule": "minimum dense taps among two-call exact realizations, then maximum reflect-boundary compatibility domain",
        "one_call_lower_bound": one_call,
        "unfused_reference": three_call,
        "closed_form_rational_trials": checked,
        "closed_form_exact_in_all_trials": exact_ok,
        "claims_verified": {
            "selected_is_two_call_tap_optimal": selected["dense_taps_per_axis"] == min_two_call_taps,
            "selected_dominates_unfused_in_calls_and_taps": (
                selected["calls_per_axis"] < three_call["calls_per_axis"]
                and selected["dense_taps_per_axis"] < three_call["dense_taps_per_axis"]
            ),
            "one_call_requires_more_taps_than_selected": one_call["dense_taps_per_axis"] > selected["dense_taps_per_axis"],
            "closed_form_is_exact_over_rationals": exact_ok,
        },
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(report["claims_verified"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
