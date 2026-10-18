"""Label-free worst-case AUC change when exact ties receive ordinal ranks.

This bound applies to ONE isolated ranking on the supplied observed corpus. It
does not bound an ensemble/blend, the hidden test, or a different population.
No actual labels are accepted or read. Equality is exact in the supplied dtype.
"""
from __future__ import annotations

import numpy as np


def max_auc_tie_bound(values_1d):
    """Return the tight label-free bound for ordinal versus average-tie AUC.

    For group sizes m_g and unknown positive counts p_g, resolving ties can
    change AUC by at most sum(p_g * (m_g-p_g)) / (2 * P * (n-P)). All other
    positive/negative pairs keep their ordering. This maximum is attainable:
    place each group's positives at one end of that group's ordinal ordering.

    At fixed P, the numerator is maximized by taking the P largest marginal
    gains from each group's sequence m_g-1, m_g-3, ..., 1-m_g. These sequences
    decrease, so the selected increments always form valid group prefixes.
    Maximizing over P=1,...,n-1 gives a tight bound over all binary labelings
    with both classes present. The smallest maximizing P is returned.

    ``tie_groups`` includes singleton groups and contains only group sizes,
    in ascending order of their associated input values. For n=1, AUC cannot
    be defined with both classes, so ``bound`` and the maximizing P are None.
    Runtime is O(n log n), with O(n) additional memory.
    """
    values = np.asarray(values_1d)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("expected a nonempty one-dimensional array")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("expected real numeric values")
    if not np.isfinite(values).all():
        raise ValueError("expected finite values")

    # Do not convert to float: large distinct integers must remain distinct.
    _, counts = np.unique(values, return_counts=True)
    groups = [int(count) for count in counts]
    n = int(values.size)
    result = {
        "n": n,
        "bound": None,
        "worst_case_positive_count": None,
        "tie_groups": groups,
        "tie_pair_count": sum(m * (m - 1) // 2 for m in groups),
        "reason": None,
        "scope": "isolated_ranking_observed_corpus_only",
    }
    if n == 1:
        result["reason"] = "auc_undefined_for_fewer_than_two_observations"
        return result

    marginal_gains = sorted(
        (gain for m in groups for gain in range(m - 1, -m, -2)),
        reverse=True,
    )
    numerator = 0
    best_numerator, best_denominator, best_p = -1, 1, None
    for p, gain in enumerate(marginal_gains[:-1], start=1):
        numerator += gain
        denominator = 2 * p * (n - p)
        # Compare exact rational values rather than rounded floating scores.
        if numerator * best_denominator > best_numerator * denominator:
            best_numerator, best_denominator, best_p = numerator, denominator, p
    result["bound"] = best_numerator / best_denominator
    result["worst_case_positive_count"] = best_p
    return result
