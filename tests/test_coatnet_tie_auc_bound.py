"""Check the bound with synthetic rankings and exhaustive synthetic labels."""
import itertools
import unittest

import numpy as np

from tools.coatnet_tie_auc_bound import max_auc_tie_bound


def manual_auc(labels, scores):
    """Direct positive/negative pair comparison, without sklearn or real data."""
    positives = [i for i, value in enumerate(labels) if value == 1]
    negatives = [i for i, value in enumerate(labels) if value == 0]
    total = 0.0
    for positive in positives:
        for negative in negatives:
            total += float(scores[positive] > scores[negative])
            total += 0.5 * float(scores[positive] == scores[negative])
    return total / (len(positives) * len(negatives))


def exhaustive_auc_change(values):
    """Enumerate every nondegenerate synthetic binary labeling."""
    values = np.asarray(values)
    ordinal = values.argsort().argsort()
    best, smallest_p = -1.0, None
    for labels in itertools.product((0, 1), repeat=len(values)):
        p = sum(labels)
        if p in (0, len(values)):
            continue
        # Raw values and average ranks induce exactly the same pairwise order.
        delta = abs(manual_auc(labels, ordinal) - manual_auc(labels, values))
        if delta > best + 1e-14:
            best, smallest_p = delta, p
        elif abs(delta - best) <= 1e-14:
            smallest_p = min(smallest_p, p)
    return best, smallest_p


class TieAucBoundTests(unittest.TestCase):
    def test_every_labeling_of_small_grouped_vectors(self):
        # Exhaust every vector over three values up to length six, modulo row
        # permutation. This covers several simultaneous ties and singleton groups.
        for n in range(2, 7):
            for values in itertools.combinations_with_replacement(range(3), n):
                with self.subTest(values=values):
                    expected, expected_p = exhaustive_auc_change(values)
                    actual = max_auc_tie_bound(values)
                    self.assertAlmostEqual(actual["bound"], expected, places=14)
                    self.assertEqual(actual["worst_case_positive_count"], expected_p)

    def test_all_ties_have_half_auc_bound(self):
        for n in (2, 3, 6, 101):
            with self.subTest(n=n):
                result = max_auc_tie_bound(np.full(n, 0.75))
                self.assertEqual(result["bound"], 0.5)
                self.assertEqual(result["worst_case_positive_count"], 1)
                self.assertEqual(result["tie_groups"], [n])
                self.assertEqual(result["tie_pair_count"], n * (n - 1) // 2)

    def test_distinct_values_have_zero_bound(self):
        result = max_auc_tie_bound([0.2, -4, 10.2, 1])
        self.assertEqual(result["bound"], 0)
        self.assertEqual(result["worst_case_positive_count"], 1)
        self.assertEqual(result["tie_groups"], [1, 1, 1, 1])
        self.assertEqual(result["tie_pair_count"], 0)

    def test_multiple_groups_can_maximize_at_more_than_one_positive(self):
        result = max_auc_tie_bound([0] * 5 + [1] * 5)
        self.assertEqual(result["bound"], 0.25)
        self.assertEqual(result["worst_case_positive_count"], 2)

    def test_bound_is_invariant_to_row_order(self):
        values = np.array([0, 0, 1, 1, 1, 2, 3])
        expected = max_auc_tie_bound(values)
        for indices in (np.arange(7)[::-1], np.array([3, 6, 0, 2, 5, 4, 1])):
            with self.subTest(indices=indices.tolist()):
                actual = max_auc_tie_bound(values[indices])
                self.assertEqual(actual, expected)
                brute, brute_p = exhaustive_auc_change(values[indices])
                self.assertAlmostEqual(actual["bound"], brute, places=14)
                self.assertEqual(actual["worst_case_positive_count"], brute_p)

    def test_equality_preserves_original_integer_precision(self):
        result = max_auc_tie_bound(np.array([2**60, 2**60 + 1], dtype=np.int64))
        self.assertEqual(result["bound"], 0)

    def test_equality_is_exact_without_rounding(self):
        values = np.array([0.5, np.nextafter(0.5, 1.0), 0.6])
        self.assertEqual(max_auc_tie_bound(values)["bound"], 0)

    def test_single_observation_has_no_auc_bound(self):
        result = max_auc_tie_bound([1.0])
        self.assertIsNone(result["bound"])
        self.assertIsNone(result["worst_case_positive_count"])
        self.assertEqual(result["tie_groups"], [1])
        self.assertEqual(result["reason"], "auc_undefined_for_fewer_than_two_observations")

    def test_invalid_inputs_are_rejected(self):
        inputs = (
            [], np.array([]), 1.0, [[1, 2]], [np.nan], [np.inf], [-np.inf],
            [1 + 0j], ["1", "2"], np.array([1], dtype=object), [True, False],
        )
        for values in inputs:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    max_auc_tie_bound(values)

    def test_scope_is_explicit(self):
        result = max_auc_tie_bound([1.0, 1.0, 2.0])
        self.assertEqual(result["scope"], "isolated_ranking_observed_corpus_only")
        self.assertIsNone(result["reason"])


if __name__ == "__main__":
    unittest.main()
