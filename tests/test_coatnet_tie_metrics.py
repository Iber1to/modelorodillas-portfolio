"""Synthetic label-free examples and metamorphic checks; no private artifacts."""
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from tools import coatnet_tie_metrics as metrics


class TieMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rankers = metrics.load_rankers()

    def analyze(self, values, names=None):
        matrix = np.asarray(values)
        return metrics.analyze_matrix(matrix, names or [f"target{i}" for i in range(matrix.shape[1])],
                                      self.rankers)

    def test_actual_source_rules_match_known_hand_ranks(self):
        values = np.array([[0.8], [0.1], [0.1], [0.4]])
        average = self.rankers.average(values)
        np.testing.assert_array_equal(average[:, 0], [1, 1 / 6, 1 / 6, 2 / 3])
        ordinal = self.rankers.ordinal(values)
        self.assertEqual(ordinal[0, 0], 1)
        self.assertEqual(ordinal[3, 0], 2 / 3)
        np.testing.assert_array_equal(np.sort(ordinal[1:3, 0]), [0, 1 / 3])

    def test_known_tie_counts_and_score_changes(self):
        result = self.analyze([[0], [0], [0.5], [0.5], [0.5], [1]])
        column = result["columns"][0]
        self.assertEqual({key: column[key] for key in (
            "n_unique", "tie_groups", "tied_rows", "excess_duplicates", "tied_pairs",
            "max_tie_size", "saturated_zero_rows", "saturated_one_rows")},
            {"n_unique": 3, "tie_groups": 2, "tied_rows": 5, "excess_duplicates": 3,
             "tied_pairs": 4, "max_tie_size": 3, "saturated_zero_rows": 2,
             "saturated_one_rows": 1})
        self.assertAlmostEqual(column["ordinal_vs_average"]["max_abs_delta"], 0.2)
        self.assertEqual(column["ordinal_vs_average"]["affected_rows"], 4)
        self.assertTrue(result["average_exactly_order_invariant"])
        self.assertTrue(result["ordinal_order_sensitive"])

    def test_no_ties_has_no_changes_or_order_sensitivity(self):
        result = self.analyze([[0.9], [0.1], [0.6], [0.4]])
        column = result["columns"][0]
        self.assertEqual(column["n_unique"], 4)
        self.assertEqual(column["tie_groups"], 0)
        self.assertEqual(column["max_tie_size"], 0)
        self.assertEqual(column["tied_pairs"], 0)
        self.assertTrue(result["ordinal_vs_average"]["exactly_equal"])
        self.assertFalse(result["ordinal_order_sensitive"])

    def test_single_row_ranks_are_zero_even_when_raw_is_one(self):
        result = self.analyze([[1, 0.3]], ["a", "b"])
        self.assertEqual(result["columns"][0]["saturated_one_rows"], 1)
        self.assertEqual(result["columns"][0]["max_tie_size"], 0)
        self.assertTrue(result["ordinal_vs_average"]["exactly_equal"])
        np.testing.assert_array_equal(self.rankers.average(np.array([[1, 0.3]])), [[0, 0]])
        self.assertFalse(result["ordinal_order_sensitive"])

    def test_all_tied_average_is_one_half_and_reverse_changes_ordinal(self):
        values = np.full((4, 1), 0.25)
        result = self.analyze(values)
        np.testing.assert_array_equal(self.rankers.average(values), np.full((4, 1), 0.5))
        self.assertEqual(result["columns"][0]["tied_pairs"], 6)
        self.assertEqual(result["ordinal_vs_average"]["max_abs_delta"], 0.5)
        self.assertEqual(result["ordinal_vs_average"]["affected_rows"], 4)
        reverse = result["order_checks"][1]
        self.assertEqual(reverse["order"], "reverse")
        self.assertEqual(reverse["ordinal"]["affected_rows"], 4)
        self.assertEqual(reverse["ordinal"]["max_abs_delta"], 1)

    def test_independent_columns_and_exact_equality(self):
        next_value = np.nextafter(np.float32(0.5), np.float32(1))
        values = np.array([[0.5, 0], [0.5, next_value], [0.5, 0.5]], dtype=np.float32)
        result = self.analyze(values, ["tied", "distinct"])
        self.assertEqual(result["columns"][0]["tied_rows"], 3)
        self.assertEqual(result["columns"][1]["tie_groups"], 0)
        self.assertTrue(result["columns"][1]["ordinal_vs_average"]["exactly_equal"])

    def test_original_reverse_and_frozen_seed_orders_are_reproducible(self):
        values = np.full((10, 2), 0.5)
        result = self.analyze(values)
        self.assertEqual([item["order"] for item in result["order_checks"]],
                         ["original", "reverse", "seed_20260909", "seed_20260910", "seed_20260911"])
        self.assertEqual(result, self.analyze(values))
        self.assertTrue(result["order_checks"][0]["ordinal"]["exactly_equal"])
        self.assertTrue(all(item["average"]["exactly_equal"] for item in result["order_checks"]))
        json.dumps(result, allow_nan=False)

    def test_strict_monotone_transform_preserves_diagnostics(self):
        values = np.array([[0.2, 0.4], [0.2, 0.9], [0.7, 0.4], [0.9, 0.6]])
        original = self.analyze(values)
        transformed = self.analyze(0.1 + 0.8 * values)
        self.assertEqual(original["columns"], transformed["columns"])
        self.assertEqual(original["order_checks"], transformed["order_checks"])

    def test_column_reordering_only_reorders_per_column_results(self):
        values = np.array([[0, 0.2], [0, 0.9], [1, 0.5]])
        original = self.analyze(values, ["a", "b"])
        reversed_columns = self.analyze(values[:, ::-1], ["b", "a"])
        self.assertEqual(original["columns"], reversed_columns["columns"][::-1])

    def test_input_is_not_mutated(self):
        values = np.array([[0.4, 0.5], [0.4, 0.6]])
        frozen = values.copy()
        values.setflags(write=False)
        self.analyze(values)
        np.testing.assert_array_equal(values, frozen)

    def test_invalid_matrices_are_rejected(self):
        invalid = [[], [0.1, 0.2], np.empty((0, 2)), np.empty((2, 0)),
                   [[[0.5]]], [[np.nan]], [[np.inf]], [[-np.inf]],
                   [[-0.001]], [[1.001]], [[1j]], [["0.5"]], [[True]],
                   np.array([[0.5]], dtype=object)]
        for values in invalid:
            with self.subTest(values=str(values)):
                with self.assertRaises(ValueError):
                    metrics.analyze_matrix(values, ["target"], self.rankers)

    def test_invalid_names_are_rejected(self):
        for names in ("ab", None, ["a"], ["a", "a"], ["a", ""], ["a", 7]):
            with self.subTest(names=names):
                with self.assertRaises(ValueError):
                    metrics.analyze_matrix([[0.1, 0.2]], names, self.rankers)

    def test_broken_average_invariance_is_a_failure(self):
        broken = metrics.Rankers(self.rankers.ordinal, self.rankers.ordinal, {})
        with self.assertRaisesRegex(ValueError, "not exactly invariant"):
            metrics.analyze_matrix(np.full((4, 1), 0.5), ["a"], broken)

    def test_invalid_extracted_output_is_a_failure(self):
        for function in (lambda values: np.full_like(values, np.nan),
                         lambda values: values[:, :1]):
            rankers = metrics.Rankers(function, self.rankers.average, {})
            with self.assertRaises(ValueError):
                metrics.analyze_matrix(np.ones((3, 2)), ["a", "b"], rankers)

    def test_ast_extraction_does_not_execute_module_imports_or_top_level_code(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.py"
            path.write_text("import absent_heavy_dependency\nraise RuntimeError('top-level')\n"
                            "def rankpct(x):\n    return np.zeros_like(x)\n", encoding="utf-8")
            function, provenance = metrics._extract_rankpct(path)
            np.testing.assert_array_equal(function(np.ones((2, 1))), [[0], [0]])
            self.assertEqual(len(provenance["source_sha256"]), 64)

    def test_ast_extraction_rejects_missing_duplicate_or_importing_function(self):
        sources = ["rankpct = None\n", "def rankpct(x):\n return x\ndef rankpct(x):\n return x\n",
                   "def rankpct(x):\n import torch\n return x\n"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.py"
            for source in sources:
                with self.subTest(source=source):
                    path.write_text(source, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        metrics._extract_rankpct(path)


class FusionDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rankers = metrics.load_rankers()
        cls.names = [f"target{i}" for i in range(12)]

    def raw(self, own_values, public_values):
        arms = np.array([own_values, own_values, own_values, public_values], dtype=np.float64)
        return np.repeat(arms[:, :, None], 12, axis=2)

    def compare(self, raw):
        return metrics.compare_fusions(raw, self.names, self.rankers)

    def test_pair_order_partition_known_example(self):
        result = metrics.compare_pair_orders([0, 0, 0.5, 1], [0, 0.5, 0.5, 0])
        self.assertEqual(result, {"n_pairs": 6, "inversions": 2, "tied_to_untied": 1,
                                  "untied_to_tied": 2, "both_tied": 0,
                                  "same_strict_order": 1, "changed_pair_relations": 5})

    def test_pair_order_direction_and_row_relabeling(self):
        a = np.array([0, 0, 0.5, 1])
        b = np.array([0, 0.5, 0.5, 0])
        forward = metrics.compare_pair_orders(a, b)
        reverse = metrics.compare_pair_orders(b, a)
        self.assertEqual(forward["inversions"], reverse["inversions"])
        self.assertEqual(forward["tied_to_untied"], reverse["untied_to_tied"])
        order = [2, 0, 3, 1]
        self.assertEqual(forward, metrics.compare_pair_orders(a[order], b[order]))

    def test_no_public_ties_means_same_fusion_even_with_own_ties(self):
        result = self.compare(self.raw([0.5, 0.5, 0.5], [0.1, 0.9, 0.4]))
        self.assertTrue(result["average_to_public_ordinal"]["exactly_equal"])
        self.assertFalse(result["public_ordinal_order_sensitive"])
        self.assertTrue(all(column["pair_orders_average_to_public_ordinal"]["changed_pair_relations"] == 0
                            for column in result["columns"]))

    def test_only_public_rule_changes_known_tied_fusion(self):
        raw = self.raw([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
        result = self.compare(raw)
        self.assertEqual(result["average_to_public_ordinal"]["max_abs_delta"], 0.25)
        self.assertEqual(result["average_to_public_ordinal"]["affected_rows"], 4)
        self.assertEqual(result["columns"][0]["pair_orders_average_to_public_ordinal"]["tied_to_untied"], 6)
        self.assertEqual(result["columns"][0]["pair_orders_average_to_public_ordinal"]["inversions"], 0)
        self.assertTrue(all(check["own_component"]["exactly_equal"] for check in result["order_checks"]))
        self.assertTrue(result["public_ordinal_order_sensitive"])
        self.assertTrue(result["serialization_applied"])
        self.assertEqual(result["score_dtype"], "float32")
        self.assertEqual(result["float64_before_cast"]["average_to_public_ordinal"]["max_abs_delta"], 0.25)
        json.dumps(result, allow_nan=False)

    def test_three_distinct_own_arms_are_averaged_without_reranking(self):
        raw = self.raw([0.1, 0.5, 0.9], [0.1, 0.1, 0.1])
        raw[1] = raw[1, ::-1]
        raw[2] = 0.5
        own, average, ordinal = metrics._fusion_scores(raw, self.rankers)
        np.testing.assert_array_equal(own, np.full((3, 12), 0.5))
        np.testing.assert_array_equal(average, np.full((3, 12), 0.5))
        np.testing.assert_array_equal(np.sort(ordinal, axis=0),
                                      np.repeat(np.array([[0.25], [0.5], [0.75]]), 12, axis=1))

    def test_pair_counts_follow_float32_output_when_rounding_merges_scores(self):
        # The last two mathematical scores are nominally 1/4, reached through
        # different floating-point sums. Their tiny float64 difference disappears
        # in source36's output cast. The diagnostic must use the emitted tie.
        arms = np.array([[1, 1 / 3, 1 / 3, 2 / 3],
                         [2 / 3, 1, 1 / 3, 2 / 3],
                         [2 / 3, 0, 0, 0],
                         [1, 2 / 3, 1 / 3, 0]])
        result = self.compare(np.repeat(arms[:, :, None], 12, axis=2))
        emitted = result["columns"][0]["pair_orders_average_to_public_ordinal"]
        before_cast = result["float64_before_cast"]["columns"][0]["pair_orders_average_to_public_ordinal"]
        self.assertEqual(emitted["both_tied"], 1)
        self.assertEqual(before_cast["both_tied"], 0)
        self.assertEqual(emitted["changed_pair_relations"], 0)

    def test_one_row_fusion_has_no_pairs_and_no_changes(self):
        result = self.compare(self.raw([0.8], [1]))
        self.assertTrue(result["average_to_public_ordinal"]["exactly_equal"])
        self.assertEqual(result["columns"][0]["pair_orders_average_to_public_ordinal"]["n_pairs"], 0)
        self.assertFalse(result["public_ordinal_order_sensitive"])

    def test_fusion_input_is_not_mutated(self):
        raw = self.raw([0.2, 0.5, 0.9], [0.2, 0.2, 0.9])
        original = raw.copy()
        raw.setflags(write=False)
        self.compare(raw)
        np.testing.assert_array_equal(raw, original)

    def test_fusion_shapes_values_and_names_rejected(self):
        invalid = [np.zeros((3, 3, 12)), np.zeros((4, 3, 11)), np.zeros((4, 0, 12)),
                   np.zeros((4, 12)), np.full((4, 3, 12), np.nan),
                   np.full((4, 3, 12), -1), np.full((4, 3, 12), 2),
                   np.full((4, 3, 12), 1j)]
        for raw in invalid:
            with self.subTest(shape=raw.shape, dtype=raw.dtype):
                with self.assertRaises(ValueError):
                    self.compare(raw)
        with self.assertRaises(ValueError):
            metrics.compare_fusions(self.raw([0.1], [0.2]), ["wrong"], self.rankers)

    def test_pair_comparison_rejects_invalid_input(self):
        for first, second in (([], []), ([0], [0, 1]), ([[0]], [[0]]),
                              ([np.nan], [0]), ([0], [2]), ([1j], [0])):
            with self.subTest(first=first, second=second):
                with self.assertRaises(ValueError):
                    metrics.compare_pair_orders(first, second)


if __name__ == "__main__":
    unittest.main()
