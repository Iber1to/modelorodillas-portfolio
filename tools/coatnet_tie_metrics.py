"""Label-free, post-hoc diagnostics for the frozen CoAtNet ranking rules.

This module loads only ``rankpct`` function ASTs from the two local inference
sources. It never imports or runs either inference script. Inputs are prediction
matrices and target *names*, never outcomes; no AUC, tuning, or submission lives
here. Exact equality defines both prediction ties and score changes.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE22 = ROOT / "kaggle/22_public_coatnet_anchor/infer_public_coatnet.py"
SOURCE36 = ROOT / "kaggle/36_coatnet_fusion_guarded/infer_fusion_guarded.py"
PERMUTATION_SEEDS = (20260909, 20260910, 20260911)


@dataclass(frozen=True)
class Rankers:
    ordinal: Callable[[np.ndarray], np.ndarray]
    average: Callable[[np.ndarray], np.ndarray]
    provenance: dict


def _extract_rankpct(path: Path) -> tuple[Callable, dict]:
    """Compile only the unique top-level function, with NumPy as its sole import."""
    source_bytes = path.read_bytes()
    source = source_bytes.decode("utf-8-sig").replace("\r\n", "\n")
    tree = ast.parse(source, filename=str(path))
    matches = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "rankpct"]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected exactly one top-level rankpct")
    function = matches[0]
    if function.decorator_list or function.args.defaults or function.args.kw_defaults:
        raise ValueError(f"{path}: unexpected decorated/defaulted rankpct")
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(function)):
        raise ValueError(f"{path}: rankpct must not import dependencies")
    namespace = {"np": np}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    function_ast = ast.dump(function, include_attributes=False)
    return namespace["rankpct"], {
        "path": str(path),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "normalized_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "function_ast_sha256": hashlib.sha256(function_ast.encode("utf-8")).hexdigest(),
    }


def load_rankers() -> Rankers:
    """Extract the actual rules from frozen local sources22 and36 on demand."""
    ordinal, provenance22 = _extract_rankpct(SOURCE22)
    average, provenance36 = _extract_rankpct(SOURCE36)
    return Rankers(ordinal, average, {"ordinal_source22": provenance22,
                                     "average_source36": provenance36})


def _validate_values(values, name: str, ndim: int) -> np.ndarray:
    try:
        matrix = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: cannot form a numeric array") from error
    if matrix.ndim != ndim or not all(matrix.shape):
        raise ValueError(f"{name}: expected a nonempty {ndim}D array")
    if not np.issubdtype(matrix.dtype, np.number) or np.iscomplexobj(matrix):
        raise ValueError(f"{name}: expected real numeric values")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name}: non-finite values")
    if (matrix < 0).any() or (matrix > 1).any():
        raise ValueError(f"{name}: values outside [0,1]")
    return matrix


def _validate_names(target_names: Sequence[str], n_columns: int) -> list[str]:
    if isinstance(target_names, (str, bytes)):
        raise ValueError("target_names: supply a sequence of target names")
    try:
        names = list(target_names)
    except TypeError as error:
        raise ValueError("target_names: supply a sequence of target names") from error
    if len(names) != n_columns:
        raise ValueError("target_names: count does not match matrix columns")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("target_names: names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("target_names: duplicate names")
    return names


def _rank(matrix: np.ndarray, function: Callable, name: str) -> np.ndarray:
    result = _validate_values(function(matrix), f"{name} rank output", 2)
    if result.shape != matrix.shape:
        raise ValueError(f"{name} rank output: shape mismatch")
    return result


def _change(first: np.ndarray, second: np.ndarray) -> dict:
    delta = second.astype(np.float64) - first.astype(np.float64)
    return {
        "exactly_equal": bool(np.array_equal(first, second)),
        "affected_values": int(np.count_nonzero(first != second)),
        "affected_rows": int(np.count_nonzero(np.any(first != second, axis=1)))
            if first.ndim == 2 else int(np.count_nonzero(first != second)),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "min_signed_delta": float(np.min(delta)),
        "max_signed_delta": float(np.max(delta)),
    }


def _orders(n_rows: int):
    original = np.arange(n_rows, dtype=np.int64)
    yield "original", None, original
    yield "reverse", None, original[::-1]
    for seed in PERMUTATION_SEEDS:
        yield f"seed_{seed}", seed, np.random.default_rng(seed).permutation(n_rows)


def _order_record(name: str, seed: int | None, order: np.ndarray) -> dict:
    return {"order": name, "seed": seed,
            "row_permutation_sha256": hashlib.sha256(
                order.astype("<i8").tobytes()).hexdigest()}


def _restore_order(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    restored = np.empty_like(values)
    restored[order] = values
    return restored


def _tie_counts(column: np.ndarray) -> dict:
    _, counts = np.unique(column, return_counts=True)
    tied = counts[counts > 1]
    # Python integers avoid overflow in combinatorial counts.
    sizes = [int(size) for size in tied]
    return {
        "n_unique": int(len(counts)),
        "tie_groups": len(sizes),
        "tied_rows": sum(sizes),
        "excess_duplicates": sum(size - 1 for size in sizes),
        "tied_pairs": sum(size * (size - 1) // 2 for size in sizes),
        "max_tie_size": max(sizes, default=0),
        "saturated_zero_rows": int(np.count_nonzero(column == 0)),
        "saturated_one_rows": int(np.count_nonzero(column == 1)),
    }


def analyze_matrix(values, target_names: Sequence[str], rankers: Rankers | None = None) -> dict:
    """Describe raw ties and rule/order sensitivity, with no observed outcomes.

    ``max_tie_size`` is zero when there is no repeated value. Affected rows and
    permutations use exact equality. Each permutation is undone before comparison
    with the original row order. Average-rank non-invariance fails the diagnostic.
    """
    matrix = _validate_values(values, "matrix", 2)
    names = _validate_names(target_names, matrix.shape[1])
    rankers = rankers or load_rankers()
    ordinal = _rank(matrix, rankers.ordinal, "ordinal")
    average = _rank(matrix, rankers.average, "average")
    columns = []
    for index, name in enumerate(names):
        columns.append({"target": name, **_tie_counts(matrix[:, index]),
                        "ordinal_vs_average": _change(average[:, index], ordinal[:, index])})
    order_checks = []
    for name, seed, order in _orders(len(matrix)):
        reordered = matrix[order]
        restored_ordinal = _restore_order(_rank(reordered, rankers.ordinal, "ordinal"), order)
        restored_average = _restore_order(_rank(reordered, rankers.average, "average"), order)
        if not np.array_equal(average, restored_average):
            raise ValueError(f"average ranks are not exactly invariant for {name}")
        order_checks.append({
            **_order_record(name, seed, order),
            "average": _change(average, restored_average),
            "ordinal": _change(ordinal, restored_ordinal),
            "columns": [{"target": target,
                         "average": _change(average[:, j], restored_average[:, j]),
                         "ordinal": _change(ordinal[:, j], restored_ordinal[:, j])}
                        for j, target in enumerate(names)],
        })
    return {
        "n_rows": int(matrix.shape[0]), "n_columns": int(matrix.shape[1]),
        "input_dtype": str(matrix.dtype), "tie_equality": "exact",
        "target_names": names, "ranker_provenance": rankers.provenance,
        "ordinal_vs_average": _change(average, ordinal),
        "columns": columns, "order_checks": order_checks,
        "average_exactly_order_invariant": True,
        "ordinal_order_sensitive": any(not check["ordinal"]["exactly_equal"]
                                       for check in order_checks),
    }


def compare_pair_orders(first, second) -> dict:
    """Count exact pair-order changes from first to second for one target.

    A pair is one unordered pair of distinct rows. Ties use exact equality;
    inversions require strict inequalities of opposite signs. This deliberately
    reports sensitivity, never predictive quality.
    """
    first = _validate_values(first, "first scores", 1)
    second = _validate_values(second, "second scores", 1)
    if first.shape != second.shape:
        raise ValueError("pair scores: shape mismatch")
    counts = {"inversions": 0, "tied_to_untied": 0, "untied_to_tied": 0,
              "both_tied": 0, "same_strict_order": 0}
    # O(N^2) comparisons, O(N) temporary memory; no subtraction or dtype overflow.
    for row in range(len(first) - 1):
        before = ((first[row] > first[row + 1:]).astype(np.int8)
                  - (first[row] < first[row + 1:]).astype(np.int8))
        after = ((second[row] > second[row + 1:]).astype(np.int8)
                 - (second[row] < second[row + 1:]).astype(np.int8))
        counts["inversions"] += int(np.count_nonzero(before * after == -1))
        counts["tied_to_untied"] += int(np.count_nonzero((before == 0) & (after != 0)))
        counts["untied_to_tied"] += int(np.count_nonzero((before != 0) & (after == 0)))
        counts["both_tied"] += int(np.count_nonzero((before == 0) & (after == 0)))
        counts["same_strict_order"] += int(np.count_nonzero((before != 0) & (before == after)))
    total = len(first) * (len(first) - 1) // 2
    if sum(counts.values()) != total:
        raise AssertionError("pair-order counts must partition every pair")
    return {"n_pairs": total, **counts,
            "changed_pair_relations": counts["inversions"] + counts["tied_to_untied"]
                                      + counts["untied_to_tied"]}


def _fusion_scores(raw: np.ndarray, rankers: Rankers) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    own = np.mean(np.stack([_rank(raw[index], rankers.average, "own average")
                            for index in range(3)]), axis=0)
    public_average = _rank(raw[3], rankers.average, "public average")
    public_ordinal = _rank(raw[3], rankers.ordinal, "public ordinal")
    return own, 0.5 * own + 0.5 * public_average, 0.5 * own + 0.5 * public_ordinal


def _fusion_comparison(average: np.ndarray, ordinal: np.ndarray, names: list[str]) -> dict:
    return {
        "average_to_public_ordinal": _change(average, ordinal),
        "columns": [{"target": target,
                     "average_to_public_ordinal": _change(average[:, j], ordinal[:, j]),
                     "pair_orders_average_to_public_ordinal": compare_pair_orders(
                         average[:, j], ordinal[:, j])}
                    for j, target in enumerate(names)],
    }


def compare_fusions(raw, target_names: Sequence[str], rankers: Rankers | None = None) -> dict:
    """Change only public tie handling in the fixed half-own/half-public blend.

    Requires raw shape (4, N, 12), ordered fold0, fold1, fold2, public. The three
    own components always use average ranks; their arithmetic mean and the 50/50
    weights remain fixed. The principal score and pair-order comparisons follow
    source36's float32 cast before CSV serialization. Float64 comparisons before
    that cast are also included, explicitly separate from the emitted scores.
    """
    raw = _validate_values(raw, "raw", 3)
    if raw.shape[0] != 4 or raw.shape[2] != 12:
        raise ValueError("raw: expected shape (4, N, 12) in fold0/fold1/fold2/public order")
    names = _validate_names(target_names, 12)
    rankers = rankers or load_rankers()
    own, average64, ordinal64 = _fusion_scores(raw, rankers)
    average = average64.astype(np.float32)
    ordinal = ordinal64.astype(np.float32)
    comparison = _fusion_comparison(average, ordinal, names)
    comparison64 = _fusion_comparison(average64, ordinal64, names)
    checks = []
    for name, seed, order in _orders(raw.shape[1]):
        perm_own, perm_average, perm_ordinal = _fusion_scores(raw[:, order, :], rankers)
        restored_own = _restore_order(perm_own, order)
        restored_average64 = _restore_order(perm_average, order)
        restored_ordinal64 = _restore_order(perm_ordinal, order)
        if not np.array_equal(own, restored_own) or not np.array_equal(average64, restored_average64):
            raise ValueError(f"average fusion is not exactly invariant for {name}")
        restored_average = restored_average64.astype(np.float32)
        restored_ordinal = restored_ordinal64.astype(np.float32)
        checks.append({
            **_order_record(name, seed, order),
            "own_component": _change(own, restored_own),
            "average_fusion": _change(average, restored_average),
            "public_ordinal_fusion": _change(ordinal, restored_ordinal),
            "float64_public_ordinal_fusion_before_cast": _change(ordinal64, restored_ordinal64),
            "columns": [{"target": target,
                         "public_ordinal_fusion": _change(ordinal[:, j], restored_ordinal[:, j]),
                         "public_ordinal_pair_orders": compare_pair_orders(
                             ordinal[:, j], restored_ordinal[:, j])}
                        for j, target in enumerate(names)],
        })
    return {
        "raw_shape": [int(size) for size in raw.shape],
        "raw_arm_order": ["fold0", "fold1", "fold2", "public"],
        "weights": {"own_mean": 0.5, "public": 0.5},
        "own_tie_rule": "average", "changed_rule": "public average to source22 ordinal",
        "score_dtype": str(average.dtype), "serialization_applied": True,
        "serialization": "source36 float32 cast before CSV; no text round trip",
        "target_names": names, "ranker_provenance": rankers.provenance,
        **comparison, "float64_before_cast": comparison64, "order_checks": checks,
        "average_exactly_order_invariant": True,
        "public_ordinal_order_sensitive": any(not check["public_ordinal_fusion"]["exactly_equal"]
                                              for check in checks),
    }
