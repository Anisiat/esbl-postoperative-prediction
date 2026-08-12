"""Focused tests for the dependency-preserving synthetic-data step."""

import csv
import random
from pathlib import Path

import numpy as np

from src.generate_synthetic_data import (
    apply_categorical_pairs,
    apply_rank_dependencies,
)


def write_correlations(path: Path, correlation: float) -> None:
    """Write a two-variable correlation summary used by the test."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["variable_1", "variable_2", "correlation"],
        )
        writer.writeheader()
        for first, second, value in (
            ("x", "x", 1.0),
            ("x", "y", correlation),
            ("y", "x", correlation),
            ("y", "y", 1.0),
        ):
            writer.writerow(
                {"variable_1": first, "variable_2": second, "correlation": value}
            )


def test_rank_dependencies_preserve_margins_and_add_association(tmp_path: Path) -> None:
    """Reordering should retain values while creating the requested correlation."""
    correlation_path = tmp_path / "dependencies.csv"
    write_correlations(correlation_path, correlation=0.8)
    original = list(range(500))
    columns = {"x": original.copy(), "y": original[::-1]}

    apply_rank_dependencies(columns, correlation_path, seed=42)

    assert sorted(columns["x"]) == original
    assert sorted(columns["y"]) == original
    assert np.corrcoef(columns["x"], columns["y"])[0, 1] > 0.7


def test_square_correlation_matrix_is_supported(tmp_path: Path) -> None:
    """The conventional square export should drive the same rank dependence."""
    path = tmp_path / "correlations.csv"
    path.write_text("variable,x,y\nx,1,0.8\ny,0.8,1\n", encoding="utf-8")
    columns = {"x": list(range(500)), "y": list(reversed(range(500)))}

    apply_rank_dependencies(columns, path, seed=42)

    assert np.corrcoef(columns["x"], columns["y"])[0, 1] > 0.7


def test_categorical_pairs_preserve_joint_counts(tmp_path: Path) -> None:
    """Joint sampling should reproduce every exported crosstab cell exactly."""
    path = tmp_path / "pairs.csv"
    path.write_text(
        "variable_1,category_1,variable_2,category_2,count,proportion\n"
        "site,urine,organism,e_coli,3,0.6\n"
        "site,wound,organism,klebsiella,2,0.4\n",
        encoding="utf-8",
    )
    columns = {"site": ["x"] * 5, "organism": ["y"] * 5}

    apply_categorical_pairs(columns, path, n_rows=5, rng=random.Random(7))

    pairs = list(zip(columns["site"], columns["organism"]))
    assert pairs.count(("urine", "e_coli")) == 3
    assert pairs.count(("wound", "klebsiella")) == 2
