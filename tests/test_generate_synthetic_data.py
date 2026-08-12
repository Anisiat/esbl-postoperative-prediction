"""Focused tests for the dependency-preserving synthetic-data step."""

import csv
from pathlib import Path

import numpy as np

from src.generate_synthetic_data import apply_rank_dependencies


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
