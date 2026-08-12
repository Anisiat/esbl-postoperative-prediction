"""Generate a synthetic ESBL modelling dataset from aggregate summaries.

Marginal CSV summaries reproduce column types, missingness, ranges, quantiles,
category frequencies, row count, and patient count. An optional Spearman
correlation summary adds dependencies between numeric predictors and their
missingness patterns. Logistic-regression coefficients add fitted dependencies
between predictors and ``esbl_status``.

Example
-------
python -m src.generate_synthetic_data
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "data" / "real_data_summary"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "synthetic_data.csv"
DEFAULT_COEFFICIENT_PATH = DEFAULT_SUMMARY_DIR / "logistic_coefficients.csv"
DEFAULT_DEPENDENCY_PATH = DEFAULT_SUMMARY_DIR / "dependency_correlations.csv"
DEFAULT_PAIRWISE_PATH = DEFAULT_SUMMARY_DIR / "categorical_pairwise_summary.csv"
MISSING_LABEL = "__MISSING__"
OUTCOME_COLUMN = "esbl_status"

QUANTILE_COLUMNS = (
    ("min", 0.00),
    ("p05", 0.05),
    ("q1", 0.25),
    ("median", 0.50),
    ("q3", 0.75),
    ("p95", 0.95),
    ("max", 1.00),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file and return its rows as dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"Required summary file not found: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Summary file is empty: {path}")

    return rows


def integer_metric(rows: list[dict[str, str]], metric: str) -> int:
    """Extract a whole-number metric from the dataset summary."""
    values = {row["metric"]: row["value"] for row in rows}
    if metric not in values:
        raise ValueError(f"Dataset summary does not contain {metric!r}.")
    return int(round(float(values[metric])))


def shuffled(values: Iterable[Any], rng: random.Random) -> list[Any]:
    """Return a shuffled copy of an iterable."""
    result = list(values)
    rng.shuffle(result)
    return result


def exact_categorical_sample(
    summary_rows: list[dict[str, str]],
    n_rows: int,
    rng: random.Random,
) -> list[Any]:
    """Create a categorical sample using the published counts exactly."""
    values: list[Any] = []
    for row in summary_rows:
        value: Any = None if row["category"] == MISSING_LABEL else row["category"]
        values.extend([value] * int(row["count"]))

    if len(values) != n_rows:
        raise ValueError(
            "Categorical counts do not sum to the dataset row count "
            f"({len(values)} != {n_rows})."
        )
    return shuffled(values, rng)


def apply_categorical_pairs(
    columns: dict[str, list[Any]],
    pairwise_path: Path,
    n_rows: int,
    rng: random.Random,
) -> None:
    """Replace independent categories with exact selected joint counts."""
    if not pairwise_path.exists():
        return

    rows = read_csv(pairwise_path)
    required = {
        "variable_1", "category_1", "variable_2", "category_2", "count"
    }
    if not required.issubset(rows[0]):
        raise ValueError(
            "Pairwise CSV must contain variable/category columns and count."
        )

    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["variable_1"], row["variable_2"]), []).append(row)

    used_variables: set[str] = set()
    for (first, second), pair_rows in groups.items():
        if first in used_variables or second in used_variables:
            raise ValueError("Categorical pairs must not share variables.")
        if first not in columns or second not in columns:
            raise ValueError(f"Unknown categorical pair: {first}, {second}.")

        paired_values: list[tuple[Any, Any]] = []
        for row in pair_rows:
            first_value = None if row["category_1"] == MISSING_LABEL else row["category_1"]
            second_value = None if row["category_2"] == MISSING_LABEL else row["category_2"]
            paired_values.extend(
                [(first_value, second_value)] * int(row["count"])
            )
        if len(paired_values) != n_rows:
            raise ValueError(
                f"Joint counts for {first}/{second} do not sum to {n_rows}."
            )

        rng.shuffle(paired_values)
        columns[first] = [value[0] for value in paired_values]
        columns[second] = [value[1] for value in paired_values]
        used_variables.update((first, second))


def interpolate_quantiles(summary: dict[str, str], probability: float) -> float:
    """Sample from a piecewise-linear CDF defined by reported quantiles."""
    points = [(probability_, float(summary[column])) for column, probability_ in QUANTILE_COLUMNS]

    intervals = list(zip(points, points[1:]))
    for interval_index, (
        (lower_p, lower_value),
        (upper_p, upper_value),
    ) in enumerate(intervals):
        if probability <= upper_p:
            width = upper_p - lower_p
            fraction = (probability - lower_p) / width if width else 0.0

            # Extreme min/max values are often isolated clinical outliers. A
            # straight line through the outer 5% would create too many extreme
            # observations, so gently concentrate both tails near p05/p95.
            if interval_index == 0:
                fraction = fraction ** (1 / 8)
            elif interval_index == len(intervals) - 1:
                fraction = fraction**8
            return lower_value + fraction * (upper_value - lower_value)

    return points[-1][1]


def continuous_sample(
    summary: dict[str, str],
    dtype: str,
    n_rows: int,
    missing_count: int,
    rng: random.Random,
) -> list[Any]:
    """Generate continuous values from summary quantiles and missingness."""
    n_observed = n_rows - missing_count
    values: list[Any] = []

    # Stratification covers the CDF evenly and reduces run-to-run sampling noise.
    probabilities = [(index + rng.random()) / n_observed for index in range(n_observed)]
    rng.shuffle(probabilities)

    for probability in probabilities:
        value = interpolate_quantiles(summary, probability)
        if dtype.startswith("int"):
            value = int(round(value))
        else:
            value = round(value, 4)
        values.append(value)

    values.extend([None] * missing_count)
    return shuffled(values, rng)


def synthetic_subjects(n_rows: int, n_patients: int, rng: random.Random) -> list[str]:
    """Create pseudonymous patient IDs, including repeated episodes."""
    if not 1 <= n_patients <= n_rows:
        raise ValueError("Unique patient count must be between 1 and the row count.")

    patient_ids = [f"SYN-P{index:06d}" for index in range(1, n_patients + 1)]
    subjects = patient_ids.copy()
    subjects.extend(rng.choices(patient_ids, k=n_rows - n_patients))
    rng.shuffle(subjects)
    return subjects


def synthetic_dates(
    n_rows: int,
    start_date: date,
    end_date: date,
    rng: random.Random,
) -> list[str]:
    """Generate ISO-formatted dates within a user-specified interval."""
    span = (end_date - start_date).days
    if span < 0:
        raise ValueError("The end date must not be before the start date.")
    return [
        (start_date + timedelta(days=rng.randint(0, span))).isoformat()
        for _ in range(n_rows)
    ]


def coerce_categories(
    columns: dict[str, list[Any]],
    dtypes: dict[str, str],
) -> None:
    """Convert numeric categories read from CSV text back to numbers."""
    for name, values in columns.items():
        dtype = dtypes.get(name, "")
        if not (dtype.startswith("int") or dtype.startswith("float")):
            continue
        for index, value in enumerate(values):
            if value is None:
                continue
            number = float(value)
            values[index] = int(number) if dtype.startswith("int") else number


def nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    """Return a valid correlation matrix close to a noisy exported matrix."""
    matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Pairwise correlations and rounding can make the matrix slightly invalid.
    matrix = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-8, None)) @ eigenvectors.T
    scale = np.sqrt(np.diag(matrix))
    return matrix / np.outer(scale, scale)


def apply_rank_dependencies(
    columns: dict[str, list[Any]],
    correlations_path: Path,
    seed: int,
) -> None:
    """Reorder marginal samples to reproduce exported rank correlations.

    This is a small Gaussian rank-copula: it changes which generated values
    occur together, but not the values, category counts, or missing counts.
    """
    if not correlations_path.exists():
        # Older summary bundles remain usable, but generate independent inputs.
        return

    rows = read_csv(correlations_path)
    long_format = {"variable_1", "variable_2", "correlation"}.issubset(rows[0])
    if long_format:
        # Compatibility with summary bundles created by the earlier script.
        all_variables = sorted({row["variable_1"] for row in rows})
    elif "variable" in rows[0]:
        all_variables = [name for name in rows[0] if name != "variable"]
    else:
        raise ValueError(
            "Dependency CSV must be a square matrix with a 'variable' column."
        )

    variables = [
        variable
        for variable in all_variables
        if variable.removeprefix("__missing__") in columns
    ]
    if not variables:
        return

    positions = {name: index for index, name in enumerate(variables)}
    matrix = np.eye(len(variables))
    if long_format:
        for row in rows:
            first, second = row["variable_1"], row["variable_2"]
            if first in positions and second in positions:
                matrix[positions[first], positions[second]] = float(
                    row["correlation"]
                )
    else:
        for row in rows:
            first = row["variable"]
            if first not in positions:
                continue
            for second in variables:
                matrix[positions[first], positions[second]] = float(row[second])

    matrix = nearest_correlation(matrix)
    n_rows = len(next(iter(columns.values())))
    latent = np.random.default_rng(seed).multivariate_normal(
        mean=np.zeros(len(variables)), cov=matrix, size=n_rows
    )

    for variable in variables:
        if variable.startswith("__missing__"):
            continue
        values = columns[variable]
        value_scores = latent[:, positions[variable]]
        missing_name = f"__missing__{variable}"

        if missing_name in positions:
            missing_count = sum(value is None for value in values)
            missing_scores = latent[:, positions[missing_name]]
            missing_indices = set(np.argsort(missing_scores)[-missing_count:])
        else:
            missing_indices = set()

        observed_indices = [i for i in range(n_rows) if i not in missing_indices]
        observed_values = sorted(value for value in values if value is not None)
        ranked_indices = sorted(observed_indices, key=lambda i: value_scores[i])
        reordered: list[Any] = [None] * n_rows
        for index, value in zip(ranked_indices, observed_values):
            reordered[index] = value
        columns[variable] = reordered


def load_coefficients(path: Path) -> tuple[float, dict[str, float]]:
    """Load a model intercept and log-odds coefficients."""
    rows = read_csv(path)
    if not {"variable", "coefficient"}.issubset(rows[0]):
        raise ValueError(
            "Coefficient CSV must contain 'variable' and 'coefficient' columns."
        )

    intercept: float | None = None
    coefficients: dict[str, float] = {}
    for row in rows:
        variable = row["variable"].strip()
        if not variable:
            continue
        coefficient = float(row["coefficient"])
        if variable in {"const", "intercept", "Intercept"}:
            if intercept is not None:
                raise ValueError("Coefficient CSV contains multiple intercepts.")
            intercept = coefficient
        else:
            coefficients[variable] = coefficient

    if intercept is None:
        raise ValueError("Coefficient CSV does not contain an intercept.")
    if not coefficients:
        raise ValueError("Coefficient CSV does not contain any predictors.")
    return intercept, coefficients


def split_dummy_term(term: str, fieldnames: list[str]) -> tuple[str, str] | None:
    """Split ``variable__category`` using the longest matching variable name."""
    matches = [
        variable
        for variable in fieldnames
        if term.startswith(f"{variable}__")
    ]
    if not matches:
        return None
    variable = max(matches, key=len)
    return variable, term[len(variable) + 2 :]


def model_value(
    value: Any,
    variable: str,
    continuous_medians: dict[str, float],
) -> float:
    """Return a numeric model value, median-imputing missing continuous data."""
    if value is None:
        if variable not in continuous_medians:
            # Binary missing values are not present in the current summaries.
            return 0.0
        return continuous_medians[variable]
    return float(value)


def assign_outcome_from_model(
    columns: dict[str, list[Any]],
    fieldnames: list[str],
    coefficients_path: Path,
    continuous_summaries: list[dict[str, str]],
    n_positive: int,
    rng: random.Random,
) -> None:
    """Assign outcomes using logistic coefficients and a fixed case count.

    Independent standard logistic noise is added to each linear predictor and
    the rows with the largest latent scores are labelled positive. Conditioning
    on the published number of cases preserves outcome prevalence exactly while
    retaining the coefficient-implied ordering of individual risks.

    Missing continuous model inputs are median-imputed. A missing categorical
    value activates no dummy term and is therefore treated as the reference
    level, matching the common drop-first design-matrix convention.
    """
    intercept, coefficients = load_coefficients(coefficients_path)
    continuous_medians = {
        row["variable"]: float(row["median"]) for row in continuous_summaries
    }
    direct_terms: list[tuple[str, float]] = []
    dummy_terms: list[tuple[str, str, float]] = []

    for term, coefficient in coefficients.items():
        if term in fieldnames:
            if term == OUTCOME_COLUMN:
                raise ValueError("Outcome must not be included as its own predictor.")
            direct_terms.append((term, coefficient))
            continue
        split = split_dummy_term(term, fieldnames)
        if split is None:
            raise ValueError(
                f"Coefficient term {term!r} cannot be mapped to a dataset column."
            )
        variable, category = split
        dummy_terms.append((variable, category, coefficient))

    n_rows = len(next(iter(columns.values())))
    if not 0 <= n_positive <= n_rows:
        raise ValueError("Positive outcome count is outside the dataset row range.")

    latent_scores: list[tuple[float, int]] = []
    for index in range(n_rows):
        score = intercept
        for variable, coefficient in direct_terms:
            score += coefficient * model_value(
                columns[variable][index],
                variable,
                continuous_medians,
            )
        for variable, category, coefficient in dummy_terms:
            observed_category = columns[variable][index]
            category_matches = observed_category == category
            if isinstance(observed_category, str):
                category_matches = category_matches or (
                    observed_category.replace(" ", "_") == category
                )
            if category_matches:
                score += coefficient

        # Inverse-CDF sampling from a standard logistic distribution.
        uniform = rng.random()
        logistic_noise = math.log(uniform / (1.0 - uniform))
        latent_scores.append((score + logistic_noise, index))

    positive_indices = {
        index
        for _, index in sorted(latent_scores, reverse=True)[:n_positive]
    }
    columns[OUTCOME_COLUMN] = [
        int(index in positive_indices) for index in range(n_rows)
    ]


def build_dataset(
    summary_dir: Path,
    coefficients_path: Path,
    correlations_path: Path,
    seed: int,
    start_date: date,
    end_date: date,
    pairwise_path: Path | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build synthetic records from aggregate summaries and model coefficients."""
    dataset_summary = read_csv(summary_dir / "dataset_summary.csv")
    missingness_summary = read_csv(summary_dir / "missingness_summary.csv")
    continuous_summary = read_csv(summary_dir / "continuous_summary.csv")
    categorical_summary = read_csv(summary_dir / "categorical_summary.csv")

    n_rows = integer_metric(dataset_summary, "n_rows")
    n_patients = integer_metric(dataset_summary, "n_unique_patients")
    expected_columns = integer_metric(dataset_summary, "n_columns")
    rng = random.Random(seed)

    dtypes = {row["variable"]: row["dtype"] for row in missingness_summary}
    missing_counts = {
        row["variable"]: int(row["missing_count"]) for row in missingness_summary
    }
    fieldnames = [row["variable"] for row in missingness_summary]

    categorical_groups: dict[str, list[dict[str, str]]] = {}
    for row in categorical_summary:
        categorical_groups.setdefault(row["variable"], []).append(row)

    columns: dict[str, list[Any]] = {}
    for variable, rows in categorical_groups.items():
        columns[variable] = exact_categorical_sample(rows, n_rows, rng)

    # Selected pairs are sampled jointly, preserving both their crosstab and
    # the exact one-variable category counts.
    apply_categorical_pairs(
        columns,
        pairwise_path or summary_dir / "categorical_pairwise_summary.csv",
        n_rows,
        rng,
    )

    for summary in continuous_summary:
        variable = summary["variable"]
        columns[variable] = continuous_sample(
            summary=summary,
            dtype=dtypes.get(variable, "float64"),
            n_rows=n_rows,
            missing_count=missing_counts.get(variable, 0),
            rng=rng,
        )

    # These fields cannot be reconstructed from aggregate distributions and are
    # therefore generated independently with unmistakably synthetic values.
    columns["subject"] = synthetic_subjects(n_rows, n_patients, rng)
    columns["infection_id"] = [
        f"SYN-I{index:07d}" for index in range(1, n_rows + 1)
    ]
    columns["admission_date"] = synthetic_dates(
        n_rows, start_date, end_date, rng
    )

    missing_variables = sorted(set(fieldnames).difference(columns))
    if missing_variables:
        raise ValueError(
            "No generation rule is available for variables: "
            + ", ".join(missing_variables)
        )
    if len(fieldnames) != expected_columns:
        raise ValueError(
            "Missingness summary column count does not match dataset summary "
            f"({len(fieldnames)} != {expected_columns})."
        )

    coerce_categories(columns, dtypes)
    apply_rank_dependencies(columns, correlations_path, seed)
    assign_outcome_from_model(
        columns=columns,
        fieldnames=fieldnames,
        coefficients_path=coefficients_path,
        continuous_summaries=continuous_summary,
        n_positive=integer_metric(dataset_summary, "n_outcome_positive"),
        rng=rng,
    )
    records = [
        {field: columns[field][row_index] for field in fieldnames}
        for row_index in range(n_rows)
    ]
    return fieldnames, records


def write_dataset(
    output_path: Path,
    fieldnames: list[str],
    records: list[dict[str, Any]],
) -> None:
    """Write synthetic records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_date(value: str) -> date:
    """Parse an ISO date for argparse."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid YYYY-MM-DD date"
        ) from error


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=DEFAULT_COEFFICIENT_PATH,
        help="CSV containing logistic-regression terms and coefficients.",
    )
    parser.add_argument(
        "--correlations",
        type=Path,
        default=DEFAULT_DEPENDENCY_PATH,
        help="Optional CSV of Spearman predictor and missingness correlations.",
    )
    parser.add_argument(
        "--categorical-pairs",
        type=Path,
        default=DEFAULT_PAIRWISE_PATH,
        help="Optional CSV of selected categorical pair counts.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=date(2015, 1, 1),
        help="Earliest synthetic admission date (default: 2015-01-01).",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=date(2025, 9, 1),
        help="Latest synthetic admission date (default: 2025-09-01).",
    )
    return parser.parse_args()


def main() -> None:
    """Generate and save the synthetic dataset."""
    args = parse_args()
    fieldnames, records = build_dataset(
        summary_dir=args.summary_dir,
        coefficients_path=args.coefficients,
        correlations_path=args.correlations,
        pairwise_path=args.categorical_pairs,
        seed=args.seed,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    write_dataset(args.output, fieldnames, records)
    print(
        f"Saved {len(records):,} rows and {len(fieldnames)} columns "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
