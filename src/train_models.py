import argparse
import logging
from pathlib import Path

import pandas as pd
from rich.logging import RichHandler
from rich.markup import escape
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


if __package__:
    from .preprocess_data import (
        build_preprocessor,
        get_feature_types,
        get_features_target,
        load_data,
    )
else:
    from preprocess_data import (
        build_preprocessor,
        get_feature_types,
        get_features_target,
        load_data,
    )

INPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "splits"
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "model_results"
GROUP_COLUMN = "subject"  # Column used for grouping in StratifiedGroupKFold
N_SPLITS = 5
RANDOM_STATE = 42
RICH_MARKUP = {"markup": True}


def configure_terminal_logging() -> None:
    """Configure clear, colour-coded logging for this command-line script."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
                markup=False,
            )
        ],
        # Replace any existing root handlers so this command always uses Rich.
        force=True,
    )


def log_section(title: str) -> None:
    """Write a visually distinct section heading to the terminal."""

    logging.info(
        "[bold cyan]-- %s --[/bold cyan]",
        escape(title),
        extra=RICH_MARKUP,
    )


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    cv,
    groups: pd.Series,
    models: dict,
    scoring: list[str],
) -> list[dict]:
    """Evaluate preprocessing-and-model pipelines using grouped CV."""

    if not models:
        raise ValueError("At least one model must be provided.")

    results = []
    n_splits = cv.get_n_splits(X_train, y_train, groups)

    for model_name, model in models.items():
        logging.info(
            "Evaluating model: [bold magenta]%s[/bold magenta]",
            escape(model_name),
            extra=RICH_MARKUP,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        model_cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            groups=groups,
            scoring=scoring,
            return_train_score=True,
            error_score="raise",
        )

        for fold in range(n_splits):
            fold_result = {
                "model": model_name,
                "fold": fold + 1,
            }

            for metric in scoring:
                train_value = model_cv_results[f"train_{metric}"][fold]
                fold_result[f"train_{metric}"] = (
                    -train_value if metric.startswith("neg_") else train_value
                )
                test_value = model_cv_results[f"test_{metric}"][fold]
                output_name = metric.removeprefix("neg_")
                fold_result[output_name] = (
                    -test_value if metric.startswith("neg_") else test_value
                )

            results.append(fold_result)

    return results


def get_summary_table(cv_results: list[dict]) -> pd.DataFrame:
    """Generate a summary table of mean and std for each model and metric."""

    if not cv_results:
        raise ValueError("cv_results must not be empty.")

    results_df = pd.DataFrame(cv_results)
    summary = (
        results_df.groupby("model")
        .agg(
            **{
                f"{metric}_mean": (metric, "mean")
                for metric in results_df.columns
                if metric not in ["model", "fold"]
            },
            **{
                f"{metric}_std": (metric, "std")
                for metric in results_df.columns
                if metric not in ["model", "fold"]
            },
        )
        .reset_index()
    )

    return summary


def main() -> None:
    """Run grouped cross-validation and save fold-level results."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate model pipelines on raw split data using grouped "
            "cross-validation."
        )
    )
    parser.add_argument(
        "--input_directory",
        type=Path,
        default=INPUT_DIRECTORY,
        help="Directory containing the raw train/test split files.",
    )
    parser.add_argument(
        "--output_directory",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help="Directory where cross-validation results will be saved.",
    )
    parser.add_argument(
        "--split_type",
        choices=["random", "temporal"],
        default="random",
        help="Type of data split to use.",
    )
    parser.add_argument(
        "--target_column",
        default="esbl_status",
        help="Name of the binary target column.",
    )
    parser.add_argument(
        "--group_column",
        default=GROUP_COLUMN,
        help="Patient/group column used by StratifiedGroupKFold.",
    )

    args = parser.parse_args()

    configure_terminal_logging()

    logging.info(
        "[bold blue]ESBL postoperative model evaluation[/bold blue]",
        extra=RICH_MARKUP,
    )
    logging.info("Input directory  : %s", args.input_directory)
    logging.info("Output directory : %s", args.output_directory)
    logging.info("Data split       : %s", args.split_type)

    log_section("Load training data")
    logging.info("Loading data...")
    train_data, _ = load_data(args.input_directory, args.split_type, args.target_column)

    if args.group_column not in train_data.columns:
        raise ValueError(
            f"Group column '{args.group_column}' not found in the training dataset."
        )

    # Get group labels before identifier columns are removed from the features.
    groups = train_data[args.group_column]

    if groups.isna().any():
        raise ValueError(f"Group column '{args.group_column}' contains missing values.")

    logging.info("Training rows   : %d", len(train_data))
    logging.info("Unique patients : %d", groups.nunique())

    # Get features and target
    log_section("Prepare features")
    logging.info("Target column: %s", args.target_column)
    X_train, y_train = get_features_target(train_data, args.target_column)

    # A custom group column may not be one of the standard identifier columns.
    X_train = X_train.drop(columns=args.group_column, errors="ignore")

    if not len(X_train) == len(y_train) == len(groups):
        raise ValueError("Length of features, target, and groups must be the same.")

    if y_train.isna().any() or set(y_train.unique()) != {0, 1}:
        raise ValueError(
            "The target must contain both binary classes 0 and 1, "
            "with no missing values."
        )

    if groups.nunique() < N_SPLITS:
        raise ValueError(
            f"At least {N_SPLITS} unique groups are required for cross-validation."
        )

    for target_value in (0, 1):
        class_group_count = groups[y_train == target_value].nunique()
        if class_group_count < N_SPLITS:
            raise ValueError(
                f"Target class {target_value} occurs in only "
                f"{class_group_count} groups; "
                f"at least {N_SPLITS} are required."
            )

    # Get numerical, categorical, and binary features.
    logging.info("Identifying feature types...")
    numerical_features, categorical_features, binary_features = get_feature_types(
        X_train
    )

    logging.info(
        "Numerical features   (%d): %s",
        len(numerical_features),
        ", ".join(numerical_features) or "None",
    )
    logging.info(
        "Categorical features (%d): %s",
        len(categorical_features),
        ", ".join(categorical_features) or "None",
    )
    logging.info(
        "Binary features      (%d): %s",
        len(binary_features),
        ", ".join(binary_features) or "None",
    )

    # Preprocessing remains inside the pipeline so it is fitted per CV fold.
    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
        binary_features,
    )

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    models = {
        "Dummy Classifier": DummyClassifier(
            strategy="prior",
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression L2": LogisticRegression(
        l1_ratio=0,
        max_iter=1000,
        random_state=RANDOM_STATE,
        ),
        "Logistic Regression L1": LogisticRegression(
            l1_ratio=1,
            solver="saga",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression Elastic Net": LogisticRegression(
            l1_ratio=0.5,
            solver="saga",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(),

        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),  
    }

    scoring = [
        "roc_auc",
        "average_precision",
        "neg_brier_score",
    ]

    log_section("Cross-validation")
    logging.info("Folds   : %d", N_SPLITS)
    logging.info("Models  : %s", ", ".join(models))
    logging.info(
        "Metrics : %s",
        ", ".join(metric.removeprefix("neg_") for metric in scoring),
    )
    cv_results = train_and_evaluate_models(
        X_train,
        y_train,
        preprocessor,
        cv,
        groups,
        models,
        scoring,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)

    cv_summary_df = get_summary_table(cv_results)
    summary_output_path = args.output_directory / f"cv_summary_{args.split_type}.csv"
    cv_summary_df.to_csv(summary_output_path, index=False)

    cv_results_df = pd.DataFrame(cv_results)
    output_path = args.output_directory / f"cv_results_{args.split_type}.csv"
    cv_results_df.to_csv(output_path, index=False)

    log_section("Results")
    logging.info(
        "[bold green]Model evaluation completed successfully.[/bold green]",
        extra=RICH_MARKUP,
    )
    logging.info("Summary results : %s", summary_output_path)
    logging.info("Fold results    : %s", output_path)



if __name__ == "__main__":
    main()
