
"""Reproducible Netflix Movie / TV Show classifier training CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent

FEATURES = [
    "release_year",
    "year_added",
    "has_director",
    "rating",
    "primary_country",
    "primary_genre",
    "month_added",
]

NUMERIC = [
    "release_year",
    "year_added",
    "has_director",
]

CATEGORICAL = [
    "rating",
    "primary_country",
    "primary_genre",
    "month_added",
]

TARGET = {"Movie": 0, "TV Show": 1}


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required = {
        "type",
        "release_year",
        "date_added",
        "director",
        "rating",
        "country",
        "listed_in",
    }

    missing = required - set(raw.columns)

    if missing:
        raise ValueError(
            f"Dataset missing required columns: {', '.join(sorted(missing))}"
        )

    df = raw.loc[raw["type"].isin(TARGET)].copy()

    if df.empty or df["type"].nunique() != 2:
        raise ValueError(
            "Dataset must contain both Movie and TV Show examples"
        )

    dates = pd.to_datetime(
        df["date_added"].astype("string").str.strip(),
        errors="coerce",
        format="%B %d, %Y",
    )

    primary_country = (
        df["country"]
        .fillna("Unknown")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
        .replace("", "Unknown")
    )

    primary_genre = (
        df["listed_in"]
        .fillna("Unknown")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
        .replace("", "Unknown")
    )

    X = pd.DataFrame(
        {
            "release_year": pd.to_numeric(
                df["release_year"], errors="coerce"
            ),
            "year_added": dates.dt.year.astype(float),
            "has_director": df["director"].notna().astype(int),
            "rating": df["rating"].replace("", np.nan),
            "primary_country": primary_country,
            "primary_genre": primary_genre,
            "month_added": dates.dt.month_name().fillna("Unknown"),
        },
        index=df.index,
    )

    y = df["type"].map(TARGET).astype(int)

    return X[FEATURES], y


def make_pipeline(classifier) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessing),
            ("classifier", classifier),
        ]
    )


def metrics(y_true, y_pred):
    return {
        "accuracy": round(
            float(accuracy_score(y_true, y_pred)), 4
        ),
        "precision_tv_show": round(
            float(
                precision_score(
                    y_true, y_pred, zero_division=0
                )
            ),
            4,
        ),
        "recall_tv_show": round(
            float(
                recall_score(
                    y_true, y_pred, zero_division=0
                )
            ),
            4,
        ),
        "f1_tv_show": round(
            float(f1_score(y_true, y_pred, zero_division=0)),
            4,
        ),
        "confusion_matrix_movie_tv_show": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


def run_pipeline(
    data_path: Path | str = ROOT / "netflix_titles.csv",
    output_dir: Path | str = ROOT / "saved_models",
) -> dict:

    data_path = Path(data_path)
    output_dir = Path(output_dir)

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    raw_data = pd.read_csv(data_path)
    X, y = prepare_data(raw_data)

    # Reserve 20% for final evaluation.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Split development data into training/validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=42,
        stratify=y_dev,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    validation = {}

    for name, classifier in candidates.items():
        print(f"Training: {name}")

        model = make_pipeline(classifier)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        validation[name] = metrics(y_val, predictions)

    best_name = max(
        validation,
        key=lambda name: validation[name]["f1_tv_show"],
    )

    print(f"Selected model: {best_name}")

    final_model = make_pipeline(candidates[best_name])
    final_model.fit(X_dev, y_dev)

    test_predictions = final_model.predict(X_test)
    test_metrics = metrics(y_test, test_predictions)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir / "netflix_content_type_model.joblib"
    )

    joblib.dump(final_model, model_path)

    report = {
        "selected_model": best_name,
        "dataset_rows": int(len(X)),
        "splits": {
            "training": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "validation_results": validation,
        "held_out_test": test_metrics,
        "features": FEATURES,
        "note": (
            "Some Netflix genre names directly reveal the "
            "content type. Very high scores may reflect "
            "this metadata convention."
        ),
    }

    report_path = output_dir / "training_report.json"

    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"Model saved to: {model_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "netflix_titles.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "saved_models",
    )

    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
    )
