
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from netflix_content_type_prediction import (
    ROOT,
    FEATURES,
    prepare_data,
    run_pipeline,
)


def test_preparation_does_not_leak_target_columns():

    raw = pd.read_csv(
        ROOT / "netflix_titles.csv"
    )

    X, y = prepare_data(raw)

    assert X.columns.tolist() == FEATURES

    excluded = {
        "type",
        "duration",
        "title",
        "description",
        "show_id",
    }

    assert not excluded & set(X.columns)

    assert len(X) == len(y)

    assert set(y) == {0, 1}

    assert X["month_added"].notna().all()


def test_missing_required_columns():

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        prepare_data(
            pd.DataFrame(
                {"type": ["Movie"]}
            )
        )


def test_training_end_to_end(tmp_path):

    report = run_pipeline(
        ROOT / "netflix_titles.csv",
        tmp_path,
    )

    assert (
        0
        <= report["held_out_test"]["f1_tv_show"]
        <= 1
    )

    assert (
        report["selected_model"]
        in report["validation_results"]
    )

    assert (
        tmp_path / "training_report.json"
    ).is_file()

    model = joblib.load(
        tmp_path
        / "netflix_content_type_model.joblib"
    )

    X, _ = prepare_data(
        pd.read_csv(
            ROOT / "netflix_titles.csv"
        )
    )

    predicted = model.predict(
        X.iloc[:3]
    )

    probability = model.predict_proba(
        X.iloc[:3]
    )

    assert len(predicted) == 3

    assert probability.shape == (3, 2)

    assert np.allclose(
        probability.sum(axis=1),
        1,
    )


def test_unknown_categories_supported(tmp_path):

    run_pipeline(
        ROOT / "netflix_titles.csv",
        tmp_path,
    )

    model = joblib.load(
        tmp_path
        / "netflix_content_type_model.joblib"
    )

    X, _ = prepare_data(
        pd.read_csv(
            ROOT / "netflix_titles.csv"
        )
    )

    row = X.iloc[:1].copy()

    row["primary_country"] = "Atlantis"

    row["primary_genre"] = (
        "Unknown Future Genre"
    )

    row["year_added"] = np.nan

    prediction = model.predict(row)[0]

    assert prediction in [0, 1]
