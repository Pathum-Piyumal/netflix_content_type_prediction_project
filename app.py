
"""Streamlit frontend for Netflix content type prediction."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT / "saved_models" / "netflix_content_type_model.joblib"
)

LABELS = {
    0: "MOVIE",
    1: "TV SHOW",
}

MONTHS = [
    "Unknown",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

st.set_page_config(
    page_title="Netflix Content Type Predictor",
    page_icon="🎬",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main-title {
        color: #E50914;
        text-align: center;
        font-weight: bold;
        font-size: 2.3rem;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: #E50914;
        color: white;
        width: 100%;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="main-title">🎬 Netflix Content Type Predictor</h1>',
    unsafe_allow_html=True,
)

st.caption(
    "Predict whether Netflix metadata represents "
    "a Movie or TV Show."
)


@st.cache_resource
def load_model(path: str, modified: float):
    return joblib.load(path)


if not MODEL_PATH.is_file():
    st.error(
        "Trained model not found. "
        "Please run: python netflix_content_type_prediction.py"
    )
    st.stop()


try:
    pipeline = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )

    if set(pipeline.classes_) != {0, 1}:
        raise ValueError(
            "Unexpected model classes. Retrain the model."
        )

except Exception as exc:
    st.error(
        f"Unable to load the prediction model: {exc}"
    )
    st.info(
        "Run the training script to rebuild the model."
    )
    st.stop()


with st.form("prediction_form"):

    st.subheader("Enter Netflix Content Details")

    left, right = st.columns(2)

    with left:

        release_year = st.number_input(
            "Release year",
            min_value=1900,
            max_value=2100,
            value=2021,
        )

        year_added_unknown = st.checkbox(
            "Date added unknown"
        )

        year_added = st.number_input(
            "Year added to Netflix",
            min_value=2007,
            max_value=2100,
            value=2021,
            disabled=year_added_unknown,
        )

        month_added = st.selectbox(
            "Month added",
            MONTHS,
            index=1,
        )

        has_director = st.checkbox(
            "Director is known",
            value=True,
        )

    with right:

        rating = st.selectbox(
            "Content rating",
            [
                "Unknown",
                "TV-MA",
                "TV-14",
                "TV-PG",
                "R",
                "PG-13",
                "TV-Y7",
                "TV-Y",
                "PG",
                "TV-G",
                "NR",
                "G",
                "UR",
                "NC-17",
            ],
        )

        primary_country = st.text_input(
            "Primary country",
            value="United States",
            max_chars=100,
        )

        primary_genre = st.text_input(
            "Primary genre",
            value="Dramas",
            max_chars=100,
            help="Enter the primary Netflix genre.",
        )

    submitted = st.form_submit_button(
        "Predict Content Type"
    )


if submitted:

    data = pd.DataFrame(
        [
            {
                "release_year": release_year,
                "year_added": (
                    float("nan")
                    if year_added_unknown
                    else year_added
                ),
                "has_director": int(has_director),
                "rating": (
                    rating
                    if rating != "Unknown"
                    else None
                ),
                "primary_country": (
                    primary_country.strip()
                    or "Unknown"
                ),
                "primary_genre": (
                    primary_genre.strip()
                    or "Unknown"
                ),
                "month_added": (
                    "Unknown"
                    if year_added_unknown
                    else month_added
                ),
            }
        ]
    )

    try:

        prediction = int(
            pipeline.predict(data)[0]
        )

        probability_values = (
            pipeline.predict_proba(data)[0]
        )

        probabilities = dict(
            zip(
                pipeline.classes_,
                probability_values,
            )
        )

        movie_probability = float(
            probabilities[0]
        )

        tv_probability = float(
            probabilities[1]
        )

        predicted_probability = float(
            probabilities[prediction]
        )

        st.success(
            f"Predicted Content Type: "
            f"**{LABELS[prediction]}**"
        )

        st.metric(
            "Estimated Prediction Probability",
            f"{predicted_probability:.1%}",
        )

        st.subheader("Prediction Probabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Movie",
                f"{movie_probability:.1%}",
            )

        with col2:
            st.metric(
                "TV Show",
                f"{tv_probability:.1%}",
            )

        st.progress(
            tv_probability,
            text="TV Show Probability",
        )

        st.info(
            "Predictions are estimates based on "
            "the model's training data."
        )

    except Exception as exc:

        st.error(
            f"Prediction failed: {exc}"
        )

        st.info(
            "Check your inputs and make sure "
            "the model was trained successfully."
        )


st.divider()

st.caption(
    "Netflix Content Type Classification | "
    "Machine Learning Project"
)

st.caption(
    "Some genre categories explicitly identify "
    "Movies or TV Shows. High classification "
    "accuracy does not guarantee performance "
    "on other datasets."
)
