# Netflix Content Type Prediction

## Overview

A Streamlit machine-learning application that predicts
whether Netflix content is a Movie or TV Show.

## Technology

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Pytest

## Features

- Netflix dataset preprocessing
- Missing-value handling
- Categorical feature encoding
- Logistic Regression
- Decision Tree
- Random Forest
- Model comparison using validation F1
- Held-out test evaluation
- Saved machine-learning model
- Streamlit prediction interface
- Automated testing

## Installation

Use Python 3.11–3.13.

Install dependencies:

python -m pip install -r requirements.txt

## Model Training

python netflix_content_type_prediction.py

## Start Application

python -m streamlit run app.py

## Automated Tests

python -m pytest -q

## Notes

Movie = 0
TV Show = 1

This project demonstrates metadata classification.
Some Netflix genre labels reveal the content type
directly, so very high accuracy should be interpreted
with caution.

Only load trusted Joblib files.
