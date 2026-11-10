import os
import json
import pandas as pd

from config import config
from data_pipeline.feature_builder import build_feature_frame
from modeling.model_registry import save_model, save_feature_schema
from modeling.amendment_model import create_amendment_model
from modeling.exposure_model import create_exposure_model

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score


def main():
    os.makedirs("output/models", exist_ok=True)

    metrics = {}

    # ---------------------------------------------------------
    # 1. Amendment classification
    # ---------------------------------------------------------

    d = pd.read_csv(config.CLEAN_SUBMISSIONS_TRAIN_FILE)

    y = d[config.SUBMISSION_TARGET_COLUMN].map(
        config.SUBMISSION_TARGET_ENCODING
    )

    valid = y.notna()
    d = d.loc[valid].copy()
    y = y.loc[valid].astype(int)

    numeric = config.AMENDMENT_NUMERIC_FEATURES
    categorical = config.AMENDMENT_CATEGORICAL_FEATURES
    schema = numeric + categorical

    X = build_feature_frame(d, numeric, categorical)[schema]

    base_model = create_amendment_model(numeric, categorical)

    amendment_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=3
    )

    amendment_model.fit(X, y)

    save_model(
        amendment_model,
        config.AMENDMENT_MODEL_FILE
    )

    save_feature_schema(
        schema,
        config.AMENDMENT_SCHEMA_FILE
    )

    # Evaluate the saved model on the first 6,000 training rows.
    metrics_train_frame = pd.read_csv(
        config.CLEAN_SUBMISSIONS_TRAIN_FILE
    ).head(6000)

    metrics_train_y = metrics_train_frame[
        config.SUBMISSION_TARGET_COLUMN
    ].map(config.SUBMISSION_TARGET_ENCODING)

    metrics_train_valid = metrics_train_y.notna()

    metrics_train_frame = metrics_train_frame.loc[
        metrics_train_valid
    ].copy()

    metrics_train_y = metrics_train_y.loc[
        metrics_train_valid
    ].astype(int)

    metrics_train_X = build_feature_frame(
        metrics_train_frame,
        numeric,
        categorical
    )[schema]

    train_probabilities = amendment_model.predict_proba(
        metrics_train_X
    )[:, 1]

    train_roc_auc = roc_auc_score(
        metrics_train_y,
        train_probabilities
    )

    # Evaluate on the held-out evaluation dataset.
    eval_frame = pd.read_csv(
        config.CLEAN_SUBMISSIONS_EVAL_FILE
    )

    eval_y = eval_frame[
        config.SUBMISSION_TARGET_COLUMN
    ].map(config.SUBMISSION_TARGET_ENCODING)

    eval_valid = eval_y.notna()

    eval_frame = eval_frame.loc[eval_valid].copy()
    eval_y = eval_y.loc[eval_valid].astype(int)

    eval_X = build_feature_frame(
        eval_frame,
        numeric,
        categorical
    )[schema]

    eval_probabilities = amendment_model.predict_proba(
        eval_X
    )[:, 1]

    held_out_roc_auc = roc_auc_score(
        eval_y,
        eval_probabilities
    )

    metrics["amendment"] = {
        "training_rows": len(y),
        "features": schema,
        "train_roc_auc": float(train_roc_auc),
        "held_out_roc_auc": float(held_out_roc_auc),
        "generalisation_gap": float(
            train_roc_auc - held_out_roc_auc
        )
    }

    # ---------------------------------------------------------
    # 2. Exposure regression
    # ---------------------------------------------------------

    a = pd.read_csv(config.CLEAN_AUDIENCES_TRAIN_FILE)

    ya = pd.to_numeric(
        a[config.AUDIENCE_TARGET_COLUMN],
        errors="coerce"
    )

    mask = ya.notna()

    a = a.loc[mask].copy()
    ya = ya.loc[mask]

    numeric = config.EXPOSURE_NUMERIC_FEATURES
    categorical = config.EXPOSURE_CATEGORICAL_FEATURES
    schema = numeric + categorical

    Xe = build_feature_frame(
        a,
        numeric,
        categorical
    )[schema]

    exposure_model = create_exposure_model(
        numeric,
        categorical
    )

    exposure_model.fit(Xe, ya)

    save_model(
        exposure_model,
        config.EXPOSURE_MODEL_FILE
    )

    save_feature_schema(
        schema,
        config.EXPOSURE_SCHEMA_FILE
    )

    metrics["exposure"] = {
        "training_rows": len(ya),
        "features": schema
    }

    # ---------------------------------------------------------
    # 3. Issue classification
    # ---------------------------------------------------------

    n = pd.read_csv(config.CLEAN_CLEARANCE_NOTES_FILE)

    n = n[
        n[config.NOTE_TARGET_COLUMN].notna()
    ].copy()

    base_issue_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                max_features=90000
            )
        ),
        (
            "clf",
            LinearSVC(class_weight="balanced")
        )
    ])

    issue_model = CalibratedClassifierCV(
        estimator=base_issue_model,
        method="sigmoid",
        cv=3
    )

    issue_model.fit(
        n[config.NOTE_TEXT_COLUMN].fillna("").astype(str),
        n[config.NOTE_TARGET_COLUMN].astype(str)
    )

    save_model(
        issue_model,
        config.ISSUE_MODEL_FILE
    )

    metrics["issue"] = {
        "training_rows": len(n),
        "labels": list(issue_model.classes_)
    }

    # ---------------------------------------------------------
    # 4. Save metrics
    # ---------------------------------------------------------

    with open(
        config.MODEL_METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    print("Model training completed successfully.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
