"""Train, select, evaluate, and persist the three required churn classifiers."""
from __future__ import annotations

import json
import os
import shutil
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from PIL import Image, ImageDraw
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss
from ml.evaluation.thresholds import ranking_metrics, select_operational_threshold, threshold_metrics

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "ml" / "reports"
ARTIFACTS = ROOT / "ml" / "artifacts"
RANDOM_STATE = 42
THRESHOLD = 0.5


def _canvas(title, width=900, height=600):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), title, fill="#172554")
    return image, draw


def _line_plot(path, title, curves, xlabel, ylabel):
    image, draw = _canvas(title)
    left, top, right, bottom = 75, 70, 840, 520
    draw.line((left, bottom, right, bottom), fill="#334155", width=2)
    draw.line((left, top, left, bottom), fill="#334155", width=2)
    palette = ["#0891b2", "#7c3aed", "#ea580c"]
    for idx, (label, xs, ys) in enumerate(curves):
        pts = [(left + float(x) * (right-left), bottom - float(y) * (bottom-top)) for x, y in zip(xs, ys)]
        draw.line(pts, fill=palette[idx % len(palette)], width=3)
        draw.text((left + 10, top + idx*24), label, fill=palette[idx % len(palette)])
    draw.text((left, bottom+18), xlabel, fill="#334155")
    draw.text((5, top), ylabel, fill="#334155")
    image.save(path)


def _confusion_plot(path, title, matrix):
    image, draw = _canvas(title, 540, 450)
    colors = ["#dbeafe", "#0e7490", "#0e7490", "#dbeafe"]
    for row in range(2):
        for col in range(2):
            x, y = 160 + col*150, 110 + row*120
            draw.rectangle((x, y, x+140, y+110), fill=colors[row*2+col], outline="#334155", width=2)
            draw.text((x+52, y+42), str(int(matrix[row][col])), fill="#0f172a")
    draw.text((170, 365), "Predicted: No churn / Churn", fill="#334155")
    draw.text((10, 205), "Actual", fill="#334155")
    image.save(path)


def _schema():
    dd = pd.read_csv(RAW / "AURA_Data_Dictionary.csv")
    return (dd.loc[dd.role.eq("Input feature"), "parameter"].tolist(),
            dd.loc[dd.role.eq("Input feature") & dd.type.eq("object"), "parameter"].tolist(),
            dd.loc[dd.role.eq("Input feature") & dd.type.ne("object"), "parameter"].tolist())


def _pipeline(model, categorical, numeric):
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", prep), ("classifier", model)])


def train():
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    # Preserve the previously generated comparison as an immutable baseline report.
    for filename in ("model_comparison.csv", "model_comparison.json"):
        source = REPORTS / filename
        baseline = REPORTS / filename.replace("model_comparison", "model_comparison_initial")
        if source.exists() and not baseline.exists(): shutil.copy2(source, baseline)
    features, cat, num = _schema()
    train_df = pd.read_csv(RAW / "AURA_Synthetic_Train.csv")
    test_df = pd.read_csv(RAW / "AURA_Synthetic_Test.csv")
    X_train, y_train = train_df[features], train_df.churn.astype(int)
    X_test, y_test = test_df[features], test_df.churn.astype(int)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    specs = {
        "logistic_regression": (LogisticRegression(max_iter=1200, class_weight="balanced", random_state=RANDOM_STATE),
                                {"classifier__C": [0.1, 1.0, 10.0]}),
        "random_forest": (RandomForestClassifier(n_estimators=160, class_weight="balanced_subsample", n_jobs=1, random_state=RANDOM_STATE),
                          {"classifier__max_depth": [None, 14], "classifier__min_samples_leaf": [1, 4]}),
        "xgboost": (XGBClassifier(n_estimators=240, max_depth=4, learning_rate=0.06, subsample=0.9,
                                  colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
                                  n_jobs=1, random_state=RANDOM_STATE),
                    {"classifier__max_depth": [3, 5], "classifier__min_child_weight": [1, 4]}),
    }
    fitted, validations = {}, {}
    # Model selection sees only cross-validation folds from the supplied training split.
    for name, (estimator, grid) in specs.items():
        print(f"Training {name} with training-only cross-validation", flush=True)
        search = GridSearchCV(_pipeline(estimator, cat, num), grid, scoring="roc_auc", cv=cv,
                              n_jobs=1, refit=True, error_score="raise")
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        print(f"Finished {name}", flush=True)
        validations[name] = {"cv_roc_auc": float(search.best_score_), "best_params": search.best_params_}
    # Held-out test is touched only after every candidate and hyperparameter has been selected.
    results, probs = [], {}
    for name, model in fitted.items():
        print(f"Evaluating {name} on held-out test", flush=True)
        probability = model.predict_proba(X_test)[:, 1]
        print(f"Predicted {name}", flush=True)
        prediction = (probability >= THRESHOLD).astype(int)
        probs[name] = probability
        metrics = {
            "model": name, "accuracy": accuracy_score(y_test, prediction),
            "precision": precision_score(y_test, prediction, zero_division=0),
            "recall": recall_score(y_test, prediction, zero_division=0),
            "f1": f1_score(y_test, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probability),
            "pr_auc": average_precision_score(y_test, probability),
            "threshold": THRESHOLD, "validation_roc_auc": validations[name]["cv_roc_auc"],
        }
        results.append(metrics)
        (REPORTS / f"classification_report_{name}.json").write_text(json.dumps({
            "metrics": metrics, "classification_report": classification_report(y_test, prediction, output_dict=True),
            "confusion_matrix": [[int((y_test.eq(0) & (prediction == 0)).sum()), int((y_test.eq(0) & (prediction == 1)).sum())],
                                 [int((y_test.eq(1) & (prediction == 0)).sum()), int((y_test.eq(1) & (prediction == 1)).sum())]],
        }, indent=2))
        print(f"Wrote report {name}", flush=True)
        matrix = confusion_matrix(y_test, prediction)
        _confusion_plot(REPORTS / f"confusion_matrix_{name}.png", name.replace("_", " ").title(), matrix)
    comparison = pd.DataFrame(results).sort_values(["recall", "pr_auc", "f1"], ascending=False)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    (REPORTS / "model_comparison.json").write_text(comparison.to_json(orient="records", indent=2))
    from sklearn.metrics import precision_recall_curve, roc_curve
    _line_plot(REPORTS / "roc_curve_comparison.png", "ROC curve comparison",
               [(name, *roc_curve(y_test, p)[:2]) for name, p in probs.items()], "False positive rate", "True positive rate")
    _line_plot(REPORTS / "precision_recall_comparison.png", "Precision-recall comparison",
               [(name, *precision_recall_curve(y_test, p)[1::-1]) for name, p in probs.items()], "Recall", "Precision")
    # Select using training-only cross-validation. The held-out test split does not choose the model.
    selected_name = max(validations, key=lambda candidate: validations[candidate]["cv_roc_auc"])
    selected = fitted[selected_name]
    # Threshold selection uses out-of-fold predictions on training rows only.
    selected_oof = cross_val_predict(selected, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    capacity_rate = float(os.getenv("AURA_CONTACT_CAPACITY_RATE", "0.20"))
    decision_threshold, threshold_table, threshold_rationale = select_operational_threshold(
        y_train, selected_oof, capacity_rate=capacity_rate)
    threshold_table.to_csv(REPORTS / "threshold_analysis_oof.csv", index=False)
    # Apply this training-selected threshold once to the held-out test split.
    test_threshold_metrics = threshold_metrics(y_test, probs[selected_name], decision_threshold)
    test_threshold_metrics["brier_score"] = float(brier_score_loss(y_test, probs[selected_name]))
    test_threshold_metrics["ranking_metrics"] = ranking_metrics(y_test, probs[selected_name], (0.10, 0.20, 0.30))
    threshold_report = {"selected_model": selected_name, "selection_data": "training-only out-of-fold predictions",
                        "capacity_rate": capacity_rate, "decision_threshold": decision_threshold,
                        "selection_rationale": threshold_rationale,
                        "oof_selected_threshold_metrics": threshold_metrics(y_train, selected_oof, decision_threshold),
                        "held_out_test_metrics": test_threshold_metrics,
                        "probability_calibration": {"brier_score_test": test_threshold_metrics["brier_score"],
                            "calibrated": False, "note": "Brier score reported as a calibration diagnostic; no calibration transform was fitted."}}
    (REPORTS / "threshold_evaluation.json").write_text(json.dumps(threshold_report, indent=2, default=float))
    threshold_curves = [(label, threshold_table.threshold, threshold_table[column]) for label, column in
                        [("F1", "f1"), ("Precision", "precision"), ("Recall", "recall"), ("Specificity", "specificity")]]
    _line_plot(REPORTS / "threshold_performance.png", "Training OOF threshold performance", threshold_curves,
               "Decision threshold", "Metric value")
    joblib.dump(selected, ARTIFACTS / "best_model.pkl")
    # Validated model-specific coefficient/impurity importance on fitted model.
    prep = selected.named_steps["preprocess"]
    names = prep.get_feature_names_out()
    clf = selected.named_steps["classifier"]
    importance = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else clf.feature_importances_
    fi = pd.DataFrame({"feature": names, "importance": importance}).sort_values("importance", ascending=False).head(20)
    fi.to_csv(REPORTS / "feature_importance.csv", index=False)
    image, draw = _canvas(f"Top model-specific feature importance — {selected_name}", 1100, 700)
    max_v = max(float(fi.importance.max()), 1e-12)
    for i, row in enumerate(fi.sort_values("importance", ascending=False).itertuples()):
        y = 70 + i*29
        draw.text((15, y), str(row.feature)[:48], fill="#334155")
        draw.rectangle((390, y+2, 390 + int(650 * row.importance/max_v), y+20), fill="#0891b2")
    image.save(REPORTS / "feature_importance.png")
    selected_metrics = next(row for row in results if row["model"] == selected_name)
    metadata = {"model_version": "churn-v1", "selected_model": selected_name,
                "selection_rule": "highest stratified training-only cross-validation ROC-AUC",
                "threshold": decision_threshold, "threshold_selection": {"capacity_rate": capacity_rate,
                    "method": "highest F1 among OOF thresholds within capacity; ties prefer precision then higher threshold",
                    "rationale": threshold_rationale},
                "selected_metrics": selected_metrics, "threshold_test_metrics": test_threshold_metrics, "validation": validations,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(), "features": features,
                "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__, "joblib": joblib.__version__}
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=float))
    (ARTIFACTS / "feature_schema.json").write_text(json.dumps({"features": features, "categorical": cat, "numeric": num}, indent=2))
    (REPORTS / "training_summary.md").write_text(
        f"# Model training summary\n\nSelected **{selected_name}** by highest stratified training-only cross-validation ROC-AUC. Candidate comparison metrics use the fixed {THRESHOLD:.2f} threshold. "
        "All candidates were selected with stratified cross-validation on the 9,600-row training split; the 2,400-row supplied test split was used once for final reporting. "
        "Recall-first ranking supports retention outreach but may create more false positives; review precision before using actions. "
        f"The operational decision threshold is {decision_threshold:.2f}; selected using training-only out-of-fold predictions with a {capacity_rate:.0%} stated contact-capacity limit. "
        "Held-out results at that threshold are in threshold_evaluation.json. Feature importance describes fitted-model associations, not causes. SHAP is not generated; the selected estimator's coefficient magnitude or impurity importance is reported as an alternative.\n\n"
        + "```\n" + comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}") + "\n```\n")
    # Fresh load and an actual held-out customer inference smoke check.
    reloaded = joblib.load(ARTIFACTS / "best_model.pkl")
    p = float(reloaded.predict_proba(X_test.iloc[[0]])[0, 1])
    assert np.isfinite(p) and 0 <= p <= 1
    from ml.inference.predict import ChurnPredictor
    service = ChurnPredictor(ARTIFACTS / "best_model.pkl")
    verified = service.predict(str(test_df.iloc[0].customer_id), X_test.iloc[0].to_dict())
    assert verified["model_version"] == metadata["model_version"]
    assert 0 <= verified["churn_probability"] <= 1
    invalid_checks = {}
    for label, bad in [("missing_feature", {k: v for k, v in X_test.iloc[0].to_dict().items() if k != features[0]}),
                       ("extra_feature", {**X_test.iloc[0].to_dict(), "customer_id": "must-not-be-used"})]:
        try: service.predict(str(test_df.iloc[0].customer_id), bad)
        except ValueError: invalid_checks[label] = "rejected"
        else: raise AssertionError(f"Malformed payload was accepted: {label}")
    (REPORTS / "inference_verification.json").write_text(json.dumps({
        "status": "passed", "customer_id": verified["customer_id"], "probability": verified["churn_probability"],
        "risk_level": verified["risk_level"], "model_version": verified["model_version"],
        "model_reloaded_from_saved_artifact": True, "invalid_input_checks": invalid_checks
    }, indent=2))
    return {"selected_model": selected_name, "sample_customer_id": str(test_df.iloc[0].customer_id), "sample_probability": p,
            "metrics": comparison.to_dict(orient="records")}


if __name__ == "__main__":
    print(json.dumps(train(), indent=2, default=float))
