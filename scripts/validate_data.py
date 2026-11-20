"""Validate the supplied AURA dataset, split files, and data dictionary."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FULL = RAW / "AURA_Synthetic_Customer_Retention_Dataset.csv"
TRAIN = RAW / "AURA_Synthetic_Train.csv"
TEST = RAW / "AURA_Synthetic_Test.csv"
DICTIONARY = RAW / "AURA_Data_Dictionary.csv"


def validate() -> dict:
    frames = {"full": pd.read_csv(FULL), "train": pd.read_csv(TRAIN), "test": pd.read_csv(TEST)}
    dictionary = pd.read_csv(DICTIONARY)
    expected = dictionary.parameter.tolist()
    assert len(expected) == len(set(expected)), "Duplicate dictionary parameters"
    assert all(list(df.columns) == expected for df in frames.values()), "Schema/order differs from dictionary"
    inputs = dictionary.loc[dictionary.role.eq("Input feature"), "parameter"].tolist()
    assert len(inputs) == 34 and "customer_id" not in inputs and "churn" not in inputs
    assert dictionary.loc[dictionary.role.eq("Identifier"), "parameter"].tolist() == ["customer_id"]
    assert dictionary.loc[dictionary.role.eq("Target"), "parameter"].tolist() == ["churn"]
    report = {"schema_columns": len(expected), "input_features": len(inputs), "files": {}}
    numeric_columns = dictionary.loc[dictionary.type.isin(["int64", "float64"]), "parameter"].tolist()
    categorical_columns = dictionary.loc[dictionary.type.eq("object"), "parameter"].tolist()
    range_checks = {"age": (0, 120), "tenure_months": (0, None), "monthly_charges": (0, None),
                    "total_charges": (0, None), "days_to_renewal": (0, None), "login_frequency_30d": (0, None),
                    "feature_usage_rate": (0, 100), "last_activity_days": (0, None), "active_days_30d": (0, 30),
                    "satisfaction_score": (1, 5), "nps_score": (-100, 100), "churn": (0, 1)}
    for name, df in frames.items():
        assert df.isna().sum().sum() == 0, f"Missing values in {name}"
        assert not df.duplicated().any(), f"Duplicate rows in {name}"
        assert set(df.churn.unique()) <= {0, 1}, f"Invalid churn labels in {name}"
        assert df.customer_id.is_unique, f"Duplicate IDs in {name}"
        for column in numeric_columns:
            assert pd.api.types.is_numeric_dtype(df[column]), f"{column} is not numeric in {name}"
            assert df[column].map(lambda value: pd.notna(value) and abs(float(value)) != float("inf")).all(), f"Invalid numeric value in {column}"
        for column in categorical_columns:
            assert df[column].dtype.name in ("object", "str", "string"), f"{column} is not categorical text in {name}"
        for column, (low, high) in range_checks.items():
            if low is not None: assert df[column].ge(low).all(), f"{column} below valid range in {name}"
            if high is not None: assert df[column].le(high).all(), f"{column} above valid range in {name}"
        report["files"][name] = {"rows": len(df), "churn": int(df.churn.sum()), "non_churn": int((df.churn == 0).sum())}
    assert set(frames["train"].customer_id).isdisjoint(frames["test"].customer_id), "Train/test ID overlap"
    combined = pd.concat([frames["train"], frames["test"]], ignore_index=True)
    assert len(combined) == len(frames["full"]) and combined.customer_id.nunique() == len(frames["full"]), "Split does not reconcile"
    full_counts = frames["full"].churn.value_counts().to_dict()
    split_counts = combined.churn.value_counts().to_dict()
    assert full_counts == split_counts, "Full and split target counts differ"
    full_sorted = frames["full"].sort_values("customer_id").reset_index(drop=True)
    combined_sorted = combined.sort_values("customer_id").reset_index(drop=True)
    assert full_sorted.equals(combined_sorted), "Full dataset rows differ from train/test split union"
    for column in categorical_columns:
        assert set(frames["train"][column]).issubset(set(frames["full"][column])), f"Unexpected train category in {column}"
        assert set(frames["test"][column]).issubset(set(frames["full"][column])), f"Unexpected test category in {column}"
    report["train_test_id_overlap"] = 0
    report["full_reconciles_with_splits"] = True
    report["schema_types_valid"] = True
    report["numeric_ranges_valid"] = True
    report["categories_within_full_dataset"] = True
    out = ROOT / "data" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
