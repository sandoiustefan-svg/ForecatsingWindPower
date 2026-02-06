"""
run_pipeline.py

Main entry point for the full wind power forecasting preprocessing pipeline.

Pipeline steps:
  1) Align forecast + nowcast + metadata + power production into one dataset
  2) Apply feature engineering + scaling + window building
  3) Save ready-to-train tensors + preprocessing artifacts

Outputs:
  - windows_h61.npz         : train/val/test window tensors
  - feature_scaler.joblib   : fitted StandardScaler (train-only)
  - meta_{train,val,test}   : mapping from window index → (region, time_ref)
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .region_dataset_builder import Forecast_Nowcast_Allign
from .preprocessor import Processor


# Input feature columns used by the model (after feature engineering)
FEATURE_COLS: list[str] = [
    "ws10m_mean",
    "ws10m_std",
    "rh2m_mean",
    "rh2m_std",
    "t2m_mean",
    "t2m_std",
    "g10m_mean",
    "g10m_std",
    "mslp_mean",
    "mslp_std",
    "air_temperature_2m",
    "air_pressure_at_sea_level",
    "relative_humidity_2m",
    "wind_speed_10m",
    "capacity_total",
    "precipitation_amount",
    "ws2",
    "ws3",
    "turbulence",
    "gust_factor",
    "air_density_proxy",
    "ws_x_density",
    "wd_sin",
    "wd_cos",
    "wd_mean_sin",
    "wd_mean_cos",
    "tref_hour_sin",
    "tref_hour_cos",
    "tref_doy_sin",
    "tref_doy_cos",
    "lt_scaled",
    "lt_sin",
    "lt_cos",
]


def main():
    """
    Run the full dataset preparation pipeline.

    This produces model-ready horizon windows of shape:
      X: (num_windows, 61, num_features)
      y: (num_windows, 61)

    And saves preprocessing artifacts for reproducibility.
    """

    aligner = Forecast_Nowcast_Allign(
        path_forecast="src/raw_data/met_forecast.parquet",
        path_nowcast="src/raw_data/met_nowcast.parquet",
        path_windpark_bindzone="src/raw_data/windparks_bidzone.csv",
        path_power_per_windpark="src/raw_data/wind_power_per_bidzone.parquet",
    )
    aligner.merge_datasets()

    aligned_path = "src/processed_data/pipeline_data/alligned_data.parquet"

    df = pd.read_parquet(aligned_path)

    # Standardize target column name expected by Processor
    if "powerMW" in df.columns and "power_MW" not in df.columns:
        df = df.rename(columns={"powerMW": "power_MW"})

    print(f"Using {len(FEATURE_COLS)} feature columns:")
    print(FEATURE_COLS)

    processor = Processor(
        dataset=df,
        feature_cols=FEATURE_COLS,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
    )

    out = processor.run(horizon=61, dropna_features=False)

    X_train, y_train, meta_train = out["windows"]["train"]
    X_val, y_val, meta_val = out["windows"]["val"]
    X_test, y_test, meta_test = out["windows"]["test"]

    print("\nFinal tensors:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
    print("X_test: ", X_test.shape, "y_test: ", y_test.shape)

    # Save train/val/test tensors for direct model training
    np.savez_compressed(
        "src/processed_data/pipeline_data/windows_h61.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    # Save fitted scaler (needed for consistent inference later)
    joblib.dump(out["scaler"], "src/processed_data/pipeline_data/feature_scaler.joblib")

    # Save metadata for traceability of each forecast run window
    pd.DataFrame(meta_train, columns=["region", "time_ref"]).to_parquet(
        "src/processed_data/pipeline_data/meta_train.parquet", index=False
    )
    pd.DataFrame(meta_val, columns=["region", "time_ref"]).to_parquet(
        "src/processed_data/pipeline_data/meta_val.parquet", index=False
    )
    pd.DataFrame(meta_test, columns=["region", "time_ref"]).to_parquet(
        "src/processed_data/pipeline_data/meta_test.parquet", index=False
    )

    print("\nSaved artifacts:")
    print("- windows_h61.npz")
    print("- feature_scaler.joblib")
    print("- meta_{train,val,test}.parquet")


if __name__ == "__main__":
    main()
