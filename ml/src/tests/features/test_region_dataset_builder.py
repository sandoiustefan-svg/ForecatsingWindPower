from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.region_dataset_builder import Forecast_Nowcast_Allign


def _write_parquet(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return str(path)


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def toy_inputs(tmp_path: Path):
    # ---- Forecast: needs ensemble member columns that later become *_mean/_std, incl wd10m
    forecast = pd.DataFrame(
        {
            "time_ref": [
                pd.Timestamp("2020-01-01 00:00:00"),
                pd.Timestamp("2020-01-01 00:00:00"),
            ],
            "time": [
                pd.Timestamp("2020-01-01 01:00:00"),
                pd.Timestamp("2020-01-01 02:00:00"),
            ],
            "lt": [1, 2],
            "sid": ["A. Vindpark", "A. Vindpark"],
            # ensembles for mean/std creation
            "ws10m_1": [2.0, 4.0],
            "ws10m_2": [4.0, 6.0],
            "rh2m_1": [0.8, 0.7],
            "rh2m_2": [0.9, 0.6],
            "t2m_1": [275.0, 276.0],
            "t2m_2": [277.0, 278.0],
            "g10m_1": [5.0, 6.0],
            "g10m_2": [7.0, 8.0],
            "mslp_1": [100000.0, 100100.0],
            "mslp_2": [100020.0, 100080.0],
            "wd10m_1": [180.0, 190.0],
            "wd10m_2": [200.0, 210.0],
        }
    )

    # ---- Nowcast: must contain columns referenced later in wmean_cols + circular mean + precipitation sum
    # index is time (like your real data), will be reset_index() inside the pipeline
    nowcast = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2020-01-01 00:00:00"),
                pd.Timestamp("2020-01-01 01:00:00"),
            ],
            "windpark": ["A. Vindpark", "A. Vindpark"],
            "air_temperature_2m": [10.0, 999.0],
            "air_pressure_at_sea_level": [1010.0, 8888.0],
            "relative_humidity_2m": [0.5, 0.9],
            "precipitation_amount": [1.0, 2.0],
            "wind_speed_10m": [10.0, 999.0],
            "wind_direction_10m": [90.0, 270.0],
        }
    ).set_index("time")

    # ---- Windpark metadata CSV: must have substation_name and bidding_area etc.
    windparks = pd.DataFrame(
        {
            "bidding_area": ["ELSPOT NO1"],
            "substation_name": ["A"],  # IMPORTANT: cleaned name must match merged sid ("A")
            "operating_power_max": [50.0],
            "prod_start_new": [pd.Timestamp("2019-01-01 00:00:00")],
        }
    )

    # ---- Power per bidding zone parquet: index is time, columns are ELSPOT NOx
    # must match region_df merge on ["region","time"] where time is forecast valid time
    power = pd.DataFrame(
        {
            "ELSPOT NO1": [111.0, 222.0],
            "ELSPOT NO2": [0.0, 0.0],
            "ELSPOT NO3": [0.0, 0.0],
            "ELSPOT NO4": [0.0, 0.0],
        },
        index=[
            pd.Timestamp("2020-01-01 01:00:00"),
            pd.Timestamp("2020-01-01 02:00:00"),
        ],
    )

    f_path = tmp_path / "met_forecast.parquet"
    n_path = tmp_path / "met_nowcast.parquet"
    w_path = tmp_path / "windparks_bidzone.csv"
    p_path = tmp_path / "wind_power_per_bidzone.parquet"

    return (
        _write_parquet(forecast, f_path),
        _write_parquet(nowcast, n_path),
        _write_csv(windparks, w_path),
        _write_parquet(power, p_path),
    )


def test_identify_prefixes_ensemble(toy_inputs):
    f_path, n_path, w_path, p_path = toy_inputs
    obj = Forecast_Nowcast_Allign(
        path_nowcast=n_path,
        path_forecast=f_path,
        path_windpark_bindzone=w_path,
        path_power_per_windpark=p_path,
    )
    prefixes = obj._identify_prefixes_ensemble(obj.forecast)
    assert "ws10m" in prefixes
    assert "wd10m" in prefixes
    assert isinstance(prefixes, set)


def test_clean_names_windparks():
    obj = Forecast_Nowcast_Allign.__new__(Forecast_Nowcast_Allign)
    cleaned = obj._clean_names_windparks("  A. Vindpark  ")
    assert cleaned == "A"
    cleaned2 = obj._clean_names_windparks("B. Vindkraft")
    assert cleaned2 == "B"
    cleaned3 = obj._clean_names_windparks("C Vindkraft")
    assert cleaned3 == "C"


def test_merge_uses_time_ref_not_valid_time(toy_inputs):
    f_path, n_path, w_path, p_path = toy_inputs
    obj = Forecast_Nowcast_Allign(
        path_nowcast=n_path,
        path_forecast=f_path,
        path_windpark_bindzone=w_path,
        path_power_per_windpark=p_path,
    )

    saved = []

    def _capture_save(df, path="ignored"):
        saved.append((path, df.copy()))

    obj.save_merged_dataset = _capture_save

    obj.merge_datasets()

    # First save is the forecast+nowcast merged table
    merged = saved[0][1]

    # Because merge is on (sid, time_ref) -> nowcast.time,
    # the nowcast value should come from 00:00 row (10.0), not the 01:00 row (999.0).
    assert "wind_speed_10m" in merged.columns
    assert merged["wind_speed_10m"].iloc[0] == 10.0
    assert merged["wind_speed_10m"].iloc[1] == 10.0


def test_ensemble_columns_collapsed_mean_std_and_original_dropped(toy_inputs):
    f_path, n_path, w_path, p_path = toy_inputs
    obj = Forecast_Nowcast_Allign(
        path_nowcast=n_path,
        path_forecast=f_path,
        path_windpark_bindzone=w_path,
        path_power_per_windpark=p_path,
    )

    saved = []

    def _capture_save(df, path="ignored"):
        saved.append((path, df.copy()))

    obj.save_merged_dataset = _capture_save
    obj.merge_datasets()

    merged = saved[0][1]

    assert "ws10m_mean" in merged.columns
    assert "ws10m_std" in merged.columns
    assert "ws10m_1" not in merged.columns
    assert "ws10m_2" not in merged.columns

    assert np.isclose(merged["ws10m_mean"].iloc[0], 3.0)
    assert np.isclose(merged["ws10m_std"].iloc[0], np.sqrt(2.0))


def test_sid_column_position_after_time(toy_inputs):
    f_path, n_path, w_path, p_path = toy_inputs
    obj = Forecast_Nowcast_Allign(
        path_nowcast=n_path,
        path_forecast=f_path,
        path_windpark_bindzone=w_path,
        path_power_per_windpark=p_path,
    )

    saved = []

    def _capture_save(df, path="ignored"):
        saved.append((path, df.copy()))

    obj.save_merged_dataset = _capture_save
    obj.merge_datasets()

    merged = saved[0][1]
    cols = merged.columns.tolist()
    time_idx = cols.index("time")
    sid_idx = cols.index("sid")
    assert sid_idx == time_idx + 1


def test_final_dataset_has_region_and_power_merge(toy_inputs):
    f_path, n_path, w_path, p_path = toy_inputs
    obj = Forecast_Nowcast_Allign(
        path_nowcast=n_path,
        path_forecast=f_path,
        path_windpark_bindzone=w_path,
        path_power_per_windpark=p_path,
    )

    saved = []

    def _capture_save(df, path="ignored"):
        saved.append((path, df.copy()))

    obj.save_merged_dataset = _capture_save
    obj.merge_datasets()

    # Second save is the final aligned dataset (after region aggregation + power merge)
    final_df = saved[-1][1]

    assert "region" in final_df.columns
    assert "powerMW" in final_df.columns

    # region should be NO1 after stripping "ELSPOT "
    assert set(final_df["region"].dropna().unique()) == {"NO1"}

    # Ensure power values were merged on (region, time) where time = forecast valid time
    # We expect (NO1, 01:00) -> 111.0 and (NO1, 02:00) -> 222.0 to appear.
    sub = final_df[["region", "time", "powerMW"]].dropna(subset=["powerMW"]).copy()
    sub["time"] = pd.to_datetime(sub["time"])
    lookup = {(r, t): v for r, t, v in sub.to_records(index=False)}

    assert ("NO1", pd.Timestamp("2020-01-01 01:00:00")) in lookup
    assert ("NO1", pd.Timestamp("2020-01-01 02:00:00")) in lookup
    assert lookup[("NO1", pd.Timestamp("2020-01-01 01:00:00"))] == 111.0
    assert lookup[("NO1", pd.Timestamp("2020-01-01 02:00:00"))] == 222.0
