# tests/test_preprocessor.py
import numpy as np
import pandas as pd
import pytest

from src.features.preprocessor import Processor


HORIZON = 61


def _make_synth_df(
    n_time_ref: int = 10,
    regions: tuple[str, ...] = ("NO1", "NO2"),
    horizon: int = HORIZON,
    start: str = "2020-01-01 00:00:00",
) -> pd.DataFrame:
    """
    Create a synthetic aligned dataset shaped like yours:
    one row per (region, time_ref, lt) with required columns.
    """
    rng = np.random.default_rng(0)

    rows = []
    start_ts = pd.Timestamp(start)

    for r_i, region in enumerate(regions):
        for k in range(n_time_ref):
            tref = start_ts + pd.Timedelta(hours=k)  # issue time
            for lt in range(horizon):
                time = tref + pd.Timedelta(hours=lt)

                ws = 5 + 0.1 * lt + 0.3 * r_i + 0.05 * k + rng.normal(0, 0.05)
                wd = (180 + 2 * lt + 10 * r_i) % 360

                rows.append(
                    {
                        "region": region,
                        "time_ref": tref,
                        "time": time,
                        "lt": lt,
                        # core met/nowcast-style columns used by Processor feature engineering
                        "ws10m_mean": ws,
                        "ws10m_std": abs(rng.normal(0.5, 0.05)),
                        "g10m_mean": ws + abs(rng.normal(0.3, 0.05)),
                        "wind_direction_10m": wd,
                        "air_pressure_at_sea_level": 101325 + 10 * rng.normal(),
                        "air_temperature_2m": 273.15 + 5 + 0.1 * rng.normal(),
                        # required for target normalization
                        "capacity_total": 1000.0 + 100.0 * r_i,
                        "power_MW": 400.0 + 20.0 * r_i + 0.5 * lt + 0.2 * rng.normal(),
                    }
                )

    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def feature_cols() -> list[str]:
    # columns produced/used by Processor after feature engineering
    return [
        # original numeric cols used as features
        "ws10m_mean",
        "ws10m_std",
        "g10m_mean",
        "air_pressure_at_sea_level",
        "air_temperature_2m",
        "capacity_total",
        # engineered cols
        "ws2",
        "ws3",
        "turbulence",
        "gust_factor",
        "air_density_proxy",
        "ws_x_density",
        "wd_sin",
        "wd_cos",
        "tref_hour_sin",
        "tref_hour_cos",
        "tref_doy_sin",
        "tref_doy_cos",
        "lt_scaled",
        "lt_sin",
        "lt_cos",
    ]


def test_split_by_time_ref_is_chronological(feature_cols):
    df = _make_synth_df(n_time_ref=10)
    proc = Processor(df, feature_cols=feature_cols, train_frac=0.7, val_frac=0.15, test_frac=0.15)

    train_df, val_df, test_df = proc.split_by_time_ref()

    # time_ref ranges should be ordered and non-overlapping
    assert train_df["time_ref"].max() <= val_df["time_ref"].min()
    assert val_df["time_ref"].max() <= test_df["time_ref"].min()

    # sizes should sum to total
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_make_horizon_windows_shapes(feature_cols):
    df = _make_synth_df(n_time_ref=6, regions=("NO1", "NO2"), horizon=HORIZON)
    proc = Processor(df, feature_cols=feature_cols)

    # run FE on whole df so feature cols exist
    proc.dataset = df.copy()
    proc.add_wind_direction_sincos()
    proc.add_time_sincos()
    proc.add_windspeed_polynomials()
    proc.add_gust_turbulence()
    proc.add_air_density_proxy()
    proc.add_lead_time_features(horizon=HORIZON)
    proc.add_target_normalized_power()

    X, y, meta = proc.make_horizon_windows(
        proc.dataset, feature_cols=feature_cols, target_col="power_norm", horizon=HORIZON
    )

    # number of windows = regions * time_ref count
    expected_windows = 2 * 6
    assert len(meta) == expected_windows
    assert X.shape == (expected_windows, HORIZON, len(feature_cols))
    assert y.shape == (expected_windows, HORIZON)


def test_make_horizon_windows_skips_incomplete_group(feature_cols):
    df = _make_synth_df(n_time_ref=4, regions=("NO1",), horizon=HORIZON)

    # Remove one lt from one time_ref to make it incomplete
    bad_tref = df["time_ref"].unique()[0]
    df = df[~((df["time_ref"] == bad_tref) & (df["lt"] == HORIZON - 1))].copy()

    proc = Processor(df, feature_cols=feature_cols)

    # FE
    proc.dataset = df.copy()
    proc.add_wind_direction_sincos()
    proc.add_time_sincos()
    proc.add_windspeed_polynomials()
    proc.add_gust_turbulence()
    proc.add_air_density_proxy()
    proc.add_lead_time_features(horizon=HORIZON)
    proc.add_target_normalized_power()

    X, y, meta = proc.make_horizon_windows(proc.dataset, feature_cols=feature_cols, horizon=HORIZON)

    # 4 time_ref total, but 1 is incomplete => 3 windows
    assert len(meta) == 3
    assert X.shape[0] == 3
    assert y.shape[0] == 3


def test_run_outputs_consistent_tensors(feature_cols):
    df = _make_synth_df(n_time_ref=10, regions=("NO1", "NO2"), horizon=HORIZON)
    proc = Processor(df, feature_cols=feature_cols, train_frac=0.7, val_frac=0.15, test_frac=0.15)

    train_pack, val_pack, test_pack = proc.run(horizon=HORIZON)

    X_train, y_train, meta_train = train_pack
    X_val, y_val, meta_val = val_pack
    X_test, y_test, meta_test = test_pack

    # basic shape checks
    assert X_train.ndim == 3 and y_train.ndim == 2
    assert X_val.ndim == 3 and y_val.ndim == 2
    assert X_test.ndim == 3 and y_test.ndim == 2

    assert X_train.shape[1] == HORIZON
    assert X_val.shape[1] == HORIZON
    assert X_test.shape[1] == HORIZON

    assert y_train.shape[1] == HORIZON
    assert y_val.shape[1] == HORIZON
    assert y_test.shape[1] == HORIZON

    # feature dimension matches
    assert X_train.shape[2] == len(feature_cols)
    assert X_val.shape[2] == len(feature_cols)
    assert X_test.shape[2] == len(feature_cols)

    # meta length matches windows
    assert len(meta_train) == X_train.shape[0]
    assert len(meta_val) == X_val.shape[0]
    assert len(meta_test) == X_test.shape[0]


def test_scaler_fits_on_train_only_basic_sanity(feature_cols):
    """
    Not a perfect leakage test, but checks that scaling produces roughly
    zero-mean/unit-std on TRAIN across all timesteps & windows.
    """
    df = _make_synth_df(n_time_ref=12, regions=("NO1", "NO2"), horizon=HORIZON)
    proc = Processor(df, feature_cols=feature_cols, train_frac=0.7, val_frac=0.15, test_frac=0.15)

    train_pack, _, _ = proc.run(horizon=HORIZON)
    X_train, _, _ = train_pack

    flat = X_train.reshape(-1, X_train.shape[-1])

    means = flat.mean(axis=0)
    stds = flat.std(axis=0)

    # allow loose tolerance (synthetic data + finite samples)
    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(stds))
    assert np.all(np.abs(means) < 0.15)
    assert np.all((stds > 0.6) & (stds < 1.4))
