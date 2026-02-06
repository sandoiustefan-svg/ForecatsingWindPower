import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class Processor:
    """
    Preprocessing pipeline for the aligned wind forecasting dataset.

    Steps:
      - Split by time_ref (train/val/test)
      - Feature engineering + target normalization
      - Scale features (fit on train only)
      - Build horizon windows per (region, time_ref)
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        feature_cols: list[str],
        scaler=None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
    ):
        """Initialize processor with dataset, feature columns, and split fractions."""
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.dataset = dataset
        self.feature_cols = feature_cols
        self.scaler = scaler if scaler is not None else StandardScaler()

    def split_by_time_ref(self):
        """
        Split dataset chronologically by time_ref.
        Prevents leakage by keeping future issue-times out of training.
        """
        assert abs(self.train_frac + self.val_frac + self.test_frac - 1.0) < 1e-6

        self.dataset = self.dataset.sort_values("time_ref").reset_index(drop=True)
        n = len(self.dataset)

        train_end = int(self.train_frac * n)
        val_end = int((self.train_frac + self.val_frac) * n)

        train_df = self.dataset.iloc[:train_end].copy()
        val_df = self.dataset.iloc[train_end:val_end].copy()
        test_df = self.dataset.iloc[val_end:].copy()

        return train_df, val_df, test_df

    def add_wind_direction_sincos(self, col="wind_direction_10m", prefix="wd"):
        """Encode wind direction as sin/cos to avoid circular discontinuity."""
        self.dataset = self.dataset.copy()
        rad = np.deg2rad(self.dataset[col].astype(float))
        self.dataset[f"{prefix}_sin"] = np.sin(rad)
        self.dataset[f"{prefix}_cos"] = np.cos(rad)
        return self.dataset

    def add_time_sincos(self, time_col="time_ref", prefix="tref"):
        """Cyclical encoding of hour-of-day and day-of-year."""
        self.dataset = self.dataset.copy()
        t = pd.to_datetime(self.dataset[time_col])

        hour = t.dt.hour + t.dt.minute / 60.0
        doy = t.dt.dayofyear.astype(float)

        self.dataset[f"{prefix}_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        self.dataset[f"{prefix}_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        self.dataset[f"{prefix}_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        self.dataset[f"{prefix}_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        return self.dataset

    def add_windspeed_polynomials(self, ws_col="ws10m_mean"):
        """Add ws^2 and ws^3 for nonlinear wind-power effects."""
        self.dataset = self.dataset.copy()
        ws = self.dataset[ws_col].astype(float)
        self.dataset["ws2"] = ws**2
        self.dataset["ws3"] = ws**3
        return self.dataset

    def add_gust_turbulence(self, ws_col="ws10m_mean", ws_std_col="ws10m_std", gust_col="g10m_mean"):
        """Add turbulence and gust factor features."""
        self.dataset = self.dataset.copy()
        ws = self.dataset[ws_col].astype(float)
        ws_std = self.dataset[ws_std_col].astype(float)
        gust = self.dataset[gust_col].astype(float)

        eps = 1e-6
        self.dataset["turbulence"] = ws_std / (ws + eps)
        self.dataset["gust_factor"] = gust / (ws + eps)
        return self.dataset

    def add_air_density_proxy(self, p_col="air_pressure_at_sea_level", t_col="air_temperature_2m"):
        """Approximate air density influence on wind energy."""
        self.dataset = self.dataset.copy()
        p = self.dataset[p_col].astype(float)
        t = self.dataset[t_col].astype(float)

        self.dataset["air_density_proxy"] = p / (t + 1e-6)
        self.dataset["ws_x_density"] = self.dataset["ws10m_mean"].astype(float) * self.dataset["air_density_proxy"]
        return self.dataset

    def add_lead_time_features(self, lt_col="lt", horizon=61):
        """Encode lead-time position within the forecast horizon."""
        self.dataset = self.dataset.copy()
        lt = self.dataset[lt_col].astype(float)

        self.dataset["lt_scaled"] = lt / float(horizon)
        self.dataset["lt_sin"] = np.sin(2 * np.pi * lt / float(horizon))
        self.dataset["lt_cos"] = np.cos(2 * np.pi * lt / float(horizon))
        return self.dataset

    def add_target_normalized_power(self, power_col="power_MW", cap_col="capacity_total", out_col="power_norm"):
        """Normalize power output by installed capacity."""
        self.dataset = self.dataset.copy()
        self.dataset[out_col] = self.dataset[power_col].astype(float) / (self.dataset[cap_col].astype(float) + 1e-6)
        return self.dataset

    def make_horizon_windows(self, df, feature_cols, target_col="power_norm", horizon=61):
        """
        Build one window per (region, time_ref), using lt=0..horizon-1.
        """
        df = df.sort_values(["region", "time_ref", "lt"])

        X_list, y_list, meta = [], [], []

        for (region, tref), g in df.groupby(["region", "time_ref"], sort=False):
            g = g.sort_values("lt").iloc[:horizon]

            if len(g) < horizon:
                continue
            if g[target_col].isna().any():
                continue

            X_list.append(g[feature_cols].to_numpy(dtype=np.float32))
            y_list.append(g[target_col].to_numpy(dtype=np.float32))
            meta.append((region, tref))

        X = np.stack(X_list) if X_list else np.empty((0, horizon, len(feature_cols)), dtype=np.float32)
        y = np.stack(y_list) if y_list else np.empty((0, horizon), dtype=np.float32)

        return X, y, meta

    def run(self, horizon=61):
        """
        Full preprocessing run:
          split → FE → scale → window creation.
        Returns ready-to-train tensors.
        """
        train_df, val_df, test_df = self.split_by_time_ref()

        # Feature engineering per split
        for df in [train_df, val_df, test_df]:
            self.dataset = df
            self.add_wind_direction_sincos()
            self.add_time_sincos()
            self.add_windspeed_polynomials()
            self.add_gust_turbulence()
            self.add_air_density_proxy()
            self.add_lead_time_features(horizon=horizon)
            self.add_target_normalized_power()

        # Fit scaler on train only
        self.scaler.fit(train_df[self.feature_cols])
        train_df[self.feature_cols] = self.scaler.transform(train_df[self.feature_cols])
        val_df[self.feature_cols] = self.scaler.transform(val_df[self.feature_cols])
        test_df[self.feature_cols] = self.scaler.transform(test_df[self.feature_cols])

        # Windows
        train_pack = self.make_horizon_windows(train_df, self.feature_cols, horizon=horizon)
        val_pack = self.make_horizon_windows(val_df, self.feature_cols, horizon=horizon)
        test_pack = self.make_horizon_windows(test_df, self.feature_cols, horizon=horizon)

        return train_pack, val_pack, test_pack
