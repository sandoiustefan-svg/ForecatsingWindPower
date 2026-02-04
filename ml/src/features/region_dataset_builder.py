from dataclasses import dataclass

import numpy as np
import pandas as pd
import re

@dataclass
class RegionDatasetConfig:
    """
    Stores column names used across forecast, nowcast, windpark metadata,
    and regional wind power production datasets.
    """
    # MET forecast dataset column names
    MET_forecast: str = "MET_Forecast"
    time_ref_col: str = "time_ref"
    forecast_time_col: str = "time"
    lt_col: str = "lt"
    sid_col: str = "sid"

    # MET nowcast dataset column names
    MET_nowcast: str = "MET_Nowcast"
    windpark_col: str = "windpark"
    nowcast_time_col = "time"

    # windparks_bindzone
    windparks: str = "windparks_bindzone"
    region_col: str = "region"
    bidding_area_col: str = "bidding_area"
    prod_start_new_col: str = "prod_start_new"
    substation_name_col: str = "substation_name"
    operating_power_max_col: str = "operating_power_max"

    # wind_power_per_bindzone
    power_time_col: str = "time"
    power_col: str = "powerMW"


class Forecast_Nowcast_Allign:
    """
    Aligns MET forecast and nowcast data, aggregates windparks into regions,
    and merges wind power production as ground truth.
    """
    def __init__(
            self,
            path_nowcast: str,
            path_forecast: str,
            path_windpark_bindzone: str,
            path_power_per_windpark: str,
            ):
        """
        Loads forecast, nowcast, windpark metadata, and power production datasets.
        """
        
        self.nowcast = pd.read_parquet(path_nowcast)
        self.forecast = pd.read_parquet(path_forecast)
        self.windpark = pd.read_csv(path_windpark_bindzone)
        self.power = pd.read_parquet(path_power_per_windpark)
        self.config = RegionDatasetConfig()

    def _get_shape_dataset_missing_val(self, dataset: pd.DataFrame, name: str) -> str:
        """Print dataset size and missing-value counts."""
        print(f"The {name} has the following: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
        print(f"Missing values: {dataset.isnull().sum()} missing values")

    def _identify_prefixes_ensemble(self, dataset: pd.DataFrame) -> set[str]:
        """
        Finds ensemble prefixes like 'ws10m' from columns 'ws10m_1', 'ws10m_2', etc.
        """
        prefixes = set(
            re.match(r"(.*)_\d+$", c).group(1)
            for c in dataset.columns
            if re.match(r".*_\d+$", c)
        )
        print(f"The prefixes of ensemgle: {prefixes}")
        return prefixes
    
    def _windparks_dataset(self, dataset: pd.DataFrame, windpark_column: str, name: str):
        """Print unique windparks contained in a dataset."""
        windparks_list = dataset[windpark_column].unique().tolist()
        print(f"The {name} dataset has the following windparks: {windparks_list}")

    def _clean_names_windparks(self, name: str):
        """Standardizes windpark names by removing suffixes and whitespace."""
        name = name.strip()
        name = name.replace(". Vindpark", "")
        name = name.replace(". Vindkraft", "")
        name = name.replace(" Vindkraft", "")
        return name
    
    def _min_max_date_dataset(self, column: str, dataset: pd.DataFrame):
        """Filters dataset to its own min/max datetime range."""
        min_date, max_date = dataset[column].min(), dataset[column].max()
        dataset = dataset[
            dataset[column].between(min_date, max_date)
        ].copy()

    def _make_sure_timestep(self):
        """Ensures all time columns are parsed as datetime."""
        self.nowcast[self.config.nowcast_time_col] = pd.to_datetime(self.nowcast[self.config.nowcast_time_col])
        self.forecast[self.config.forecast_time_col] = pd.to_datetime(self.forecast[self.config.forecast_time_col])
        self.forecast[self.config.time_ref_col] = pd.to_datetime(self.forecast[self.config.time_ref_col])

    def _add_windparks_into_regions(self, dataset: pd.DataFrame, wcol: str, keys: list[str], wmean_cols: list[str]):
        """Computes capacity-weighted regional averages of meteorological features."""
        tmp = dataset[keys + [wcol] + wmean_cols].copy()
        w_sum = tmp.groupby(keys)[wcol].sum().rename("capacity_total")

        for c in wmean_cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            tmp[c] = tmp[c] * tmp[wcol]
        
        num_sum = tmp.groupby(keys)[wmean_cols].sum()

        region_df = (num_sum.div(w_sum, axis=0)
                    .reset_index()
                    .merge(w_sum.reset_index(), on=keys, how="left"))
        
        return region_df
    
    def _add_circular_mean(self, region_df, df, colname, outname, wcol, keys):
        """Adds capacity-weighted circular mean for wind direction features."""
        ang = pd.to_numeric(df[colname], errors="coerce")
        w = pd.to_numeric(df[wcol], errors="coerce")
        m = ang.notna() & w.notna()

        sub = df.loc[m, keys].copy()
        rad = np.deg2rad(ang[m].values)

        sub["_sin"] = np.sin(rad) * w[m].values
        sub["_cos"] = np.cos(rad) * w[m].values

        agg = sub.groupby(keys)[["_sin","_cos"]].sum().reset_index()
        agg[outname] = (np.rad2deg(np.arctan2(agg["_sin"], agg["_cos"])) % 360)
        agg = agg.drop(columns=["_sin","_cos"])

        return region_df.merge(agg, on=keys, how="left")

    def save_merged_dataset(self, dataset: pd.DataFrame, path: str = "src/processed_data/pipeline_data/met_forecast_nowcast_merged.parquet"):
        """Saves a dataset to Parquet."""
        dataset.to_parquet(path)

    def merge_datasets(self):
        """
        Full pipeline:
        forecast+nowcast merge → metadata join → regional aggregation → power merge.
        """
        self._get_shape_dataset_missing_val(self.nowcast, name=self.config.MET_nowcast)

        prefixes = self._identify_prefixes_ensemble(self.forecast)

        for prefix in prefixes:
            cols = [column for column in self.forecast.columns if column.startswith(prefix + "_")]

            self.forecast[f"{prefix}_mean"] = self.forecast[cols].mean(axis=1)
            self.forecast[f"{prefix}_std"] = self.forecast[cols].std(axis=1)
        
        ensemble_cols = [column for column in self.forecast.columns if re.match(r".*_\d+$", column)]
        self.forecast = self.forecast.drop(columns=ensemble_cols)

        self._get_shape_dataset_missing_val(self.forecast, name=self.config.MET_forecast)
        self.nowcast[self.config.windpark_col] = self.nowcast[self.config.windpark_col].apply(self._clean_names_windparks)
        self.forecast[self.config.sid_col] = self.forecast[self.config.sid_col].apply(self._clean_names_windparks)

        self._windparks_dataset(self.forecast, self.config.sid_col, self.config.MET_forecast)

        self.nowcast = self.nowcast.reset_index()
        self._make_sure_timestep()
        self._min_max_date_dataset(self.config.time_ref_col, self.forecast)
        self._min_max_date_dataset(self.config.nowcast_time_col, self.nowcast)

        self.nowcast = self.nowcast.rename(columns={self.config.windpark_col: self.config.sid_col})

        merged = self.forecast.merge(
            self.nowcast,
            left_on=[self.config.sid_col, self.config.time_ref_col],
            right_on=[self.config.sid_col, self.config.nowcast_time_col],
            how="left",
            suffixes=("", "_now")
        ).drop(columns=["time_now"])

        cols = merged.columns.tolist()
        cols.remove(self.config.sid_col)

        time_index = cols.index(self.config.forecast_time_col)
        cols.insert(time_index + 1, self.config.sid_col)

        merged = merged[cols]

        self.save_merged_dataset(merged)

        self._windparks_dataset(self.windpark, self.config.substation_name_col, self.config.windparks)

        self.windpark[self.config.region_col] = self.windpark[self.config.bidding_area_col].str.replace("ELSPOT ", "", regex=False).str.strip()
        self.windpark[self.config.prod_start_new_col] = pd.to_datetime(self.windpark[self.config.prod_start_new_col], errors="coerce")
        self.windpark = self.windpark.drop(columns=self.config.bidding_area_col)

        name_map = {
            "Tonst": "Tonst. Vindpark",
            "Oklaverk": "Okla Vindkraftverk",
            "Sandøy": "Sandøy Vindkraft",
            "Sørfj": "Sørfj. Vindkraft"
        }

        merged[self.config.sid_col] = merged[self.config.sid_col].replace(name_map)

        meta_merged = merged.merge(
            self.windpark[[self.config.substation_name_col, self.config.region_col, self.config.operating_power_max_col, self.config.prod_start_new_col]],
            left_on=self.config.sid_col,
            right_on=self.config.substation_name_col,
            how="left"
        ).drop(columns=[self.config.substation_name_col])

        wcol = "operating_power_max"
        keys = ["region", "time_ref", "time", "lt"]

        wmean_cols = [
            "ws10m_mean","ws10m_std","rh2m_mean","rh2m_std","t2m_mean","t2m_std",
            "g10m_mean","g10m_std","mslp_mean","mslp_std",
            "air_temperature_2m","air_pressure_at_sea_level","relative_humidity_2m",
            "wind_speed_10m"
        ]

        region_df = self._add_windparks_into_regions(meta_merged, wcol, keys, wmean_cols)

        prec = meta_merged.assign(precipitation_amount=pd.to_numeric(meta_merged["precipitation_amount"], errors="coerce")) \
            .groupby(keys)["precipitation_amount"].sum().reset_index()

        region_df = region_df.merge(prec, on=keys, how="left")

        region_df = self._add_circular_mean(region_df, meta_merged, "wd10m_mean", "wd10m_mean", wcol, keys)
        region_df = self._add_circular_mean(region_df, meta_merged, "wind_direction_10m", "wind_direction_10m", wcol, keys)
        
        self.power = self.power.reset_index().rename(columns={"index": "time"})

        self.power = self.power.melt(
            id_vars=self.config.power_time_col,
            var_name=self.config.region_col,
            value_name=self.config.power_col
        )

        self.power[self.config.region_col] = self.power[self.config.region_col].str.replace(
            "ELSPOT ", "", regex=False
        )

        final_dataset = region_df.merge(
            self.power,
            on=[self.config.region_col, self.config.power_time_col],
            how="left"
        )

        self.save_merged_dataset(final_dataset, "src/processed_data/pipeline_data/alligned_data.parquet")


if __name__ == "__main__":
    """Runs the full alignment pipeline and saves the processed dataset."""
    forecast_nowcast_alling = Forecast_Nowcast_Allign(
        path_forecast="src/raw_data/met_forecast.parquet",
        path_nowcast="src/raw_data/met_nowcast.parquet",
        path_windpark_bindzone="src/raw_data/windparks_bidzone.csv",
        path_power_per_windpark="src/raw_data/wind_power_per_bidzone.parquet"
        )
    forecast_nowcast_alling.merge_datasets()
