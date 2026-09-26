
import pandas as pd
import numpy as np
from ntl_functions import _date_cols


# --- date filtering ---
def _between(cols, start, end):
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    return [c for c in cols if s <= pd.to_datetime(c) <= e]


# --- melt to long format ---
def _to_long(wide_df):
    date_cols = _date_cols(wide_df)
    long = wide_df.melt(
        id_vars=["Latitude", "Longitude"],
        value_vars=date_cols,
        var_name="Date",
        value_name="DNB"
    )
    long["Date"] = pd.to_datetime(long["Date"])
    long["DNB"]  = pd.to_numeric(long["DNB"], errors="coerce")
    long = long.dropna(subset=["DNB"])
    long = long[(long["DNB"] >= 0) & (long["DNB"] < 1e6)].copy()
    return long


# --- assign 16-day vza group ---
def _assign_vza_group(df, b0):
    df = df.copy()
    df["vza_group"] = ((df["Date"] - b0).dt.days % 16) + 1
    return df


# --- filter pixels with enough baseline observations ---
def _filter_min_obs(coef_data, min_obs_per_pixel):
    n_pixel = (
        coef_data.groupby(["Latitude", "Longitude"])["DNB"]
        .count()
        .reset_index()
        .rename(columns={"DNB": "n_pixel_obs"})
    )
    coef_data = coef_data.merge(n_pixel, on=["Latitude", "Longitude"], how="left")
    return coef_data[coef_data["n_pixel_obs"] >= min_obs_per_pixel].copy()


# --- compute per-pixel baseline median ---
def _baseline_median(coef_data):
    return (
        coef_data.groupby(["Latitude", "Longitude"])["DNB"]
        .median()
        .reset_index()
        .rename(columns={"DNB": "DNB_baseline_median"})
    )


# --- compute per-pixel per-group median ---
def _group_median(coef_data):
    return (
        coef_data.groupby(["Latitude", "Longitude", "vza_group"])["DNB"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"median": "DNB_group_median", "count": "n_group_obs"})
    )


# --- compute correction factor A_i ---
def _compute_coef(group_med, baseline_med, min_obs_per_group):
    coef = group_med.merge(baseline_med, on=["Latitude", "Longitude"], how="left")
    coef["A_i"] = coef["DNB_group_median"] / coef["DNB_baseline_median"]
    coef.loc[
        (coef["DNB_baseline_median"] <= 0) |
        (coef["n_group_obs"] < min_obs_per_group) |
        (~np.isfinite(coef["A_i"])) |
        (coef["A_i"] <= 0),
        "A_i"
    ] = np.nan
    return coef


# --- apply correction and return corrected values ---
def _apply_coef(long, coef, b0, s1):
    apply_data = long[long["Date"] <= s1].copy()
    apply_data  = _assign_vza_group(apply_data, b0)
    apply_data  = apply_data.merge(
        coef[["Latitude", "Longitude", "vza_group", "A_i"]],
        on=["Latitude", "Longitude", "vza_group"],
        how="left"
    )
    apply_data["DNB_corrected"] = (
        apply_data["DNB"].astype(float) /
        apply_data["A_i"].astype(float)
    )
    apply_data["Date_str"] = apply_data["Date"].dt.strftime("%Y-%m-%d")
    return apply_data


# --- pivot corrected long to wide ---
def _pivot_wide(apply_data):
    return (
        apply_data.pivot_table(
            index=["Latitude", "Longitude"],
            columns="Date_str",
            values="DNB_corrected",
            aggfunc="median"
        )
        .reset_index()
        .sort_values(["Latitude", "Longitude"], ascending=[False, True])
    )


# --- main entry point ---
def apply_angular_correction(
    wide_df,
    baseline_start,
    baseline_end,
    series_end,
    min_obs_per_pixel=30,
    min_obs_per_group=2
):
    b0 = pd.to_datetime(baseline_start)
    b1 = pd.to_datetime(baseline_end)
    s1 = pd.to_datetime(series_end)

    long      = _to_long(wide_df)
    coef_data = long[(long["Date"] >= b0) & (long["Date"] <= b1)].copy()
    coef_data = _assign_vza_group(coef_data, b0)
    coef_data = _filter_min_obs(coef_data, min_obs_per_pixel)

    baseline_med = _baseline_median(coef_data)
    group_med    = _group_median(coef_data)
    coef         = _compute_coef(group_med, baseline_med, min_obs_per_group)

    apply_data    = _apply_coef(long, coef, b0, s1)
    corrected_wide = _pivot_wide(apply_data)

    return corrected_wide, apply_data, coef