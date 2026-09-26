
from __future__ import annotations

import gc
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

# --- Module Constants ---

_HDF = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"

_SKIP = {
    "Latitude", "Longitude", "geometry",
    "building_count", "customers_per_pixel", "total_customers",
    "NAME", "STATEFP", "COUNTYFP", "GEOID", "fips_code", "hurricane",
    "Date", "ntl", "fraction_outage", "pct_cust_out",
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Scale factor converting median absolute deviation to a Gaussian-equivalent SD.
MAD_TO_SD = 1.4826

# Name of the outage target, kept identical to the ground-truth file's column.
TARGET_COL = "fraction_outage"

# Baseline statistic columns produced by compute_pixel_stats.
BASELINE_STAT_COLS = [
    "ntl_n_obs_pre",
    "ntl_mean_pre", "ntl_median_pre", "ntl_min_pre", "ntl_max_pre", "ntl_95th_pre",
    "ntl_sd_pre", "ntl_mad_pre", "ntl_robust_sd_pre", "ntl_iqr_pre",
    "ntl_skew_pre", "ntl_kurtosis_pre", "ntl_cv_pre",
]

# Deviation metric columns produced by add_deviation_metrics.
DEVIATION_MEAN_COLS = [
    "ntl_difference", "relative_ntl_change", "light_loss", "zscore",
    "log_comparability_score_mean",
]

DEVIATION_MEDIAN_COLS = [
    "ntl_difference_median", "relative_ntl_change_median", "light_loss_median",
    "zscore_median", "log_comparability_score",
]

DEVIATION_COLS = DEVIATION_MEAN_COLS + DEVIATION_MEDIAN_COLS

# --- Low-Level Helpers ---

def _date_cols(df):
    """Return columns whose names are ISO dates, ignoring known metadata fields."""
    return [c for c in df.columns if c not in _SKIP and _DATE_RE.match(str(c))]


def _between(cols, start, end):
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    return [c for c in cols if s <= pd.to_datetime(c) <= e]


def _hdf_date(fname):
    p = fname.split(".")[1]
    return datetime.strptime(f"{p[1:5]}-{p[5:8]}", "%Y-%j").strftime("%Y-%m-%d")


def _quality_mask(mqf, cloud, snow, dnb):
    return (mqf == 0) & (((cloud >> 6) & 3) == 0) & (snow == 0) & (dnb >= 0) & (dnb < 1e6)


def _safe_div(num, den, min_den=0.0):
    """Element-wise division with non-positive / missing denominators mapped to NaN."""
    den = pd.to_numeric(den, errors="coerce")
    den = den.where(den > min_den)
    return pd.to_numeric(num, errors="coerce") / den


def _parse_dates(series):
    """Parse a date column that may be m/d/Y or ISO, without silent misreads."""
    parsed = pd.to_datetime(series, format="%m/%d/%Y", errors="coerce")
    if parsed.isna().any():
        fallback = pd.to_datetime(series, errors="coerce")
        parsed = parsed.fillna(fallback)
    return parsed


def save(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"    Saved -> {path}  ({len(df):,} rows)")
    return df

# --- VIIRS Radiance Extraction ---

def extract_mosaic_radiance(data_dir, bbox, end_date, start_date=None, round_deg=6):
    """
    Build a wide (pixel x date) radiance table from VNP46A2 granules.

    Each granule is masked to the bounding box and quality-screened *before* any
    concatenation, so peak memory scales with the county footprint rather than
    with the full tile stack.
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    end_dt = pd.to_datetime(end_date)
    start_dt = pd.to_datetime(start_date) if start_date is not None else None

    files_by_date = defaultdict(list)
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".h5"):
            continue
        try:
            file_dt = pd.to_datetime(_hdf_date(fname))
        except (IndexError, ValueError):
            continue
        if file_dt > end_dt:
            continue
        if start_dt is not None and file_dt < start_dt:
            continue
        files_by_date[file_dt.strftime("%Y-%m-%d")].append(os.path.join(data_dir, fname))

    frames = []
    for date_str, paths in sorted(files_by_date.items()):
        lat_keep, lon_keep, dnb_keep = [], [], []

        for fpath in paths:
            with h5py.File(fpath, "r") as f:
                grid = f[_HDF]
                lat = grid["lat"][:]
                lon = grid["lon"][:]
                if lat.ndim == 1:
                    lat, lon = np.meshgrid(lat, lon, indexing="ij")

                inside = (
                    (lat >= lat_min) & (lat <= lat_max) &
                    (lon >= lon_min) & (lon <= lon_max)
                )
                if not inside.any():
                    continue

                dnb = grid["DNB_BRDF-Corrected_NTL"][:].astype(np.float32)
                mask = inside & _quality_mask(
                    grid["Mandatory_Quality_Flag"][:],
                    grid["QF_Cloud_Mask"][:],
                    grid["Snow_Flag"][:],
                    dnb,
                )
                if not mask.any():
                    continue

                lat_keep.append(np.round(lat[mask], round_deg).astype(np.float64))
                lon_keep.append(np.round(lon[mask], round_deg).astype(np.float64))
                dnb_keep.append(dnb[mask].astype(np.float32))

            del dnb, lat, lon, inside, mask
            gc.collect()

        if not lat_keep:
            continue

        frames.append(pd.DataFrame({
            "Latitude": np.concatenate(lat_keep),
            "Longitude": np.concatenate(lon_keep),
            "Date": date_str,
            "DNB": np.concatenate(dnb_keep),
        }))

    if not frames:
        return pd.DataFrame(columns=["Latitude", "Longitude"])

    long_df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    wide = (
        long_df.pivot_table(index=["Latitude", "Longitude"], columns="Date",
                            values="DNB", aggfunc="mean")
        .reset_index()
        .sort_values(["Latitude", "Longitude"], ascending=[False, True])
        .reset_index(drop=True)
    )
    wide.columns.name = None
    print(f"    Extracted {len(wide):,} pixels across {len(wide.columns) - 2} dates")
    return wide

# --- County Geometry ---

def get_county_record(shp_path, county_name, state_fips):
    counties = gpd.read_file(shp_path)
    county = counties[
        (counties["NAME"] == county_name) &
        (counties["STATEFP"] == str(state_fips).zfill(2))
    ]
    if county.empty:
        raise ValueError(
            f"County '{county_name}' with STATEFP='{str(state_fips).zfill(2)}' not found in {shp_path}"
        )
    return county


def get_county_fips(shp_path, county_name, state_fips):
    """Return the 5-digit GEOID used as the join key against the outage table."""
    county = get_county_record(shp_path, county_name, state_fips)
    return str(county["GEOID"].iloc[0]).zfill(5)


def get_county_bbox(shp_path, county_name, state_fips, buffer=0.05):
    county = get_county_record(shp_path, county_name, state_fips).to_crs("EPSG:4326")
    b = county.total_bounds
    return (b[1] - buffer, b[3] + buffer, b[0] - buffer, b[2] + buffer)


def clip_to_county(df, shp_path, county_name, state_fips,
                   keep_cols=("NAME", "STATEFP", "COUNTYFP", "GEOID")):
    """Spatially clip pixels to the county polygon, retaining only identifier columns."""
    county = get_county_record(shp_path, county_name, state_fips)
    keep = [c for c in keep_cols if c in county.columns]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(x, y) for x, y in zip(df["Longitude"], df["Latitude"])],
        crs="EPSG:4326",
    ).to_crs(county.crs)

    joined = gpd.sjoin(gdf, county[keep + ["geometry"]], predicate="within")
    joined = joined.drop(columns=["index_right", "geometry"]).reset_index(drop=True)
    print(f"    Pixels inside county boundary : {len(joined):,} / {len(df):,}")
    return pd.DataFrame(joined)

# --- Buildings & Customer Allocation ---

def add_building_count(df, buildings_path, pixel_half=0.0025):
    bldg = gpd.read_file(buildings_path).to_crs("EPSG:4326")
    sindex = bldg.sindex

    def _count(lat, lon):
        b = box(lon - pixel_half, lat - pixel_half, lon + pixel_half, lat + pixel_half)
        ix = list(sindex.intersection(b.bounds))
        return int(bldg.iloc[ix].intersects(b).sum()) if ix else 0

    unique = df[["Latitude", "Longitude"]].drop_duplicates().copy()
    unique["building_count"] = unique.apply(lambda r: _count(r.Latitude, r.Longitude), axis=1)
    print(f"    Buildings joined to {len(unique):,} unique pixels")
    return df.merge(unique, on=["Latitude", "Longitude"], how="left")


def add_customers(df, total_customers):
    out = df.copy()
    total_buildings = df["building_count"].sum()
    if total_buildings <= 0:
        raise ValueError("No building footprints intersect this county; cannot allocate customers.")
    out["customers_per_pixel"] = (df["building_count"] / total_buildings) * total_customers
    return out

# --- Baseline Pixel Statistics ---

def compute_pixel_stats(df, baseline_start, baseline_end, min_customers=1):
    """
    Compute per-pixel baseline statistics over the pre-event window.

    These columns are the single source of truth for every deviation metric.
    Negative radiance is treated as missing; zeros are retained because a dark
    pixel is a valid observation.
    """
    date_cols = _date_cols(df)
    pre_cols = _between(date_cols, baseline_start, baseline_end)
    if not pre_cols:
        raise ValueError(f"No baseline dates between {baseline_start} and {baseline_end}.")

    eligible = df["customers_per_pixel"] >= min_customers
    data = df.loc[eligible, pre_cols].apply(pd.to_numeric, errors="coerce")
    data = data.where(data >= 0)

    median = data.median(axis=1)
    mean = data.mean(axis=1)
    sd = data.std(axis=1)
    mad = data.sub(median, axis=0).abs().median(axis=1)

    stats = {
        "ntl_n_obs_pre": data.count(axis=1),
        "ntl_mean_pre": mean,
        "ntl_median_pre": median,
        "ntl_min_pre": data.min(axis=1),
        "ntl_max_pre": data.max(axis=1),
        "ntl_95th_pre": data.quantile(0.95, axis=1),
        "ntl_sd_pre": sd,
        "ntl_mad_pre": mad,
        "ntl_robust_sd_pre": MAD_TO_SD * mad,
        "ntl_iqr_pre": data.quantile(0.75, axis=1) - data.quantile(0.25, axis=1),
        "ntl_skew_pre": data.skew(axis=1),
        "ntl_kurtosis_pre": data.kurtosis(axis=1),
        "ntl_cv_pre": sd / mean.replace(0, np.nan),
    }

    out = df.copy()
    for col in BASELINE_STAT_COLS:
        out[col] = np.nan
    for col, values in stats.items():
        out.loc[eligible, col] = values

    print(f"    Baseline stats computed for {int(eligible.sum()):,} pixels "
          f"over {len(pre_cols)} candidate dates")
    return out

# --- Date Coverage Filter ---

def filter_dates_by_coverage(df, min_coverage=0.90):
    """Drop dates whose valid pixels represent less than min_coverage of customers."""
    date_cols = _date_cols(df)
    total_cust = df["customers_per_pixel"].sum()
    keep, drop = [], []

    for d in date_cols:
        available = df.loc[df[d].notna(), "customers_per_pixel"].sum()
        (keep if available / total_cust >= min_coverage else drop).append(d)

    print(f"    Total customers        : {total_cust:,.0f}")
    print(f"    Dates kept (>= {min_coverage * 100:.0f}%)  : {len(keep)}")
    print(f"    Dates dropped          : {len(drop)}")

    non_date = [c for c in df.columns if c not in date_cols]
    return df[non_date + keep]


def truncate_dates(df, end_date, start_date=None):
    date_cols = _date_cols(df)
    keep = _between(date_cols, start_date or "1900-01-01", end_date)
    non_date = [c for c in df.columns if c not in date_cols]
    return df[non_date + keep]

# --- Long Format Conversion ---

def to_long_radiance(df_wide, value_name="ntl"):
    """Melt the wide radiance table to long format, preserving raw radiance."""
    date_cols = _date_cols(df_wide)
    id_cols = [c for c in df_wide.columns if c not in date_cols]

    long_df = df_wide.melt(id_vars=id_cols, value_vars=date_cols,
                           var_name="Date", value_name=value_name)
    long_df["Date"] = pd.to_datetime(long_df["Date"]).dt.strftime("%Y-%m-%d")
    long_df = long_df.dropna(subset=[value_name]).reset_index(drop=True)
    print(f"    Long format rows       : {len(long_df):,}")
    return long_df

# --- Deviation Metrics ---

def add_deviation_metrics(df_long, value_col="ntl", log_clip=None, drop_clipped=True):
    """
    Append mean- and median-referenced deviation metrics.

    Mean-referenced
        ntl_difference               ntl - ntl_mean_pre
        relative_ntl_change          (ntl - ntl_mean_pre) / ntl_mean_pre
        light_loss                   1 - ntl / ntl_mean_pre
        zscore                       (ntl - ntl_mean_pre) / ntl_sd_pre
        log_comparability_score_mean log(ntl / ntl_mean_pre)

    Median-referenced
        ntl_difference_median        ntl - ntl_median_pre
        relative_ntl_change_median   (ntl - ntl_median_pre) / ntl_median_pre
        light_loss_median            1 - ntl / ntl_median_pre
        zscore_median                (ntl - ntl_median_pre) / (1.4826 * ntl_mad_pre)
        log_comparability_score      log(ntl / ntl_median_pre)

    Non-positive or missing baselines yield NaN rather than a clamped sentinel
    value, so unmeasurable pixels stay distinguishable from genuine darkness.
    If log_clip is supplied, log ratios outside the range are set to NaN when
    drop_clipped is True, otherwise clipped to the bounds.
    """
    out = df_long.copy()

    x = pd.to_numeric(out[value_col], errors="coerce").where(lambda s: s >= 0)
    x_pos = x.where(x > 0)

    mean = out["ntl_mean_pre"]
    median = out["ntl_median_pre"]

    out["ntl_difference"] = x - pd.to_numeric(mean, errors="coerce")
    out["relative_ntl_change"] = _safe_div(out["ntl_difference"], mean)
    out["light_loss"] = 1.0 - _safe_div(x, mean)
    out["zscore"] = _safe_div(x - pd.to_numeric(mean, errors="coerce"), out["ntl_sd_pre"])
    out["log_comparability_score_mean"] = np.log(_safe_div(x_pos, mean))

    out["ntl_difference_median"] = x - pd.to_numeric(median, errors="coerce")
    out["relative_ntl_change_median"] = _safe_div(out["ntl_difference_median"], median)
    out["light_loss_median"] = 1.0 - _safe_div(x, median)
    out["zscore_median"] = _safe_div(
        x - pd.to_numeric(median, errors="coerce"), out["ntl_robust_sd_pre"]
    )
    out["log_comparability_score"] = np.log(_safe_div(x_pos, median))

    if log_clip is not None:
        lo, hi = log_clip
        for col in ("log_comparability_score", "log_comparability_score_mean"):
            if drop_clipped:
                out[col] = out[col].where(out[col].between(lo, hi))
            else:
                out[col] = out[col].clip(lo, hi)

    valid = out["log_comparability_score"].notna().sum()
    print(f"    Deviation metrics added ; valid LCS rows : {valid:,} / {len(out):,}")
    return out

# --- EagleI Outage Ground Truth ---

def load_daily_outage(csv_path, state=None, value_col=TARGET_COL,
                      percent_fallback_scale=100.0, max_plausible=1.05):
    """
    Load the pre-aggregated county-daily outage table.

    `fraction_outage` is already on the 0-1 scale used as the model target and is
    read as-is. If that column is absent, `percent_outage` (0-100 scale) is used
    and divided by percent_fallback_scale. Either way the resulting column is
    named TARGET_COL, so the name is identical in the ground-truth file, the
    per-county outputs, and the pooled training table.
    """
    df = pd.read_csv(csv_path, dtype={"fips_code": str})
    df["fips_code"] = df["fips_code"].astype(str).str.strip().str.zfill(5)

    if state is not None and "state" in df.columns:
        df = df[df["state"].astype(str).str.strip().str.lower() == state.strip().lower()]

    df["Date"] = _parse_dates(df["date"]).dt.strftime("%Y-%m-%d")

    if value_col in df.columns:
        values = pd.to_numeric(df[value_col], errors="coerce")
        source = value_col
    elif "percent_outage" in df.columns:
        values = pd.to_numeric(df["percent_outage"], errors="coerce") / percent_fallback_scale
        source = f"percent_outage / {percent_fallback_scale:g}"
    else:
        raise KeyError(f"Neither '{value_col}' nor 'percent_outage' present in {csv_path}")

    df[TARGET_COL] = values

    keep = [c for c in ["fips_code", "county", "state", "Date", TARGET_COL,
                        "customers_out", "total_customers", "n_intervals"] if c in df.columns]
    df = df[keep].dropna(subset=["Date", TARGET_COL]).reset_index(drop=True)

    over = int((df[TARGET_COL] > max_plausible).sum())
    if over:
        print(f"  WARNING: {over:,} county-days exceed {max_plausible:.2f}; "
              f"check the scale of '{source}'.")

    print(f"  Outage table loaded    : {len(df):,} county-days | "
          f"{df['fips_code'].nunique()} counties | {df['Date'].min()} -> {df['Date'].max()}")
    print(f"  Target column          : {TARGET_COL} (from {source}) | "
          f"range {df[TARGET_COL].min():.6f} -> {df[TARGET_COL].max():.4f}")
    return df


def add_fraction_outage(df_long, daily_outage, fips_code, min_intervals=None,
                        legacy_alias=None):
    """
    Join outage ground truth on (fips_code, Date) — never on county name alone.

    Set legacy_alias to a column name (e.g. "pct_cust_out") to emit a duplicate
    of the target under the old name for code that has not been migrated yet.
    """
    fips_code = str(fips_code).zfill(5)
    sub = daily_outage[daily_outage["fips_code"] == fips_code].copy()

    if sub.empty:
        raise ValueError(f"No outage records for fips_code '{fips_code}' in the outage table.")

    if min_intervals is not None and "n_intervals" in sub.columns:
        before = len(sub)
        sub = sub[sub["n_intervals"] >= min_intervals]
        print(f"    Interval filter (>= {min_intervals}) : {len(sub):,} / {before:,} county-days kept")

    out = df_long.copy()
    out["Date"] = pd.to_datetime(out["Date"].astype(str)).dt.strftime("%Y-%m-%d")
    out = out.merge(sub[["Date", TARGET_COL]], on="Date", how="left")

    if legacy_alias:
        out[legacy_alias] = out[TARGET_COL]

    matched = out[TARGET_COL].notna().sum()
    print(f"    Rows matched to outage : {matched:,} / {len(out):,} "
          f"({100 * matched / max(len(out), 1):.1f}%)")
    return out


def get_total_customers_from_outage(daily_outage, fips_code):
    """Fallback denominator, taken from the same table that defines the target."""
    sub = daily_outage[daily_outage["fips_code"] == str(fips_code).zfill(5)]
    if sub.empty or "total_customers" not in sub.columns:
        return None
    value = pd.to_numeric(sub["total_customers"], errors="coerce").median()
    return None if pd.isna(value) else int(value)


def filter_by_outage(df_long, min_fraction=0.01, target_col=TARGET_COL):
    before = df_long["Date"].nunique()
    out = df_long[df_long[target_col] >= min_fraction].copy()
    after = out["Date"].nunique()
    print(f"    Dates before : {before}  |  kept (>= {min_fraction:.1%}) : {after}  "
          f"|  removed : {before - after}")
    return out
