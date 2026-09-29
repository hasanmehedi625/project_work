"""
data_collection_pipeline_idalia_fl_2023.py

Per-county VIIRS nighttime-lights (VNP46A2) + EagleI power-outage data
collection pipeline for Hurricane Idalia (Florida, 2023).

For each county this script:
  1. Extracts raw DNB radiance from VNP46A2 granules over the county's
     bounding box, for the baseline period through the post-landfall window.
  2. Applies BRDF/angular correction.
  3. Clips the corrected radiance to the county boundary.
  4. Attaches building-footprint counts and estimated customers per pixel.
  5. Filters out pixels with too few customers.
  6. Computes baseline (pre-storm) pixel statistics.
  7. Truncates the series to the configured end date.
  8. Filters out dates with insufficient customer coverage.
  9. Reshapes to long format (one row per pixel-date).
  10. Computes deviation metrics (mean/median-referenced) from the baseline.
  11. Joins EagleI county-daily outage ground truth on (fips_code, Date) --
      never on county name, which avoids cross-state name collisions
      (e.g. a "Jefferson" county in the wrong state).
  12. Filters to rows with meaningful outage and writes the final CSV.

Run modes:
  --county-index N   process a single county by index into COUNTIES
                      (intended for SLURM array jobs, one task per county)
  --combine          skip processing; just concatenate whatever per-county
                      13_final.csv files already exist under OUTPUT_ROOT
                      into one training CSV (run after all array tasks finish)
  (no flag)           process every county in COUNTIES sequentially, then combine

Configuration:
  All filesystem paths are read from environment variables (with sensible
  defaults for a `./data` layout) rather than hardcoded, so this script can
  run on any machine. See the README for the expected directory layout, or
  set OUTAGE_BASE to point at your own data root.

Dependencies:
  pandas, plus the local helper modules `ntl_functions_update1.py` and
  `angular_correction_functions.py` (not included here -- see README).
"""

import os
import traceback
from datetime import datetime
import argparse
import pandas as pd

from ntl_functions_update1 import (
    BASELINE_STAT_COLS,
    DEVIATION_COLS,
    TARGET_COL,
    add_building_count,
    add_customers,
    add_deviation_metrics,
    add_fraction_outage,
    clip_to_county,
    compute_pixel_stats,
    extract_mosaic_radiance,
    filter_by_outage,
    filter_dates_by_coverage,
    get_county_bbox,
    get_county_fips,
    get_total_customers_from_outage,
    load_daily_outage,
    save,
    to_long_radiance,
    truncate_dates,
)
from angular_correction_functions import apply_angular_correction


# =============================================================================
# --- Event config ---
# =============================================================================

HURRICANE_NAME = "idalia_fl_2023"
STATE_NAME = "Florida"
STATE_FIPS = "12"

# Alphabetical, lower_snake_case keys. Index order defines the
# --county-index / SLURM array indices (0-18).
COUNTIES = [
    "alachua",
    "baker",
    "bradford",
    "citrus",
    "columbia",
    "dixie",
    "gilchrist",
    "gulf",
    "hamilton",
    "jefferson",
    "lafayette",
    "leon",
    "levy",
    "madison",
    "nassau",
    "suwannee",
    "taylor",
    "union",
    "wakulla",
]

# Every county in this list is a single word, so the default
# `county_key.replace("_", " ").title()` already produces the correct
# official Census name for all of them -- no overrides needed.
COUNTY_DISPLAY_OVERRIDES = {}

# Maps a county key to a non-default building-footprint filename, if any
# county's GeoJSON doesn't follow "<county_key>_buildings.geojson".
BUILDINGS_FILENAME_OVERRIDES = {}

# --- Paths ---
# All paths derive from OUTAGE_BASE, which defaults to ./data. Point it at
# your own data root via the environment, e.g.:
#   export OUTAGE_BASE=/path/to/your/data
BASE = os.environ.get("OUTAGE_BASE", os.path.join(os.getcwd(), "data"))

# Expected layout under BASE (see README for details):
#   H_Idalia_FL_2023/vnp46a2/                       VNP46A2 granules (.h5)
#   tl_2025_us_county/tl_2025_us_county.shp          US county boundaries
#   building_footprint_ms/Florida/                   building footprints + customer lookup
#   ground_truth/fl_2014_2025_county_daily_outage.csv  EagleI county-daily outage table
#   outputs/                                         pipeline outputs (created automatically)
A2_DIR = f"{BASE}/H_Idalia_FL_2023/vnp46a2"
OUTPUT_ROOT = f"{BASE}/outputs/hurricane_idalia_fl_2023"

COUNTY_SHP = f"{BASE}/tl_2025_us_county/tl_2025_us_county.shp"
BUILDINGS_DIR = f"{BASE}/building_footprint_ms/Florida"
CUSTOMERS_CSV = f"{BASE}/building_footprint_ms/Florida/florida_counties_customers.csv"
OUTAGE_CSV = f"{BASE}/ground_truth/fl_2014_2025_county_daily_outage.csv"

TRAINING_DATA_OUT = os.path.join(
    OUTPUT_ROOT, f"training_data_all_counties_{HURRICANE_NAME}.csv"
)

# --- Temporal window ---
# Idalia made landfall near Keaton Beach, FL (Big Bend) on 2023-08-30.
# Baseline = the 365 days before landfall; series runs ~36 days after
# landfall to capture the outage and recovery period.
BASELINE_START = "2022-08-30"
BASELINE_END = "2023-08-29"
SERIES_END = "2023-10-05"

# --- Filters ---
MIN_COVERAGE = 0.90              # min fraction of customers with valid data per date
MIN_OUTAGE_FRACTION = 0.01       # min fraction_outage for a row to be kept
MIN_CUSTOMERS_PER_PIXEL = 1
MIN_OUTAGE_INTERVALS = None      # e.g. 6 to require half-day EagleI coverage
LOG_CLIP = None                  # e.g. (-3.0, 1.0) to mask implausible log ratios

# fraction_outage is already on the 0-1 scale; percent_outage is the fallback.
OUTAGE_VALUE_COL = TARGET_COL
PERCENT_FALLBACK_SCALE = 100.0

LEGACY_TARGET_ALIAS = None
PREFER_OUTAGE_TABLE_CUSTOMERS = True

# --- Caching ---
# When True, re-running a county reuses any intermediate CSV already on disk
# instead of recomputing that step.
USE_CACHE = True

TRAINING_COLUMNS = (
    ["Latitude", "Longitude", "NAME", "STATEFP", "fips_code", "Date", "hurricane",
     "building_count", "customers_per_pixel", "total_customers", "ntl"]
    + BASELINE_STAT_COLS
    + DEVIATION_COLS
    + ([TARGET_COL] if LEGACY_TARGET_ALIAS is None else [TARGET_COL, LEGACY_TARGET_ALIAS])
)


# =============================================================================
# --- Naming helpers ---
# =============================================================================

def make_county_folder_name(county):
    return county.lower().replace(".", "").replace(" ", "_").replace("-", "_")


def pretty_county_name(county_key):
    return COUNTY_DISPLAY_OVERRIDES.get(county_key, county_key.replace("_", " ").title())


def buildings_geojson_name(county_key):
    if county_key in BUILDINGS_FILENAME_OVERRIDES:
        return BUILDINGS_FILENAME_OVERRIDES[county_key]
    return f"{make_county_folder_name(county_key)}_buildings.geojson"


def load_customer_lookup(csv_path):
    if not os.path.exists(csv_path):
        print(f"  Customer lookup not found at {csv_path}; using outage-table totals.")
        return None
    df = pd.read_csv(csv_path, dtype={"fips_code": str})
    df["county_clean"] = df["county"].astype(str).str.strip().str.lower()
    return df


def resolve_total_customers(lookup, daily_outage, county_key, fips_code):
    """Resolve the county customer denominator, preferring the outage table."""
    from_outage = get_total_customers_from_outage(daily_outage, fips_code)
    if PREFER_OUTAGE_TABLE_CUSTOMERS and from_outage:
        return from_outage

    if lookup is not None:
        match = lookup[lookup["county_clean"] == pretty_county_name(county_key).lower()]
        if not match.empty:
            return int(match["total_customers"].iloc[0])

    if from_outage:
        return from_outage

    raise ValueError(f"No customer total available for '{county_key}' (fips {fips_code}).")


def cached_read(path):
    if USE_CACHE and os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"    Cache hit -> {os.path.basename(path)}")
        return pd.read_csv(path)
    return None


# =============================================================================
# --- Per-county pipeline ---
# =============================================================================

def process_county(county, lookup, daily_outage):
    county_folder = make_county_folder_name(county)
    display_name = pretty_county_name(county)
    output_dir = os.path.join(OUTPUT_ROOT, county_folder)
    os.makedirs(output_dir, exist_ok=True)

    fips_code = get_county_fips(COUNTY_SHP, display_name, STATE_FIPS)
    total_cust = resolve_total_customers(lookup, daily_outage, county, fips_code)
    print(f"  FIPS {fips_code} | total customers {total_cust:,}")

    # --- Step 1: Extract raw DNB radiance ---
    print("  [1/12] Extracting raw DNB radiance...")
    raw_path = os.path.join(output_dir, "01_raw_wide.csv")
    raw_df = cached_read(raw_path)
    if raw_df is None:
        bbox = get_county_bbox(COUNTY_SHP, display_name, STATE_FIPS)
        raw_df = extract_mosaic_radiance(A2_DIR, bbox, SERIES_END, start_date=BASELINE_START)
        if raw_df.empty:
            raise ValueError("No usable granules found for this county and window.")
        save(raw_df, raw_path)

    # --- Step 2: Angular correction ---
    print("  [2/12] Applying angular correction...")
    corrected_path = os.path.join(output_dir, "02_angular_corrected_wide.csv")
    corrected_wide = cached_read(corrected_path)
    if corrected_wide is None:
        corrected_wide, corrected_long, angular_coef = apply_angular_correction(
            wide_df=raw_df,
            baseline_start=BASELINE_START,
            baseline_end=BASELINE_END,
            series_end=SERIES_END,
            min_obs_per_pixel=50,
            min_obs_per_group=5,
        )
        save(corrected_wide, corrected_path)
        save(corrected_long, os.path.join(output_dir, "02_angular_corrected_long.csv"))
        save(angular_coef, os.path.join(output_dir, "02_angular_coefficients.csv"))

    # --- Step 3: Spatial clipping ---
    print("  [3/12] Clipping to county boundary...")
    clipped_df = save(
        clip_to_county(corrected_wide, COUNTY_SHP, display_name, STATE_FIPS),
        os.path.join(output_dir, "03_clipped.csv"),
    )

    # --- Step 4: Building footprints & customer allocation ---
    print("  [4/12] Appending building footprints and customer metrics...")
    buildings_geojson = os.path.join(BUILDINGS_DIR, buildings_geojson_name(county))
    df_buildings = save(
        add_building_count(clipped_df, buildings_geojson),
        os.path.join(output_dir, "04_buildings.csv"),
    )
    df_customers = save(
        add_customers(df_buildings, total_cust),
        os.path.join(output_dir, "05_customers.csv"),
    )

    # --- Step 5: Customer filter ---
    print(f"  [5/12] Filtering customers_per_pixel >= {MIN_CUSTOMERS_PER_PIXEL}...")
    df_filtered = df_customers[
        df_customers["customers_per_pixel"] >= MIN_CUSTOMERS_PER_PIXEL
    ].reset_index(drop=True)
    df_filtered = save(df_filtered, os.path.join(output_dir, "06_customers_filtered.csv"))

    # --- Step 6: Baseline pixel statistics ---
    print("  [6/12] Computing baseline pixel statistics...")
    df_stats = save(
        compute_pixel_stats(df_filtered, BASELINE_START, BASELINE_END,
                            min_customers=MIN_CUSTOMERS_PER_PIXEL),
        os.path.join(output_dir, "07_pixel_stats.csv"),
    )

    # --- Step 7: Truncate series ---
    print(f"  [7/12] Truncating dates through {SERIES_END}...")
    df_stats = save(
        truncate_dates(df_stats, SERIES_END, start_date=BASELINE_START),
        os.path.join(output_dir, "08_through_series_end.csv"),
    )

    # --- Step 8: Date coverage filter ---
    print(f"  [8/12] Filtering dates with >= {MIN_COVERAGE * 100:.0f}% customer coverage...")
    df_cov = save(
        filter_dates_by_coverage(df_stats, min_coverage=MIN_COVERAGE),
        os.path.join(output_dir, "09_coverage_filtered.csv"),
    )

    # --- Step 9: Melt to long format ---
    print("  [9/12] Melting to long format (raw radiance preserved)...")
    df_long = save(
        to_long_radiance(df_cov, value_name="ntl"),
        os.path.join(output_dir, "10_long_radiance.csv"),
    )

    # --- Step 10: Deviation metrics ---
    print("  [10/12] Computing mean- and median-referenced deviation metrics...")
    df_long = save(
        add_deviation_metrics(df_long, value_col="ntl", log_clip=LOG_CLIP),
        os.path.join(output_dir, "11_long_metrics.csv"),
    )

    # --- Step 11: Outage ground truth join ---
    print("  [11/12] Joining outage ground truth on (fips_code, Date)...")
    df_long = add_fraction_outage(
        df_long, daily_outage, fips_code,
        min_intervals=MIN_OUTAGE_INTERVALS,
        legacy_alias=LEGACY_TARGET_ALIAS,
    )
    df_long = save(df_long, os.path.join(output_dir, "12_long_fraction_outage.csv"))

    # --- Step 12: Outage magnitude filter & metadata ---
    print(f"  [12/12] Filtering by minimum outage ({MIN_OUTAGE_FRACTION:.1%}) and finalizing...")
    df_long = filter_by_outage(df_long, min_fraction=MIN_OUTAGE_FRACTION)
    df_long["total_customers"] = total_cust
    df_long["hurricane"] = HURRICANE_NAME
    df_long["fips_code"] = fips_code
    if "STATEFP" not in df_long.columns:
        df_long["STATEFP"] = STATE_FIPS

    return save(df_long, os.path.join(output_dir, "13_final.csv"))


# =============================================================================
# --- Aggregation ---
# =============================================================================

def combine_all_counties(county_frames, out_path):
    missing_cols = set()
    frames = []

    for county, df in county_frames.items():
        cols_present = [c for c in TRAINING_COLUMNS if c in df.columns]
        missing_cols.update(set(TRAINING_COLUMNS) - set(df.columns))
        frames.append(df[cols_present])

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False)

    print(f"\nCombined training data: {combined.shape}  ->  {out_path}")
    print(f"  Counties : {combined['NAME'].nunique()} | "
          f"County-dates : {combined.groupby(['fips_code', 'Date']).ngroups:,}")
    if missing_cols:
        print(f"  Note: columns absent from at least one county output: {sorted(missing_cols)}")
    return combined


def combine_from_disk(counties_expected, out_path):
    """
    Assemble the final event-level training dataset purely from whatever
    per-county 13_final.csv files already exist under OUTPUT_ROOT.

    This is the step to run after a SLURM array job has finished: each array
    task only processes its own county (see main(), target_counties branch),
    so nothing combines the counties into one training file automatically.
    Run this once, after every array task in the job has completed.
    """
    county_frames = {}
    missing = []

    for county in counties_expected:
        county_folder = make_county_folder_name(county)
        final_path = os.path.join(OUTPUT_ROOT, county_folder, "13_final.csv")
        if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
            county_frames[county] = pd.read_csv(final_path, dtype={"fips_code": str})
        else:
            missing.append(county)

    if missing:
        print(f"Warning: {len(missing)} counties have no 13_final.csv yet and will be "
              f"excluded from this combine: {missing}")

    if not county_frames:
        raise FileNotFoundError(
            f"No 13_final.csv files found under {OUTPUT_ROOT}. "
            "Run the per-county pipeline (--county-index) before combining."
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return combine_all_counties(county_frames, out_path)


# =============================================================================
# --- Main execution ---
# =============================================================================

def main(target_counties=None):
    counties_to_run = target_counties if target_counties else COUNTIES

    print(f"Loading {STATE_NAME} outage ground truth...")
    daily_outage = load_daily_outage(
        OUTAGE_CSV,
        state=STATE_NAME,
        value_col=OUTAGE_VALUE_COL,
        percent_fallback_scale=PERCENT_FALLBACK_SCALE,
    )

    print(f"Loading {STATE_NAME} customer lookup...")
    lookup = load_customer_lookup(CUSTOMERS_CSV)

    results = []
    county_frames = {}

    for county in counties_to_run:
        print(f"\n--- Processing: {county} ---")
        try:
            df_final = process_county(county, lookup, daily_outage)
            if df_final.empty:
                results.append((county, "EMPTY", "No rows survived the outage filter."))
                print(f"Status: {county} produced no qualifying rows.")
                continue
            county_frames[county] = df_final
            results.append((county, "SUCCESS", ""))
            print(f"Status: {county} completed successfully.")
        except Exception as e:
            err_text = f"{type(e).__name__}: {e}"
            print(f"Status: {county} FAILED - {err_text}")
            traceback.print_exc()
            results.append((county, "FAILED", err_text))

    if county_frames and target_counties is None:
        combine_all_counties(county_frames, TRAINING_DATA_OUT)

    n_ok = sum(1 for _, s, _ in results if s == "SUCCESS")
    n_empty = sum(1 for _, s, _ in results if s == "EMPTY")
    n_fail = sum(1 for _, s, _ in results if s == "FAILED")
    print(f"\nExecution summary: {n_ok} succeeded, {n_empty} empty, {n_fail} failed, "
          f"out of {len(results)} total.")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    tag = counties_to_run[0].replace(" ", "_") if target_counties else "all"
    log_path = os.path.join(OUTPUT_ROOT, f"batch_run_log_{tag}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"{HURRICANE_NAME} pipeline run: {datetime.now().isoformat()}\n\n")
        for county, status, detail in results:
            f.write(f"[{status}] {county}" + (f" - {detail}\n" if detail else "\n"))
        f.write(f"\nSummary: {n_ok} succeeded, {n_empty} empty, {n_fail} failed, "
                f"out of {len(results)} total.\n")

    print(f"Log file saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-county VIIRS + EagleI outage data collection for Hurricane Idalia (FL, 2023)."
    )
    parser.add_argument("--county-index", type=int, default=None,
                        help="Index into COUNTIES to process a single county (for SLURM array jobs).")
    parser.add_argument("--combine", action="store_true",
                        help="Skip processing; assemble TRAINING_DATA_OUT from existing "
                             "per-county 13_final.csv files under OUTPUT_ROOT. Run this once, "
                             "after every SLURM array task has finished.")
    args = parser.parse_args()

    if args.combine:
        combine_from_disk(COUNTIES, TRAINING_DATA_OUT)
    elif args.county_index is not None:
        main(target_counties=[COUNTIES[args.county_index]])
    else:
        main()
