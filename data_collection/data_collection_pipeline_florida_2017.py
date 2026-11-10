import os
import re
import traceback
from datetime import datetime
import argparse
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from ntl_functions_update2 import (
    BASELINE_STAT_COLS,
    DEVIATION_COLS,
    OUTAGE_VALUE_COLS,
    TARGET_COL,
    add_building_count,
    add_customers,
    add_deviation_metrics,
    add_fraction_outage,
    clean_rows,
    compute_pixel_stats,
    extract_mosaic_radiance,
    filter_by_outage,
    filter_dates_by_coverage,
    get_clean_baseline_dates,
    get_total_customers_from_outage,
    resolve_outage_col,
    load_daily_outage,
    save,
    to_long_radiance,
    truncate_dates,
)
from angular_correction_functions import apply_angular_correction


# --- Config ---

HURRICANE_NAME = "outage_FL_2017"
STATE_NAME = "Florida"
STATE_FIPS = "12"

BASE = "/scratch/user/mdmehedi.hasan/20261029_100722/Power Outage Prediction Model"

A2_DIR = f"{BASE}/2017_vnp46a2"
OUTPUT_ROOT = f"{BASE}/outputs/{HURRICANE_NAME}"
COUNTY_SHP = f"{BASE}/tl_2025_us_county/tl_2025_us_county.shp"
BUILDINGS_DIR = f"{BASE}/building_footprint_ms/Florida"
CUSTOMERS_CSV = f"{BASE}/building_footprint_ms/Florida/florida_counties_customers.csv"
OUTAGE_CSV = f"{BASE}/ground_truth/eaglei_outage_panel/county_night_panel_2017.csv"

TRAINING_DATA_OUT = os.path.join(
    OUTPUT_ROOT, f"training_data_all_counties_{HURRICANE_NAME}.csv"
)

BASELINE_START = "2017-01-01"
BASELINE_END = "2017-12-31"
SERIES_END = "2017-12-31"

MIN_COVERAGE = 0.90
MIN_OUTAGE_FRACTION = 0.01
MIN_CUSTOMERS_PER_PIXEL = 1
MIN_OUTAGE_INTERVALS = None
LOG_CLIP = None

MAX_BASELINE_OUTAGE = MIN_OUTAGE_FRACTION
OUTAGE_EVENT_COLS = OUTAGE_VALUE_COLS
OUTAGE_VALID_RANGE = (0.0, 1.0)

PREFER_OUTAGE_TABLE_CUSTOMERS = True
USE_CACHE = True

COUNTY_GEOID_OVERRIDES = {}
BUILDINGS_FILENAME_OVERRIDES = {}


# --- County List ---

def _load_counties_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    name_col_candidates = [c for c in df.columns
                           if c.strip().lower() in ("county", "county_name", "name")]
    col = name_col_candidates[0] if name_col_candidates else df.columns[0]
    names = df[col].dropna().astype(str).str.strip()

    seen = set()
    ordered = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)
    if not ordered:
        raise ValueError(f"No counties found in {csv_path}.")
    return ordered


COUNTIES = _load_counties_from_csv(CUSTOMERS_CSV)


# --- Naming Helpers ---

def make_county_folder_name(county):
    return county.lower().replace(".", "").replace(" ", "_").replace("-", "_")


def _norm(name):
    return re.sub(r"\s+", " ", str(name).lower().replace(".", "").strip())


def buildings_candidates(county, county_rec):
    if county in BUILDINGS_FILENAME_OVERRIDES:
        return [BUILDINGS_FILENAME_OVERRIDES[county]]
    key_slug = make_county_folder_name(county)
    full_slug = make_county_folder_name(county_rec["NAMELSAD"].iloc[0])
    cands = [
        f"{key_slug}_buildings.geojson",
        f"{key_slug}_county_buildings.geojson",
        f"{full_slug}_buildings.geojson",
    ]
    return list(dict.fromkeys(cands))


def find_buildings_file(county, county_rec):
    tried = buildings_candidates(county, county_rec)
    on_disk = {f.lower(): f for f in os.listdir(BUILDINGS_DIR)}
    for name in tried:
        if name.lower() in on_disk:
            return os.path.join(BUILDINGS_DIR, on_disk[name.lower()])
    raise FileNotFoundError(
        f"No building-footprint file for '{county}' in {BUILDINGS_DIR}; tried {tried}. "
        "Add an entry to BUILDINGS_FILENAME_OVERRIDES or fix BUILDINGS_DIR."
    )


def load_customer_lookup(csv_path):
    if not os.path.exists(csv_path):
        print(f"  Customer lookup not found at {csv_path}; using outage-table totals.")
        return None
    df = pd.read_csv(csv_path, dtype={"fips_code": str})
    df["county_clean"] = df["county"].astype(str).str.strip().str.lower()
    return df


def resolve_total_customers(lookup, daily_outage, county, fips_code):
    from_outage = get_total_customers_from_outage(daily_outage, fips_code)
    if PREFER_OUTAGE_TABLE_CUSTOMERS and from_outage:
        return from_outage

    if lookup is not None:
        match = lookup[lookup["county_clean"] == county.strip().lower()]
        if not match.empty:
            return int(match["total_customers"].iloc[0])

    if from_outage:
        return from_outage

    raise ValueError(f"No customer total available for '{county}' (fips {fips_code}).")


def cached_read(path):
    if USE_CACHE and os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"    Cache hit -> {os.path.basename(path)}")
        return pd.read_csv(path)
    return None


# --- Geometry ---

_STATE_COUNTIES = None


def _state_counties():
    global _STATE_COUNTIES
    if _STATE_COUNTIES is None:
        gdf = gpd.read_file(COUNTY_SHP)
        gdf = gdf[gdf["STATEFP"] == str(STATE_FIPS).zfill(2)].copy()
        if "NAMELSAD" not in gdf.columns:
            gdf["NAMELSAD"] = gdf["NAME"]
        _STATE_COUNTIES = gdf
    return _STATE_COUNTIES


def resolve_county(county_name):
    gdf = _state_counties()

    if county_name in COUNTY_GEOID_OVERRIDES:
        geoid = str(COUNTY_GEOID_OVERRIDES[county_name]).zfill(5)
        match = gdf[gdf["GEOID"] == geoid]
        if match.empty:
            raise ValueError(f"GEOID override {geoid} for '{county_name}' not in shapefile.")
        return match.iloc[[0]]

    match = gdf[gdf["NAME"] == county_name]
    if len(match) == 1:
        return match.iloc[[0]]

    loose = gdf[gdf["NAME"].map(_norm) == _norm(county_name)]
    if len(loose) == 1:
        print(f"  (matched '{county_name}' loosely -> '{loose['NAME'].iloc[0]}')")
        return loose.iloc[[0]]
    if len(loose) > 1:
        options = ", ".join(f"{r.NAMELSAD} ({r.GEOID})" for r in loose.itertuples())
        raise ValueError(
            f"'{county_name}' is ambiguous in the shapefile: {options}. "
            "Add it to COUNTY_GEOID_OVERRIDES."
        )
    raise ValueError(f"'{county_name}' not found in {COUNTY_SHP} for STATEFP={STATE_FIPS}.")


def county_bbox(county_rec, buffer=0.05):
    b = county_rec.to_crs("EPSG:4326").total_bounds
    return (b[1] - buffer, b[3] + buffer, b[0] - buffer, b[2] + buffer)


def clip_to_county_record(df, county_rec):
    attrs = county_rec[["NAME", "STATEFP", "COUNTYFP", "GEOID", "geometry"]]
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(x, y) for x, y in zip(df["Longitude"], df["Latitude"])],
        crs="EPSG:4326",
    ).to_crs(county_rec.crs)

    joined = gpd.sjoin(gdf, attrs, predicate="within")
    joined = joined.drop(columns=["index_right", "geometry"]).reset_index(drop=True)
    print(f"    Pixels inside county boundary : {len(joined):,} / {len(df):,}")
    return pd.DataFrame(joined)


# --- Per-County Pipeline ---

def process_county(county, lookup, daily_outage, event_col):
    county_folder = make_county_folder_name(county)
    output_dir = os.path.join(OUTPUT_ROOT, county_folder)
    os.makedirs(output_dir, exist_ok=True)

    county_rec = resolve_county(county)
    fips_code = str(county_rec["GEOID"].iloc[0]).zfill(5)
    full_name = county_rec["NAME"].iloc[0]
    total_cust = resolve_total_customers(lookup, daily_outage, county, fips_code)
    print(f"  {full_name} | FIPS {fips_code} | total customers {total_cust:,}")

    print("  [1/12] Extracting raw DNB radiance...")
    raw_path = os.path.join(output_dir, "01_raw_wide.csv")
    raw_df = cached_read(raw_path)
    if raw_df is None:
        raw_df = extract_mosaic_radiance(A2_DIR, county_bbox(county_rec), SERIES_END,
                                         start_date=BASELINE_START)
        if raw_df.empty:
            raise ValueError("No usable granules found for this county and window.")
        save(raw_df, raw_path)

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

    print("  [3/12] Clipping to county boundary...")
    clipped_df = save(
        clip_to_county_record(corrected_wide, county_rec),
        os.path.join(output_dir, "03_clipped.csv"),
    )

    print("  [4/12] Appending building footprints and customer metrics...")
    buildings_geojson = find_buildings_file(county, county_rec)
    df_buildings = save(
        add_building_count(clipped_df, buildings_geojson),
        os.path.join(output_dir, "04_buildings.csv"),
    )
    df_customers = save(
        add_customers(df_buildings, total_cust),
        os.path.join(output_dir, "05_customers.csv"),
    )

    print(f"  [5/12] Filtering customers_per_pixel >= {MIN_CUSTOMERS_PER_PIXEL}...")
    df_filtered = df_customers[
        df_customers["customers_per_pixel"] >= MIN_CUSTOMERS_PER_PIXEL
    ].reset_index(drop=True)
    df_filtered = save(df_filtered, os.path.join(output_dir, "06_customers_filtered.csv"))

    print("  [6/12] Computing baseline pixel statistics...")
    clean_dates = get_clean_baseline_dates(
        daily_outage, fips_code, BASELINE_START, BASELINE_END,
        max_fraction_outage=MAX_BASELINE_OUTAGE,
        value_cols=OUTAGE_EVENT_COLS,
        bounds=OUTAGE_VALID_RANGE,
    )
    df_stats = save(
        compute_pixel_stats(df_filtered, BASELINE_START, BASELINE_END,
                            min_customers=MIN_CUSTOMERS_PER_PIXEL,
                            valid_dates=clean_dates),
        os.path.join(output_dir, "07_pixel_stats.csv"),
    )

    print(f"  [7/12] Truncating dates through {SERIES_END}...")
    df_stats = save(
        truncate_dates(df_stats, SERIES_END, start_date=BASELINE_START),
        os.path.join(output_dir, "08_through_series_end.csv"),
    )

    print(f"  [8/12] Filtering dates with >= {MIN_COVERAGE * 100:.0f}% customer coverage...")
    df_cov = save(
        filter_dates_by_coverage(df_stats, min_coverage=MIN_COVERAGE),
        os.path.join(output_dir, "09_coverage_filtered.csv"),
    )

    print("  [9/12] Melting to long format (raw radiance preserved)...")
    df_long = save(
        to_long_radiance(df_cov, value_name="ntl"),
        os.path.join(output_dir, "10_long_radiance.csv"),
    )

    print("  [10/12] Computing mean- and median-referenced deviation metrics...")
    df_long = save(
        add_deviation_metrics(df_long, value_col="ntl", log_clip=LOG_CLIP),
        os.path.join(output_dir, "11_long_metrics.csv"),
    )

    print("  [11/12] Joining outage ground truth on (fips_code, Date)...")
    df_long = add_fraction_outage(
        df_long, daily_outage, fips_code,
        min_intervals=MIN_OUTAGE_INTERVALS,
    )

    df_long = clean_rows(df_long, outage_cols=OUTAGE_EVENT_COLS,
                         bounds=OUTAGE_VALID_RANGE)
    df_long = save(df_long, os.path.join(output_dir, "12_long_fraction_outage.csv"))

    print(f"  [12/12] Keeping days with {event_col} >= {MIN_OUTAGE_FRACTION:.1%} "
          f"(the days excluded from the baseline)...")
    df_long = filter_by_outage(df_long, min_fraction=MIN_OUTAGE_FRACTION,
                               target_col=event_col)
    df_long["total_customers"] = total_cust
    df_long["hurricane"] = HURRICANE_NAME
    df_long["fips_code"] = fips_code
    df_long["NAME"] = full_name
    if "STATEFP" not in df_long.columns:
        df_long["STATEFP"] = STATE_FIPS

    return save(df_long, os.path.join(output_dir, "13_final.csv"))


# --- Aggregation ---

def training_columns(event_col):
    cols = (
        ["Latitude", "Longitude", "NAME", "STATEFP", "fips_code", "Date", "hurricane",
         "building_count", "customers_per_pixel", "total_customers", "ntl"]
        + BASELINE_STAT_COLS
        + DEVIATION_COLS
        + [c for c in OUTAGE_VALUE_COLS]
    )
    if event_col not in cols:
        cols.append(event_col)
    return cols


def combine_all_counties(county_frames, out_path, event_col):
    wanted = training_columns(event_col)
    missing_cols = set()
    frames = []

    for county, df in county_frames.items():
        cols_present = [c for c in wanted if c in df.columns]
        missing_cols.update(set(wanted) - set(df.columns))
        frames.append(df[cols_present])

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False)

    print(f"\nCombined training data: {combined.shape}  ->  {out_path}")
    print(f"  Counties : {combined['fips_code'].nunique()} | "
          f"County-dates : {combined.groupby(['fips_code', 'Date']).ngroups:,}")
    if missing_cols:
        print(f"  Note: columns absent from at least one county output: {sorted(missing_cols)}")
    return combined


def combine_from_disk(counties_expected, out_path):
    county_frames = {}
    missing = []

    for county in counties_expected:
        final_path = os.path.join(OUTPUT_ROOT, make_county_folder_name(county), "13_final.csv")
        if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
            county_frames[county] = pd.read_csv(final_path, dtype={"fips_code": str})
        else:
            missing.append(county)

    if missing:
        print(f"Warning: {len(missing)} counties have no 13_final.csv and are excluded "
              f"from this combine ({len(missing)} of {len(counties_expected)}).")
        print(f"  {missing}")

    if not county_frames:
        raise FileNotFoundError(
            f"No 13_final.csv files found under {OUTPUT_ROOT}. "
            "Run the per-county pipeline (--county-index) before combining."
        )

    any_df = next(iter(county_frames.values()))
    event_col = resolve_outage_col(any_df, OUTAGE_EVENT_COLS)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return combine_all_counties(county_frames, out_path, event_col)


# --- Main Execution ---

def main(target_counties=None):
    counties_to_run = target_counties if target_counties else COUNTIES

    print(f"Loading {STATE_NAME} outage ground truth...")
    daily_outage = load_daily_outage(OUTAGE_CSV, state=STATE_NAME)

    event_col = resolve_outage_col(daily_outage, OUTAGE_EVENT_COLS)
    print(f"  Event/baseline column  : {event_col} "
          f"(threshold >= {MIN_OUTAGE_FRACTION:.2%}; "
          f"priority {list(OUTAGE_EVENT_COLS)})")

    print(f"Loading {STATE_NAME} customer lookup...")
    lookup = load_customer_lookup(CUSTOMERS_CSV)

    results = []
    county_frames = {}

    for county in counties_to_run:
        print(f"\n--- Processing: {county} ---")
        try:
            df_final = process_county(county, lookup, daily_outage, event_col)
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
        combine_all_counties(county_frames, TRAINING_DATA_OUT, event_col)

    n_ok = sum(1 for _, s, _ in results if s == "SUCCESS")
    n_empty = sum(1 for _, s, _ in results if s == "EMPTY")
    n_fail = sum(1 for _, s, _ in results if s == "FAILED")
    print(f"\nExecution summary: {n_ok} succeeded, {n_empty} empty, {n_fail} failed, "
          f"out of {len(results)} total.")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    tag = make_county_folder_name(counties_to_run[0]) if target_counties else "all"
    log_path = os.path.join(OUTPUT_ROOT, f"batch_run_log_{tag}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"{HURRICANE_NAME} pipeline run: {datetime.now().isoformat()}\n")
        f.write(f"Baseline {BASELINE_START} .. {BASELINE_END} | series end {SERIES_END}\n")
        f.write(f"Event column: {event_col} (>= {MIN_OUTAGE_FRACTION})\n\n")
        for county, status, detail in results:
            f.write(f"[{status}] {county}" + (f" - {detail}\n" if detail else "\n"))
        f.write(f"\nSummary: {n_ok} succeeded, {n_empty} empty, {n_fail} failed, "
                f"out of {len(results)} total.\n")

    print(f"Log file saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-index", type=int, default=None,
                        help="Index into COUNTIES to process a single county.")
    parser.add_argument("--combine", action="store_true",
                        help="Assemble TRAINING_DATA_OUT from existing 13_final.csv files.")
    parser.add_argument("--count-counties", "--n-counties", dest="count_counties",
                        action="store_true",
                        help="Print len(COUNTIES) and exit.")
    args = parser.parse_args()

    if args.count_counties:
        print(len(COUNTIES))
    elif args.combine:
        combine_from_disk(COUNTIES, TRAINING_DATA_OUT)
    elif args.county_index is not None:
        main(target_counties=[COUNTIES[args.county_index]])
    else:
        main()