# Hurricane Idalia (FL, 2023) Outage Data Collection Pipeline

Per-county VIIRS nighttime-lights (VNP46A2) + EagleI power-outage data
collection pipeline for Hurricane Idalia, built for a satellite-based
hurricane power-outage prediction model.

For each of Florida's 19 counties in Idalia's path, the pipeline extracts
nighttime radiance, corrects it for viewing/illumination angle, clips it to
the county boundary, allocates it to estimated customers per pixel, and
joins it against ground-truth outage data -- producing a per-pixel,
per-date training table with an outage label.

## Pipeline steps

1. Extract raw DNB radiance from VNP46A2 granules over the county bounding box
2. Apply angular (BRDF) correction
3. Clip to the county boundary
4. Attach building-footprint counts and estimated customers per pixel
5. Filter out pixels with too few customers
6. Compute baseline (pre-storm) pixel statistics
7. Truncate the series to the configured end date
8. Filter out dates with insufficient customer coverage
9. Reshape to long format (one row per pixel-date)
10. Compute deviation metrics from baseline
11. Join EagleI outage ground truth on `(fips_code, Date)` -- never on county
    name, which avoids cross-state name collisions (e.g. two states each
    having a "Jefferson" county)
12. Filter to rows with meaningful outage and write `13_final.csv`

## Setup

```bash
pip install -r requirements.txt
```

You'll also need two local helper modules, not included in this repo:
`ntl_functions_update1.py` and `angular_correction_functions.py`.

## Data layout

Set `OUTAGE_BASE` to point at your data root (defaults to `./data`):

```bash
export OUTAGE_BASE=/path/to/your/data
```

Expected layout under `OUTAGE_BASE`:

```
data/
├── H_Idalia_FL_2023/
│   └── vnp46a2/                                   VNP46A2 granules (.h5)
├── tl_2025_us_county/
│   └── tl_2025_us_county.shp                      US county boundaries (TIGER/Line)
├── building_footprint_ms/Florida/
│   ├── <county>_buildings.geojson                 per-county building footprints
│   └── florida_counties_customers.csv             county customer counts (optional --
│                                                    falls back to outage-table totals)
├── ground_truth/
│   └── fl_2014_2025_county_daily_outage.csv        EagleI county-daily outage table
└── outputs/                                        created automatically
```

## Usage

Process every county, then combine:

```bash
python data_collection_pipeline_idalia_fl_2023.py
```

Process a single county (index into `COUNTIES`, e.g. for a SLURM array job):

```bash
python data_collection_pipeline_idalia_fl_2023.py --county-index 0
```

Combine per-county outputs into one training CSV, without reprocessing:

```bash
python data_collection_pipeline_idalia_fl_2023.py --combine
```

### Running on a SLURM cluster

See `run_idalia_fl_2023.sh` for an example array-job launcher that submits
one task per county, then chains a combine job to run automatically once
the array finishes.

## Output

`outputs/hurricane_idalia_fl_2023/training_data_all_counties_idalia_fl_2023.csv`
-- one row per pixel-date, with radiance, deviation metrics, building/customer
counts, and the `fraction_outage` label.
