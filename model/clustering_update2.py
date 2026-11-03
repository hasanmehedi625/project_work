import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.cluster import DBSCAN

# --- configuration ---

DEFAULT_PRED = Path("/home/mdmehedi.hasan/Power Outage Modeling/outage_model_update2_non_event_test/predictions_pixel.csv")
DEFAULT_OUT = Path("/home/mdmehedi.hasan/Power Outage Modeling/clustering_update2_non_event")

PROB_COL = "pixel_outage_probability"   # stage-1 pixel probability
TARGET_COL = "fraction_outage"          # bag-level truth
CUSTOMER_COL = "customers_per_pixel"
GROUP_COL = "county_date"
LAT_COL, LON_COL = "Latitude", "Longitude"

EPS_KM = 1.2        # DBSCAN neighbourhood radius
MIN_WEIGHT = 5      # DBSCAN min_samples: summed probability, not a pixel count
KM_PER_DEG = 111.32
PROB_EPS = 1e-4     # clip before inverting the sigmoid

# --- input ---

def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    if TARGET_COL not in df and "target" in df:
        df[TARGET_COL] = df["target"]

    missing = [c for c in (GROUP_COL, LAT_COL, LON_COL, CUSTOMER_COL, PROB_COL, TARGET_COL)
               if c not in df]
    if missing:
        raise KeyError(f"{missing} not present in {path}")

    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    print(f"{len(df):,} pixels, {df[GROUP_COL].nunique():,} county-dates, "
          f"mean probability {df[PROB_COL].mean():.3f}")
    return df

# --- proximity ---
# s = +1 at a cluster's deepest interior, 0 on its boundary, negative outside.
# Each cluster is divided by its own deepest inside distance R, so s means the
# same thing in a large cluster and a small one. Nothing is clipped: inside,
# s <= 1 because R is the largest inside distance; outside, s keeps falling the
# farther a pixel lies from its nearest cluster, in units of that cluster's R.

def signed_proximity(mask, spacing):
    """s for every cell of the raster. The mask must contain at least one cluster."""
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))   # 8-connected
    half = 0.5 * min(spacing)   # distances are measured from the pixel edge

    inside = ndimage.distance_transform_edt(mask, sampling=spacing) - half
    outside, nearest = ndimage.distance_transform_edt(
        ~mask, sampling=spacing, return_indices=True)
    outside = outside - half

    # R per cluster, floored so a one-pixel cluster cannot divide by nearly 0
    deepest = ndimage.maximum(inside, labels, index=np.arange(1, n + 1))
    radius = np.concatenate([[min(spacing)], np.maximum(deepest, min(spacing))])

    return np.where(mask,
                    inside / radius[labels],
                    -outside / radius[labels[tuple(nearest)]]).astype(np.float32)


def grid_step(values):
    """Smallest gap between distinct coordinates, i.e. the native pixel size."""
    gaps = np.diff(np.unique(np.round(values, 7)))
    gaps = gaps[gaps > 1e-7]
    return float(gaps.min()) if len(gaps) else 1 / 240   # 15 arc-seconds


def proximity_of_pixels(lat, lon, in_cluster):
    """Rasterise the pixels onto their native grid and read s back off it."""
    pad = 2             # blank border so a cluster at the edge has an outside
    step_lat, step_lon = grid_step(lat), grid_step(lon)
    rows = np.rint((lat - lat.min()) / step_lat).astype(int)
    cols = np.rint((lon - lon.min()) / step_lon).astype(int)
    if max(rows.max(), cols.max()) >= 4000:      # a mis-inferred step would blow up memory
        raise ValueError(f"grid {rows.max() + 1} x {cols.max() + 1} from inferred "
                         f"steps {step_lat:.6g}, {step_lon:.6g} deg")

    mask = np.zeros((rows.max() + 1 + 2 * pad, cols.max() + 1 + 2 * pad), bool)
    mask[rows + pad, cols + pad] = in_cluster

    spacing = (step_lat * KM_PER_DEG,
               step_lon * KM_PER_DEG * float(np.cos(np.radians(lat.mean()))))
    return signed_proximity(mask, spacing)[rows + pad, cols + pad]

# --- clustering ---

def to_km(lat, lon):
    """Flat local projection; about 0.5% distance error across one county."""
    x = (lon - lon.mean()) * KM_PER_DEG * np.cos(np.radians(lat.mean()))
    y = (lat - lat.mean()) * KM_PER_DEG
    return np.c_[x, y]


def cluster_one(sub):
    """DBSCAN one county-date. Returns s = None when it finds no cluster."""
    lat, lon = sub[LAT_COL].to_numpy(), sub[LON_COL].to_numpy()
    weight = sub[PROB_COL].clip(0, 1).to_numpy()

    # sample_weight makes min_samples a probability mass, not a neighbour count:
    # a core pixel needs MIN_WEIGHT of predicted probability within EPS_KM.
    labels = DBSCAN(eps=EPS_KM, min_samples=MIN_WEIGHT).fit(
        to_km(lat, lon), sample_weight=weight).labels_
    in_cluster = labels >= 0
    if not in_cluster.any():
        return None, 0, 0

    return proximity_of_pixels(lat, lon, in_cluster), int(labels.max() + 1), int(in_cluster.sum())


def build_table(df):
    pieces, counts, no_customers, no_cluster = [], [], 0, 0
    groups = df.groupby(GROUP_COL, sort=True)
    print(f"clustering {groups.ngroups:,} county-dates")

    for n, (bag, sel) in enumerate(groups, 1):
        # one row per location; repeated lat/lon rows are averaged
        sub = sel.groupby([LAT_COL, LON_COL], as_index=False)[[PROB_COL, CUSTOMER_COL]].mean()
        if not sub[CUSTOMER_COL].fillna(0).sum():   # stage 2 needs a positive bag weight
            no_customers += 1
            continue

        s, n_clusters, n_member = cluster_one(sub)
        if s is None:          # no cluster means no R, so s is undefined here
            no_cluster += 1
            continue

        sub["cluster_proximity"] = s
        sub[GROUP_COL] = bag
        sub["truth"] = float(sel[TARGET_COL].iloc[0])
        if "Date" in sel:
            sub["Date"] = sel["Date"].iloc[0]

        pieces.append(sub)
        counts.append({GROUP_COL: bag, "n_lit_pixels": len(sub),
                       "n_clusters": n_clusters, "n_member": n_member,
                       "n_noise": len(sub) - n_member})

        if n % 200 == 0:
            print(f"  {n:,}/{groups.ngroups:,} county-dates clustered", flush=True)

    if not pieces:
        raise SystemExit("no county-dates passed the filters")

    table = pd.concat(pieces, ignore_index=True)
    p = table[PROB_COL].clip(PROB_EPS, 1 - PROB_EPS)
    table["pixel_logit"] = np.log(p / (1 - p))   # frozen input to stage 2

    print(f"kept {table[GROUP_COL].nunique():,} county-dates; dropped {no_customers} "
          f"with no customers and {no_cluster} with no cluster")
    print(f"proximity: range [{table.cluster_proximity.min():.2f}, "
          f"{table.cluster_proximity.max():.2f}], "
          f"{(table.cluster_proximity > 0).mean():.1%} of pixels inside a cluster")
    return table, pd.DataFrame(counts)

# --- entry point ---

def main():
    parser = argparse.ArgumentParser(description="DBSCAN clusters and pixel proximity")
    parser.add_argument("--pred", type=Path, default=DEFAULT_PRED,
                        help="pixel predictions written by stage 1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="directory for pixels_with_proximity.csv")
    args, _ = parser.parse_known_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"pred: {args.pred}")
    print(f"out:  {args.out}")

    table, counts = build_table(load_data(args.pred))

    table.to_csv(args.out / "pixels_with_proximity.csv", index=False)
    counts.to_csv(args.out / "cluster_counts.csv", index=False)
    with open(args.out / "clustering_config.json", "w") as handle:
        json.dump({"eps_km": EPS_KM, "min_weight": MIN_WEIGHT,
                   "source": str(args.pred)}, handle, indent=2)

    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()