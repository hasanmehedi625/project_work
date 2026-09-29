"""
Score a saved outage-model ensemble on a new CSV of unseen bags.

This does not retrain anything. It loads the ensemble and the standardisation
statistics written by `outage_model.py`'s final-model step, applies that same
standardisation to the new file (never recomputed from it — the test data
must be scaled using the statistics the model was trained with), and writes
two outputs to --out:

  predictions.csv        one row per bag (county-date); predicted outage
                          fraction, and observed + metrics if the file
                          carries a known outage column.
  predictions_pixel.csv  one row per pixel; deviation score and outage
                          probability from pixel_scores(). These are never
                          fitted against a pixel-level label — supervision is
                          bag level only — so treat them as useful for
                          mapping within a bag, not as separately validated.

Usage
-----
    python evaluate.py \
        --model-dir results/baseline_cv/final_model \
        --data new_event_data.csv \
        --out results/test_scores
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from outage_model import (
    FEATURE_COLS, CUSTOMER_COL, GROUP_COL, TARGET_CANDIDATES,
    BagDataset, load_final_model, predict_ensemble, predict_pixel_ensemble, evaluate,
)


def load_unseen(path):
    """Read a test CSV and build bag identifiers, the same way training data is built.

    Unlike `outage_model.load_data`, a known-outage column is optional: this
    file may be genuinely unlabeled, or may carry ground truth kept aside for
    evaluation. Either way, nothing here is filtered or altered.
    """
    df = pd.read_csv(path, low_memory=False)

    date_col = "Date" if "Date" in df.columns else "date"
    df[GROUP_COL] = (df["STATEFP"].astype(str).str.zfill(2) + "_"
                     + df["county"].astype(str).str.strip() + "_"
                     + df[date_col].astype(str))

    target_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    has_target = target_col is not None
    if has_target:
        df["target"] = df[target_col]

    check_unseen(df, has_target)
    print(f"{len(df):,} records, {df[GROUP_COL].nunique():,} bags"
          + ("" if has_target else " (no outage column found — predictions only)"))
    return df, has_target


def check_unseen(df, has_target):
    """Verify the file meets the model's assumptions; raise rather than silently coerce."""
    numeric = FEATURE_COLS + [CUSTOMER_COL] + (["target"] if has_target else [])
    for column in numeric:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{column}: {int((~np.isfinite(values)).sum())} non-finite values")

    if has_target:
        target = df["target"].to_numpy(np.float64)
        if target.min() < 0.0 or target.max() > 1.0:
            raise ValueError(f"target outside [0, 1]: [{target.min():.4g}, {target.max():.4g}]")
        if (df.groupby(GROUP_COL)["target"].nunique() > 1).any():
            raise ValueError("target varies within at least one bag")

    if not (df.groupby(GROUP_COL)[CUSTOMER_COL].sum() > 0).all():
        raise ValueError("at least one bag has a non-positive customer total")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="the final_model/ directory saved by outage_model.py")
    parser.add_argument("--data", type=Path, required=True,
                        help="unseen test CSV, same columns as training data")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory for predictions.csv (and scores.json if labeled)")
    args, _ = parser.parse_known_args()  # tolerate being run from a notebook kernel

    args.out.mkdir(parents=True, exist_ok=True)

    members, means, sds = load_final_model(args.model_dir)
    print(f"loaded {len(members)}-member ensemble from {args.model_dir}")

    df, has_target = load_unseen(args.data)
    if not has_target:
        df["target"] = 0.0  # placeholder; BagDataset needs the column, never used for scoring
    data = BagDataset(df, means, sds)

    prediction = predict_ensemble(members, data)
    out = pd.DataFrame({GROUP_COL: data.bag_ids, "predicted": prediction})

    if has_target:
        out["observed"] = data.y.numpy()
        metrics = evaluate(data.y.numpy(), prediction)
        print(f"R2 = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}, "
              f"MAE = {metrics['mae']:.3f}, bias = {metrics['bias']:+.3f}")
        import json
        with open(args.out / "scores.json", "w") as handle:
            json.dump(metrics, handle, indent=2)

    out.to_csv(args.out / "predictions.csv", index=False)
    print(f"wrote {args.out / 'predictions.csv'}")

    # Pixel-level output. These probabilities were never fitted against a
    # pixel-level label (see outage_model.predict_pixel_ensemble) — treat
    # them as useful for within-bag mapping, not as a separately validated
    # prediction.
    d_pixel, p_pixel = predict_pixel_ensemble(members, data)
    pixel_out = data.df.copy()
    if not has_target:
        pixel_out = pixel_out.drop(columns=["target"])  # the placeholder added above
    pixel_out["deviation_score"] = d_pixel
    pixel_out["predicted_prob"] = p_pixel
    pixel_out["customer_weight_in_bag"] = data.weights.numpy()
    pixel_out.to_csv(args.out / "predictions_pixel.csv", index=False)
    print(f"wrote {args.out / 'predictions_pixel.csv'} ({len(pixel_out):,} pixels)")


if __name__ == "__main__":
    main()
