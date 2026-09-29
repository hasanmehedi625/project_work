"""
Evaluate a saved outage model on new, unseen bags.

This script does not retrain the model.

It loads the saved model and the training standardisation statistics,
then applies the same standardisation to the new data.

Outputs
-------
predictions.csv
    One row per county-date bag with the predicted outage fraction.
    If observed outage data are available, metrics are also included.

predictions_pixel.csv
    One row per pixel with its deviation score and outage probability.
    These pixel-level results are not directly validated because the
    model was trained using bag-level outage labels.

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
    args, _ = parser.parse_known_args()  
    args.out.mkdir(parents=True, exist_ok=True)

    members, means, sds = load_final_model(args.model_dir)
    print(f"loaded {len(members)}-member ensemble from {args.model_dir}")

    df, has_target = load_unseen(args.data)
    if not has_target:
        df["target"] = 0.0  
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

    d_pixel, p_pixel = predict_pixel_ensemble(members, data)
    pixel_out = data.df.copy()
    if not has_target:
        pixel_out = pixel_out.drop(columns=["target"])  
    pixel_out["deviation_score"] = d_pixel
    pixel_out["predicted_prob"] = p_pixel
    pixel_out["customer_weight_in_bag"] = data.weights.numpy()
    pixel_out.to_csv(args.out / "predictions_pixel.csv", index=False)
    print(f"wrote {args.out / 'predictions_pixel.csv'} ({len(pixel_out):,} pixels)")


if __name__ == "__main__":
    main()
