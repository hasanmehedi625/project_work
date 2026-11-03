import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# --- Settings (defaults; overridable on the command line) ---

STAGE1_CSV = Path("/home/mdmehedi.hasan/Power Outage Modeling/outage_model_update2_non_event_test/predictions_pixel.csv")
PROXIMITY_CSV = Path("/home/mdmehedi.hasan/Power Outage Modeling/clustering_update2_non_event/pixels_with_proximity.csv")
OUT_DIR = Path("/home/mdmehedi.hasan/Power Outage Modeling/stage2_update2_non_event")

STAGE2_FEATURE_COLS = ["cluster_proximity"]   # inputs to g(s)
LOGIT_COL = "pixel_logit"                     # f(x) from stage 1, frozen
CUSTOMER_COL = "customers_per_pixel"
TARGET_COL = "fraction_outage"
GROUP_COL = "county_date"

# Columns identifying a pixel-date in BOTH csv files, e.g. ["pixel_id"].
# None -> auto-detect from KEY_CANDIDATES.
MERGE_KEY = None
DEDUPE_KEY = False        # drop duplicate key rows on the proximity side before merging
KEY_CANDIDATES = [        # matched case-insensitively against both files
    ["county_date", "pixel_id"], ["pixel_id", "date"], ["pixel_id"],
    ["county_date", "pixel"], ["pixel", "date"],
    ["county_date", "grid_id"], ["grid_id", "date"],
    ["county_date", "cell_id"], ["cell_id", "date"],
    ["county_date", "latitude", "longitude"], ["latitude", "longitude", "date"],
    ["county_date", "lat", "lon"], ["lat", "lon", "date"],
    ["county_date", "x", "y"], ["x", "y", "date"],
]

SEED = 0
N_FOLDS = 5

HIDDEN_LAYERS = 2        # hidden layers in g(s)
HIDDEN_WIDTH = 16        # neurons per hidden layer
LR = 0.003
N_EPOCHS = 1000
WEIGHT_DECAY = 0.0
ZERO_INIT_OUTPUT = True  # start g(s) = 0 so training begins exactly at the stage 1 fit

# --- Model: g(s) only, f(x) enters as a frozen offset ---
class ProximityModel(nn.Module):
    def __init__(self, n_features, hidden_layers=None, hidden_width=None,
                 zero_init_output=None):
        super().__init__()
        hidden_layers = HIDDEN_LAYERS if hidden_layers is None else hidden_layers
        hidden_width = HIDDEN_WIDTH if hidden_width is None else hidden_width
        zero_init_output = ZERO_INIT_OUTPUT if zero_init_output is None else zero_init_output
        assert hidden_layers >= 1, "at least one hidden layer"
        sizes = [n_features] + [hidden_width] * hidden_layers
        hidden = [m for a, b in zip(sizes, sizes[1:]) for m in (nn.Linear(a, b), nn.ReLU())]
        output = nn.Linear(hidden_width, 1)
        if zero_init_output:
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        self.net = nn.Sequential(*hidden, output)

    def pixel_adjustment(self, S):
        return self.net(S).squeeze(-1)          # g(s)

    def pixel_logit(self, bags):
        return bags.f + self.pixel_adjustment(bags.S)   # f(x) + g(s)

    def forward(self, bags):
        p = torch.sigmoid(self.pixel_logit(bags))
        return torch.zeros(bags.n).index_add(0, bags.idx, bags.w * p)

# --- Data ---
class Bags:
    def __init__(self, df, means, sds):
        df = df.sort_values(GROUP_COL, kind="mergesort").reset_index(drop=True)
        codes, uniques = pd.factorize(df[GROUP_COL], sort=True)
        customers = df[CUSTOMER_COL].to_numpy(float)
        totals = np.bincount(codes, weights=customers, minlength=len(uniques))

        self.df, self.ids, self.n = df, list(uniques), len(uniques)
        self.S = torch.tensor(((df[STAGE2_FEATURE_COLS] - means) / sds).to_numpy(np.float32))
        self.f = torch.tensor(df[LOGIT_COL].to_numpy(np.float32))   # frozen stage 1 logit
        self.idx = torch.from_numpy(codes.astype(np.int64))
        self.w = torch.tensor(customers / totals[codes], dtype=torch.float32)
        self.y = torch.tensor(df.groupby(GROUP_COL, sort=True)["target"]
                              .first().to_numpy(np.float32))

def normalise_date(df):
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    return df

def resolve_columns(key, frame):
    """Match requested column names against a frame, ignoring case. None if any are absent."""
    lookup = {c.lower(): c for c in frame.columns}
    if any(c.lower() not in lookup for c in key):
        return None
    return [lookup[c.lower()] for c in key]

def find_merge_key(left, right):
    if MERGE_KEY is not None:
        left_key, right_key = resolve_columns(MERGE_KEY, left), resolve_columns(MERGE_KEY, right)
        if left_key is None or right_key is None:
            raise SystemExit(
                f"merge key {MERGE_KEY} is not present in both files.\n"
                f"stage 1 columns:   {sorted(left.columns)}\n"
                f"proximity columns: {sorted(right.columns)}")
        dup_left, dup_right = left.duplicated(left_key).sum(), right.duplicated(right_key).sum()
        if (dup_left or dup_right) and not DEDUPE_KEY:
            raise SystemExit(
                f"merge key {MERGE_KEY} is not unique: {dup_left:,} duplicate rows in the "
                f"stage 1 file, {dup_right:,} in the proximity file. Add a column that "
                f"separates them, or pass --dedupe-key to keep the first proximity row "
                f"per key.")
        return left_key, right_key

    rejected = []
    for key in KEY_CANDIDATES:
        left_key, right_key = resolve_columns(key, left), resolve_columns(key, right)
        if left_key is None or right_key is None:
            continue
        dup_left, dup_right = left.duplicated(left_key).sum(), right.duplicated(right_key).sum()
        if not dup_left and not dup_right:
            return left_key, right_key
        rejected.append(f"  {left_key}: {dup_left:,} duplicate rows in stage 1, "
                        f"{dup_right:,} in proximity")

    message = ["could not auto-detect a unique merge key. Pass --merge-key col1,col2."]
    if rejected:
        message.append("these shared keys exist but are not unique:")
        message.extend(rejected)
        message.append("pass one of them with --dedupe-key, or add a separating column.")
    else:
        shared = sorted(set(left.columns) & set(right.columns))
        message.append(f"columns shared by both files: {shared}")
    message.append(f"stage 1 columns:   {sorted(left.columns)}")
    message.append(f"proximity columns: {sorted(right.columns)}")
    raise SystemExit("\n".join(message))

def load_data(stage1_path, proximity_path):
    stage1 = normalise_date(pd.read_csv(stage1_path, low_memory=False))
    proximity = normalise_date(pd.read_csv(proximity_path, low_memory=False))

    for col in [LOGIT_COL, CUSTOMER_COL, TARGET_COL, GROUP_COL]:
        assert col in stage1.columns, f"'{col}' not in {stage1_path.name}"
    for col in STAGE2_FEATURE_COLS:
        assert col in proximity.columns, f"'{col}' not in {proximity_path.name}"

    left_key, right_key = find_merge_key(stage1, proximity)
    print(f"merging on {left_key}" + (f" <- {right_key}" if right_key != left_key else "")
          + (" (requested on the command line)" if MERGE_KEY is not None else " (auto-detected)"))

    right = proximity[right_key + STAGE2_FEATURE_COLS]
    if DEDUPE_KEY:
        before = len(right)
        right = right.drop_duplicates(right_key)
        if len(right) < before:
            print(f"dropped {before - len(right):,} duplicate key rows from the proximity file")

    df = stage1.merge(right, how="left", left_on=left_key, right_on=right_key,
                      validate="many_to_one" if DEDUPE_KEY else "one_to_one",
                      suffixes=("", "_proximity"))

    missing = df[STAGE2_FEATURE_COLS].isna().any(axis=1).sum()
    if missing:
        fill = df[STAGE2_FEATURE_COLS].median()
        print(f"warning: {missing:,} pixels have no proximity value, filled with median "
              f"{fill.to_dict()}")
        df[STAGE2_FEATURE_COLS] = df[STAGE2_FEATURE_COLS].fillna(fill)

    df["target"] = df[TARGET_COL]
    print(f"{len(df):,} records, {df[GROUP_COL].nunique():,} bags")
    return df

def standardise(df):
    return df[STAGE2_FEATURE_COLS].mean(), df[STAGE2_FEATURE_COLS].std().replace(0, 1.0)

def assign_folds(df, n_folds=None, seed=None):
    n_folds = N_FOLDS if n_folds is None else n_folds
    seed = SEED if seed is None else seed
    bags = np.sort(df[GROUP_COL].unique())
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of_bag = {bag: fold
                   for fold, (_, held) in enumerate(splitter.split(bags))
                   for bag in bags[held]}
    return df[GROUP_COL].map(fold_of_bag).to_numpy()

# --- Training ---
def fit(bags, hidden_layers=None, hidden_width=None, lr=None,
        n_epochs=None, weight_decay=None, seed=None):
    hidden_layers = HIDDEN_LAYERS if hidden_layers is None else hidden_layers
    hidden_width = HIDDEN_WIDTH if hidden_width is None else hidden_width
    lr = LR if lr is None else lr
    n_epochs = N_EPOCHS if n_epochs is None else n_epochs
    weight_decay = WEIGHT_DECAY if weight_decay is None else weight_decay
    seed = SEED if seed is None else seed

    torch.manual_seed(seed)
    model = ProximityModel(len(STAGE2_FEATURE_COLS), hidden_layers, hidden_width)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    for _ in range(n_epochs):
        optimiser.zero_grad()
        loss = ((model(bags) - bags.y) ** 2).mean()
        loss.backward()
        optimiser.step()
        history.append(float(loss.detach()))

    assert np.isfinite(history).all(), f"training diverged at lr={lr}"
    model.eval()
    return model, history

def predict(model, bags):
    with torch.no_grad():
        return model(bags).numpy()

def predict_stage1(bags):
    """County prediction using f(x) alone, i.e. g(s) = 0."""
    with torch.no_grad():
        p = torch.sigmoid(bags.f)
        return torch.zeros(bags.n).index_add(0, bags.idx, bags.w * p).numpy()

def evaluate(y, yhat):
    r = np.asarray(yhat, float) - np.asarray(y, float)
    return {"n": len(r), "r2": float(r2_score(y, yhat)),
            "rmse": float(np.sqrt((r ** 2).mean())),
            "mae": float(np.abs(r).mean()), "bias": float(r.mean())}

# --- Cross-validation ---
def cross_validate(df, n_folds=None, seed=None, verbose=True, **kwargs):
    n_folds = N_FOLDS if n_folds is None else n_folds
    seed = SEED if seed is None else seed
    folds = assign_folds(df, n_folds, seed)
    rows, observed, predicted, baseline, ids = [], [], [], [], []

    for fold in range(n_folds):
        means, sds = standardise(df[folds != fold])
        train = Bags(df[folds != fold], means, sds)
        valid = Bags(df[folds == fold], means, sds)

        model, _ = fit(train, seed=seed + fold, **kwargs)
        y, yhat = valid.y.numpy(), predict(model, valid)
        base = predict_stage1(valid)
        metrics = evaluate(y, yhat)
        in_train = evaluate(train.y.numpy(), predict(model, train))
        base_metrics = evaluate(y, base)

        rows.append({"fold": fold + 1, **metrics, "train_r2": in_train["r2"],
                     "stage1_only_r2": base_metrics["r2"]})
        if verbose:
            print(f"fold {fold + 1}/{n_folds}: {train.n:,} train / {valid.n:,} valid bags | "
                  f"train R2 = {in_train['r2']:.3f} | held out R2 = {metrics['r2']:.3f} "
                  f"(stage 1 alone {base_metrics['r2']:.3f}), "
                  f"RMSE = {metrics['rmse']:.3f}, bias = {metrics['bias']:+.3f}")

        observed.append(y)
        predicted.append(yhat)
        baseline.append(base)
        ids.extend(valid.ids)

    table = pd.DataFrame(rows)
    observed = np.concatenate(observed)
    predicted = np.concatenate(predicted)
    baseline = np.concatenate(baseline)
    pooled = evaluate(observed, predicted)
    pooled_stage1 = evaluate(observed, baseline)

    if verbose:
        print(f"\nacross folds: R2 = {table['r2'].mean():.3f} (SD {table['r2'].std(ddof=1):.3f}), "
              f"RMSE = {table['rmse'].mean():.3f}")
        print(f"pooled out of fold: R2 = {pooled['r2']:.3f}, RMSE = {pooled['rmse']:.3f}, "
              f"MAE = {pooled['mae']:.3f}, bias = {pooled['bias']:+.3f}")
        print(f"pooled stage 1 alone: R2 = {pooled_stage1['r2']:.3f}, "
              f"RMSE = {pooled_stage1['rmse']:.3f}")

    return table, pooled, pooled_stage1, pd.DataFrame(
        {GROUP_COL: ids, "observed": observed, "predicted": predicted,
         "predicted_stage1_only": baseline})

# --- Command line ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: frozen stage-1 pixel logit f(x) plus a neural network g(s) "
                    "on cluster proximity, combined as sigmoid(f(x) + g(s)).")
    parser.add_argument("--stage1", type=Path, default=STAGE1_CSV,
                        help="stage 1 final_pixel_predictions.csv (supplies pixel_logit)")
    parser.add_argument("--pixels", type=Path, default=PROXIMITY_CSV,
                        help="clustering pixels_with_proximity.csv (supplies cluster_proximity)")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    parser.add_argument("--merge-key", default=None,
                        help="comma separated columns joining the two files, matched "
                             "case-insensitively, e.g. county_date,Latitude,Longitude "
                             "(default: auto-detect)")
    parser.add_argument("--dedupe-key", action="store_true",
                        help="if the merge key is not unique, keep the first proximity "
                             "row per key instead of failing")
    parser.add_argument("--features", default=",".join(STAGE2_FEATURE_COLS),
                        help="comma separated stage 2 feature columns")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--hidden-layers", type=int, default=HIDDEN_LAYERS)
    parser.add_argument("--hidden-width", type=int, default=HIDDEN_WIDTH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--random-init-output", action="store_true",
                        help="use the default random output layer instead of zero init")
    return parser.parse_args()

def main():
    global STAGE1_CSV, PROXIMITY_CSV, OUT_DIR, STAGE2_FEATURE_COLS, MERGE_KEY, DEDUPE_KEY
    global SEED, N_FOLDS, HIDDEN_LAYERS, HIDDEN_WIDTH, LR, N_EPOCHS, WEIGHT_DECAY
    global ZERO_INIT_OUTPUT

    args = parse_args()
    STAGE1_CSV, PROXIMITY_CSV, OUT_DIR = args.stage1, args.pixels, args.out
    STAGE2_FEATURE_COLS = [c.strip() for c in args.features.split(",") if c.strip()]
    MERGE_KEY = ([c.strip() for c in args.merge_key.split(",") if c.strip()]
                 if args.merge_key else None)
    DEDUPE_KEY = args.dedupe_key
    SEED, N_FOLDS = args.seed, args.folds
    HIDDEN_LAYERS, HIDDEN_WIDTH = args.hidden_layers, args.hidden_width
    LR, N_EPOCHS, WEIGHT_DECAY = args.lr, args.epochs, args.weight_decay
    ZERO_INIT_OUTPUT = not args.random_init_output

    config = (f"stage 2: frozen f(x) from stage 1 + g(s) MLP {HIDDEN_LAYERS} x {HIDDEN_WIDTH} "
              f"(ReLU) on {STAGE2_FEATURE_COLS}, sigmoid(f+g) | Adam lr={LR}, "
              f"{N_EPOCHS} full-batch updates, weight_decay={WEIGHT_DECAY}, "
              f"zero_init_output={ZERO_INIT_OUTPUT}")

    torch.manual_seed(SEED)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    torch.use_deterministic_algorithms(True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"stage 1:   {STAGE1_CSV}\nproximity: {PROXIMITY_CSV}\nout:       {OUT_DIR}\n"
          f"model:     {config}\n")

    df = load_data(STAGE1_CSV, PROXIMITY_CSV)
    fold_table, pooled, pooled_stage1, oof = cross_validate(df)
    fold_table.to_csv(OUT_DIR / "cross_validation_folds.csv", index=False)
    oof.to_csv(OUT_DIR / "cross_validation_predictions.csv", index=False)

    # --- Final model on all bags ---
    means, sds = standardise(df)
    final = Bags(df, means, sds)
    model, history = fit(final, seed=SEED)
    in_sample = evaluate(final.y.numpy(), predict(model, final))
    in_sample_stage1 = evaluate(final.y.numpy(), predict_stage1(final))
    print(f"\nfinal model: {final.n:,} bags | in sample R2 = {in_sample['r2']:.3f}, "
          f"RMSE = {in_sample['rmse']:.3f} (stage 1 alone R2 = {in_sample_stage1['r2']:.3f})")

    pd.DataFrame({"update": np.arange(1, len(history) + 1), "train_mse": history}).to_csv(
        OUT_DIR / "final_training_curve.csv", index=False)

    with torch.no_grad():
        g = model.pixel_adjustment(final.S).numpy()
    logit = final.f.numpy() + g
    pixels = final.df.copy()
    pixels["customer_weight"] = final.w.numpy()
    pixels["pixel_logit_stage1"] = final.f.numpy()
    pixels["pixel_proximity_adjustment"] = g
    pixels["pixel_logit_combined"] = logit
    pixels["pixel_outage_probability"] = 1 / (1 + np.exp(-logit))
    pixels.to_csv(OUT_DIR / "final_pixel_predictions_stage2.csv", index=False)

    # --- Save ---
    torch.save(model.state_dict(), OUT_DIR / "final_model_stage2.pt")
    with open(OUT_DIR / "results.json", "w") as handle:
        json.dump({"config": config, "seed": SEED, "n_folds": N_FOLDS,
                   "stage1_csv": str(STAGE1_CSV), "proximity_csv": str(PROXIMITY_CSV),
                   "stage2_features": STAGE2_FEATURE_COLS, "frozen_logit": LOGIT_COL,
                   "merge_key": MERGE_KEY,
                   "hidden_layers": HIDDEN_LAYERS, "hidden_width": HIDDEN_WIDTH, "lr": LR,
                   "n_epochs": N_EPOCHS, "weight_decay": WEIGHT_DECAY,
                   "zero_init_output": ZERO_INIT_OUTPUT,
                   "folds": fold_table.to_dict("records"), "pooled_out_of_fold": pooled,
                   "pooled_out_of_fold_stage1_only": pooled_stage1,
                   "in_sample": in_sample, "in_sample_stage1_only": in_sample_stage1},
                  handle, indent=2)
    with open(OUT_DIR / "standardisation.json", "w") as handle:
        json.dump({"features": STAGE2_FEATURE_COLS, "means": means.to_dict(),
                   "sds": sds.to_dict()}, handle, indent=2)

    print(f"saved model, results and standardisation to {OUT_DIR}")

if __name__ == "__main__":
    main()