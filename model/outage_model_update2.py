import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# --- Settings ---

DATA_CSV = Path("/home/mdmehedi.hasan/Power Outage Modeling/training_data/combined_all_20_storms_post_landfall.csv")
OUT_DIR = Path("/home/mdmehedi.hasan/Power Outage Modeling/outage_model_update2_20events_post_dates")

FEATURE_COLS = ["log_comparability_score", "ntl_mean_pre", "ntl_sd_pre",
                "ntl_skew_pre", "building_count"]
CUSTOMER_COL = "customers_per_pixel"
TARGET_COL = "fraction_outage"
GROUP_COL = "county_date"

SEED = 0
N_FOLDS = 5

HIDDEN_LAYERS = 3        # number of hidden layers
HIDDEN_WIDTH = 32        # neurons per hidden layer
LR = 0.003               # learning rate
N_EPOCHS = 1000          # full-batch updates
WEIGHT_DECAY = 0.0       # Adam default

CONFIG = (f"MLP {HIDDEN_LAYERS} x {HIDDEN_WIDTH} (ReLU), sigmoid output | Adam lr={LR}, "
          f"{N_EPOCHS} full-batch updates, weight_decay={WEIGHT_DECAY}")

torch.manual_seed(SEED)
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
torch.use_deterministic_algorithms(True)

# --- Model ---
class OutageModel(nn.Module):
    def __init__(self, n_features, hidden_layers=HIDDEN_LAYERS, hidden_width=HIDDEN_WIDTH):
        super().__init__()
        assert hidden_layers >= 1, "at least one hidden layer"
        sizes = [n_features] + [hidden_width] * hidden_layers
        hidden = [m for a, b in zip(sizes, sizes[1:]) for m in (nn.Linear(a, b), nn.ReLU())]
        self.net = nn.Sequential(*hidden, nn.Linear(hidden_width, 1))

    def pixel_logit(self, X):
        return self.net(X).squeeze(-1)

    def forward(self, bags):
        p = torch.sigmoid(self.pixel_logit(bags.X))
        return torch.zeros(bags.n).index_add(0, bags.idx, bags.w * p)

# --- Data ---
class Bags:
    def __init__(self, df, means, sds):
        df = df.sort_values(GROUP_COL, kind="mergesort").reset_index(drop=True)
        codes, uniques = pd.factorize(df[GROUP_COL], sort=True)
        customers = df[CUSTOMER_COL].to_numpy(float)
        totals = np.bincount(codes, weights=customers, minlength=len(uniques))

        self.df, self.ids, self.n = df, list(uniques), len(uniques)
        self.X = torch.tensor(((df[FEATURE_COLS] - means) / sds).to_numpy(np.float32))
        self.idx = torch.from_numpy(codes.astype(np.int64))
        self.w = torch.tensor(customers / totals[codes], dtype=torch.float32)
        self.y = torch.tensor(df.groupby(GROUP_COL, sort=True)["target"]
                              .first().to_numpy(np.float32))

def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    date_col = "Date" if "Date" in df.columns else "date"
    df["target"] = df[TARGET_COL]
    df[GROUP_COL] = (df["STATEFP"].astype(str).str.zfill(2) + "_"
                     + df["county"].astype(str).str.strip() + "_"
                     + df[date_col].astype(str))
    print(f"{len(df):,} records, {df[GROUP_COL].nunique():,} bags")
    return df

def standardise(df):
    return df[FEATURE_COLS].mean(), df[FEATURE_COLS].std().replace(0, 1.0)

def assign_folds(df, n_folds=N_FOLDS, seed=SEED):
    bags = np.sort(df[GROUP_COL].unique())
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of_bag = {bag: fold
                   for fold, (_, held) in enumerate(splitter.split(bags))
                   for bag in bags[held]}
    return df[GROUP_COL].map(fold_of_bag).to_numpy()

# --- Training ---
def fit(bags, hidden_layers=HIDDEN_LAYERS, hidden_width=HIDDEN_WIDTH, lr=LR,
        n_epochs=N_EPOCHS, weight_decay=WEIGHT_DECAY, seed=SEED):
    torch.manual_seed(seed)
    model = OutageModel(len(FEATURE_COLS), hidden_layers, hidden_width)
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

def evaluate(y, yhat):
    r = np.asarray(yhat, float) - np.asarray(y, float)
    return {"n": len(r), "r2": float(r2_score(y, yhat)),
            "rmse": float(np.sqrt((r ** 2).mean())),
            "mae": float(np.abs(r).mean()), "bias": float(r.mean())}

# --- Cross-validation ---
def cross_validate(df, n_folds=N_FOLDS, seed=SEED, verbose=True, **kwargs):
    folds = assign_folds(df, n_folds, seed)
    rows, observed, predicted, ids = [], [], [], []

    for fold in range(n_folds):
        means, sds = standardise(df[folds != fold])
        train = Bags(df[folds != fold], means, sds)
        valid = Bags(df[folds == fold], means, sds)

        model, _ = fit(train, seed=seed + fold, **kwargs)
        y, yhat = valid.y.numpy(), predict(model, valid)
        metrics = evaluate(y, yhat)
        in_train = evaluate(train.y.numpy(), predict(model, train))

        rows.append({"fold": fold + 1, **metrics, "train_r2": in_train["r2"]})
        if verbose:
            print(f"fold {fold + 1}/{n_folds}: {train.n:,} train / {valid.n:,} valid bags | "
                  f"train R2 = {in_train['r2']:.3f} | held out R2 = {metrics['r2']:.3f}, "
                  f"RMSE = {metrics['rmse']:.3f}, bias = {metrics['bias']:+.3f}")

        observed.append(y)
        predicted.append(yhat)
        ids.extend(valid.ids)

    table = pd.DataFrame(rows)
    observed, predicted = np.concatenate(observed), np.concatenate(predicted)
    pooled = evaluate(observed, predicted)

    if verbose:
        print(f"\nacross folds: R2 = {table['r2'].mean():.3f} (SD {table['r2'].std(ddof=1):.3f}), "
              f"RMSE = {table['rmse'].mean():.3f}")
        print(f"pooled out of fold: R2 = {pooled['r2']:.3f}, RMSE = {pooled['rmse']:.3f}, "
              f"MAE = {pooled['mae']:.3f}, bias = {pooled['bias']:+.3f}")

    return table, pooled, pd.DataFrame({GROUP_COL: ids, "observed": observed,
                                        "predicted": predicted})

# --- Run ---
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"data:  {DATA_CSV}\nout:   {OUT_DIR}\nmodel: {CONFIG}\n")

df = load_data(DATA_CSV)
fold_table, pooled, oof = cross_validate(df)
fold_table.to_csv(OUT_DIR / "cross_validation_folds.csv", index=False)
oof.to_csv(OUT_DIR / "cross_validation_predictions.csv", index=False)

# --- Final model on all bags ---
means, sds = standardise(df)
final = Bags(df, means, sds)
model, history = fit(final, seed=SEED)
in_sample = evaluate(final.y.numpy(), predict(model, final))
print(f"\nfinal model: {final.n:,} bags | in sample R2 = {in_sample['r2']:.3f}, "
      f"RMSE = {in_sample['rmse']:.3f}")

pd.DataFrame({"update": np.arange(1, len(history) + 1), "train_mse": history}).to_csv(
    OUT_DIR / "final_training_curve.csv", index=False)

with torch.no_grad():
    logit = model.pixel_logit(final.X).numpy()
pixels = final.df.copy()
pixels["customer_weight"] = final.w.numpy()
pixels["pixel_logit"] = logit
pixels["pixel_outage_probability"] = 1 / (1 + np.exp(-logit))
pixels.to_csv(OUT_DIR / "final_pixel_predictions.csv", index=False)

# --- Save ---
torch.save(model.state_dict(), OUT_DIR / "final_model.pt")
with open(OUT_DIR / "results.json", "w") as handle:
    json.dump({"config": CONFIG, "seed": SEED, "n_folds": N_FOLDS, "features": FEATURE_COLS,
               "hidden_layers": HIDDEN_LAYERS, "hidden_width": HIDDEN_WIDTH, "lr": LR,
               "n_epochs": N_EPOCHS, "weight_decay": WEIGHT_DECAY,
               "folds": fold_table.to_dict("records"), "pooled_out_of_fold": pooled,
               "in_sample": in_sample}, handle, indent=2)
with open(OUT_DIR / "standardisation.json", "w") as handle:
    json.dump({"features": FEATURE_COLS, "means": means.to_dict(), "sds": sds.to_dict()},
              handle, indent=2)

print(f"saved model, results and standardisation to {OUT_DIR}")