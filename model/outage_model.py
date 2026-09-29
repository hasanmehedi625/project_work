"""
Bag-level outage model: an MLP deviation score with a shared logistic response.

Model
-----
For pixel j with feature vector x_j and customer count c_j:

    d_j     = f_phi(x_j)                     deviation score (MLP)
    p_j     = sigmoid(a * d_j + b)           shared logistic response
    w_ij    = c_j / sum_{k in bag i} c_k     customer share within a bag
    y_hat_i = sum_{j in bag i} w_ij * p_j    bag-level prediction

The network parameters phi and the response parameters a and b are estimated
jointly by minimising the mean squared error between y_hat_i and the observed
outage fraction y_i at the county-date (bag) level. Pixel-level outage labels
are never observed; supervision is bag-level only.

Evaluation
----------
Five-fold cross-validation with folds assigned at the bag level, so pixels from
the same bag never appear in both the training and validation partitions.
Within each fold, feature standardisation statistics and the early-stopping
split are derived from training bags only, so each bag is scored exactly once
by a model estimated without access to it.

Each fold fits an ensemble of independently initialised networks and averages
their bag-level predictions; the across-initialisation spread is reported
alongside the ensemble score. A final ensemble is refitted on the complete
dataset to obtain the reported response-curve parameters. Its in-sample fit is
reported for completeness and is not a generalisation estimate.

Usage
-----
    python outage_model.py --data data/15events_post_event_dates.csv \
                           --out results/baseline_cv
"""

import argparse
import copy
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# --- configuration ---

DEFAULT_DATA = Path("data/15events_post_event_dates_clean.csv")
DEFAULT_OUT = Path("results/baseline_cv")

FEATURE_COLS = ["log_comparability_score", "ntl_mean_pre", "ntl_sd_pre",
                "ntl_skew_pre", "building_count"]
CUSTOMER_COL = "customers_per_pixel"
TARGET_CANDIDATES = ("percent_outage", "fraction_outage")
GROUP_COL = "county_date"

N_FOLDS = 5
N_MEMBERS = 10                      # ensemble members per fold
SEED = 0

HIDDEN_DIM = 8
N_EPOCHS = 300
MIN_EPOCHS = 50
LR = 0.05
WEIGHT_DECAY = 1e-3                 # applied to the MLP only, not to a and b
PATIENCE = 20                       # early-stopping patience, epochs
SCHED_PATIENCE = 8                  # learning-rate schedule patience, epochs
ES_FRACTION = 0.15                  # share of training bags held out for early stopping
MIN_PRED_SD = 0.05                  # below this a fit is treated as degenerate
MAX_REFITS = 3


# --- model ---

class OutageModel(nn.Module):
    """Pixel deviation score, shared logistic response, customer-share pooling."""

    def __init__(self, n_features, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.deviation = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.a = nn.Parameter(torch.tensor(-1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def pixel_scores(self, X):
        """Return the deviation scores d_j and outage probabilities p_j."""
        d = self.deviation(X).squeeze(-1)
        return d, torch.sigmoid(self.a * d + self.b)

    def forward(self, data):
        # The weights sum to one within each bag and p_j lies in (0, 1), so the
        # pooled prediction is a convex combination and needs no clipping.
        _, p = self.pixel_scores(data.X)
        return torch.zeros(data.n_bags).index_add_(0, data.bag_idx, data.weights * p)

    def curve(self):
        """Return (a, b, d50), where d50 = -b / a is the deviation score at
        which the modelled outage probability equals 0.5."""
        a, b = float(self.a), float(self.b)
        return a, b, (-b / a if abs(a) > 1e-6 else float("nan"))


# --- data preparation ---

def load_data(path):
    """Read the pixel table and construct bag identifiers.

    No records are excluded and no values are altered: every pixel in the input
    file enters the model as supplied. `check_data` verifies that the file meets
    the assumptions the estimator relies on and raises if it does not.
    """
    df = pd.read_csv(path, low_memory=False)

    target_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target_col is None:
        raise KeyError(f"none of {TARGET_CANDIDATES} present in {path}")
    date_col = "Date" if "Date" in df.columns else "date"

    df["target"] = df[target_col]
    df[GROUP_COL] = (df["STATEFP"].astype(str).str.zfill(2) + "_"
                     + df["county"].astype(str).str.strip() + "_"
                     + df[date_col].astype(str))

    check_data(df)
    print(f"{len(df):,} records, {df[GROUP_COL].nunique():,} bags")
    return df


def check_data(df):
    """Verify the input, raising on any violation. Nothing is modified.

    The estimator assumes finite predictors, an outage fraction in [0, 1] that
    is constant within a bag, and a positive customer total per bag. These are
    properties of the input file, not choices made here.
    """
    numeric = FEATURE_COLS + [CUSTOMER_COL, "target"]
    for column in numeric:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{column}: {int((~np.isfinite(values)).sum())} non-finite values")

    target = df["target"].to_numpy(np.float64)
    if target.min() < 0.0 or target.max() > 1.0:
        raise ValueError(f"target outside [0, 1]: [{target.min():.4g}, {target.max():.4g}]")

    grouped = df.groupby(GROUP_COL)
    if (grouped["target"].nunique() > 1).any():
        raise ValueError("target varies within at least one bag")
    if not (grouped[CUSTOMER_COL].sum() > 0).all():
        raise ValueError("at least one bag has a non-positive customer total")


class BagDataset:
    """Tensor view of one partition.

    Standardisation is a reparameterisation of the predictors, not a change to
    the data: the statistics are supplied by the caller and must be derived from
    training bags only. Customer shares are precomputed because they are fixed
    by the data and do not depend on the model parameters.
    """

    def __init__(self, df, means, sds):
        df = df.sort_values(GROUP_COL, kind="mergesort").reset_index(drop=True)
        self.df = df  # sorted, row-aligned with X/weights — lets pixel-level
                       # predictions be matched back to their source rows

        self.X = torch.tensor(((df[FEATURE_COLS] - means) / sds).to_numpy(np.float32))

        codes, uniques = pd.factorize(df[GROUP_COL], sort=True)
        self.bag_ids = list(uniques)
        self.n_bags = len(uniques)
        self.bag_idx = torch.from_numpy(codes.astype(np.int64))

        customers = df[CUSTOMER_COL].to_numpy(np.float64)
        totals = np.bincount(codes, weights=customers, minlength=self.n_bags)
        self.weights = torch.tensor(customers / totals[codes], dtype=torch.float32)

        self.y = torch.tensor(
            df.groupby(GROUP_COL, sort=True)["target"].first().to_numpy(np.float32)
        )


def standardisation_stats(df):
    """Feature means and standard deviations; zero-variance features stay unscaled."""
    return df[FEATURE_COLS].mean(), df[FEATURE_COLS].std().replace(0, 1.0)


def assign_folds(df, n_folds, seed):
    """Partition bags, not pixel records, into folds."""
    bags = np.sort(df[GROUP_COL].unique())
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of_bag = {bag: fold
                   for fold, (_, held_out) in enumerate(splitter.split(bags))
                   for bag in bags[held_out]}
    return df[GROUP_COL].map(fold_of_bag).to_numpy()


# --- estimation ---

def fit_single(data, seed):
    """Fit one model by full-batch Adam with early stopping.

    The early-stopping subset is drawn from the bags supplied in `data`, which
    during cross-validation contains training bags only.
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    n_holdout = max(2, int(ES_FRACTION * data.n_bags))
    is_holdout = torch.zeros(data.n_bags, dtype=torch.bool)
    is_holdout[torch.from_numpy(rng.permutation(data.n_bags)[:n_holdout])] = True
    is_fit = ~is_holdout

    model = OutageModel(len(FEATURE_COLS), hidden_dim=HIDDEN_DIM)
    optimiser = torch.optim.Adam(
        [{"params": model.deviation.parameters(), "weight_decay": WEIGHT_DECAY},
         {"params": [model.a, model.b], "weight_decay": 0.0}],
        lr=LR,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=SCHED_PATIENCE, factor=0.5
    )

    best_loss, best_state, remaining = float("inf"), None, PATIENCE

    for epoch in range(N_EPOCHS):
        model.train()
        optimiser.zero_grad()
        loss = ((model(data)[is_fit] - data.y[is_fit]) ** 2).mean()
        loss.backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            holdout_loss = float(((model(data)[is_holdout] - data.y[is_holdout]) ** 2).mean())
        if not np.isfinite(holdout_loss):
            break
        scheduler.step(holdout_loss)

        if holdout_loss < best_loss - 1e-5:
            best_loss, remaining = holdout_loss, PATIENCE
            best_state = copy.deepcopy(model.state_dict())
        else:
            remaining -= 1
            if remaining <= 0 and epoch + 1 >= MIN_EPOCHS:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def fit_ensemble(data, n_members, label):
    """Fit `n_members` independently initialised models on `data`.

    A fit is repeated with a shifted initialisation if its training predictions
    are degenerate (standard deviation below MIN_PRED_SD). The criterion uses
    training predictions only and never inspects held-out bags; the number of
    repetitions is returned so that it can be reported.
    """
    members, diagnostics, n_refits = [], [], 0

    for member in range(n_members):
        for attempt in range(MAX_REFITS):
            model = fit_single(data, SEED + member + 1000 * attempt)
            with torch.no_grad():
                pred = model(data).numpy()
            if np.isfinite(pred).all() and pred.std() >= MIN_PRED_SD:
                break
            n_refits += 1

        a, b, d50 = model.curve()
        diagnostics.append({"member": member, "a": a, "b": b, "d50": d50,
                            **evaluate(data.y.numpy(), pred)})
        members.append(model)

    print(f"  {label}: {n_members} members fitted ({n_refits} reinitialisations)")
    return members, pd.DataFrame(diagnostics), n_refits


def predict_ensemble(members, data):
    with torch.no_grad():
        return np.column_stack([m(data).numpy() for m in members]).mean(axis=1)


def predict_pixel_ensemble(members, data):
    """Per-pixel deviation score and outage probability, averaged over the ensemble.

    These were never fitted against a pixel-level label — supervision is bag
    level only (see module docstring) — so treat them as a byproduct of the
    model's internal structure, useful for mapping within a bag, not as a
    separately validated prediction.
    """
    with torch.no_grad():
        scores = [m.pixel_scores(data.X) for m in members]
        d = np.column_stack([s[0].numpy() for s in scores]).mean(axis=1)
        p = np.column_stack([s[1].numpy() for s in scores]).mean(axis=1)
    return d, p


def evaluate(observed, predicted):
    observed = np.asarray(observed, np.float64)
    predicted = np.asarray(predicted, np.float64)
    residual = predicted - observed
    return {
        "n": int(len(observed)),
        "r2": float(r2_score(observed, predicted)) if len(observed) >= 2 else np.nan,
        "rmse": float(np.sqrt((residual ** 2).mean())),
        "mae": float(np.abs(residual).mean()),
        "bias": float(residual.mean()),
    }


# --- reporting ---

def response_curve_figure(members, data, path):
    """Plot the median fitted response curve over the deviation-score distribution."""
    with torch.no_grad():
        d = np.column_stack([m.pixel_scores(data.X)[0].numpy() for m in members]).mean(axis=1)

    curves = np.array([m.curve() for m in members])
    a, b = np.median(curves[:, 0]), np.median(curves[:, 1])
    d50 = -b / a if abs(a) > 1e-6 else np.nan

    lower, upper = np.percentile(d, [1, 99])
    margin = 0.25 * (upper - lower)
    grid = np.linspace(lower - margin, upper + margin, 400)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    histogram = ax.twinx()
    histogram.hist(d, bins=60, color="0.75", alpha=0.5)
    histogram.set_ylabel("Pixel count", color="0.45")
    histogram.tick_params(axis="y", colors="0.45")

    ax.set_zorder(histogram.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.plot(grid, 1.0 / (1.0 + np.exp(-(a * grid + b))), color="black", lw=2,
            label=f"$a$ = {a:.2f}, $b$ = {b:.2f}")
    if np.isfinite(d50):
        ax.axvline(d50, color="black", ls=":", lw=1, label=f"$d_{{50}}$ = {d50:.2f}")

    ax.set_xlabel("Deviation score $d_j$")
    ax.set_ylabel("Outage probability $p_j$")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def cross_validate(df, out_dir):
    """Run bag-level k-fold cross-validation and write the per-fold tables."""
    folds = assign_folds(df, N_FOLDS, SEED)
    fold_results, observed, predicted, bag_ids = [], [], [], []

    for fold in range(N_FOLDS):
        train_df = df[folds != fold].reset_index(drop=True)
        valid_df = df[folds == fold].reset_index(drop=True)

        means, sds = standardisation_stats(train_df)
        train_data = BagDataset(train_df, means, sds)
        valid_data = BagDataset(valid_df, means, sds)
        print(f"fold {fold + 1}/{N_FOLDS}: {train_data.n_bags:,} training bags, "
              f"{valid_data.n_bags:,} validation bags")

        members, diagnostics, n_refits = fit_ensemble(train_data, N_MEMBERS, f"fold {fold + 1}")
        diagnostics.insert(0, "fold", fold + 1)
        diagnostics.to_csv(out_dir / f"members_fold{fold + 1}.csv", index=False)

        y_true = valid_data.y.numpy()
        fold_prediction = predict_ensemble(members, valid_data)
        metrics = evaluate(y_true, fold_prediction)
        member_r2 = [evaluate(y_true, predict_ensemble([m], valid_data))["r2"] for m in members]

        fold_results.append({"fold": fold + 1, **metrics,
                             "member_r2_mean": float(np.mean(member_r2)),
                             "member_r2_sd": float(np.std(member_r2, ddof=1)),
                             "reinitialisations": n_refits})
        print(f"  held out: R2 = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}, "
              f"MAE = {metrics['mae']:.3f}, bias = {metrics['bias']:+.3f}")

        observed.append(y_true)
        predicted.append(fold_prediction)
        bag_ids.extend(valid_data.bag_ids)

    observed, predicted = np.concatenate(observed), np.concatenate(predicted)
    fold_table = pd.DataFrame(fold_results)
    fold_table.to_csv(out_dir / "cross_validation_folds.csv", index=False)
    pd.DataFrame({GROUP_COL: bag_ids, "observed": observed, "predicted": predicted}).to_csv(
        out_dir / "cross_validation_predictions.csv", index=False
    )

    pooled = evaluate(observed, predicted)
    print("\ncross-validation")
    print(f"  across folds: R2 = {fold_table['r2'].mean():.3f} "
          f"(SD {fold_table['r2'].std(ddof=1):.3f}), "
          f"RMSE = {fold_table['rmse'].mean():.3f} "
          f"(SD {fold_table['rmse'].std(ddof=1):.3f})")
    print(f"  pooled out of fold: R2 = {pooled['r2']:.3f}, RMSE = {pooled['rmse']:.3f}, "
          f"MAE = {pooled['mae']:.3f}, bias = {pooled['bias']:+.3f}")

    return fold_results, fold_table, pooled


def save_final_model(members, means, sds, out_dir):
    """Save the fitted ensemble and the standardisation it was trained with.

    The standardisation stats travel with the model because they are part of
    it: a saved network expects features scaled the same way it was trained
    on, and those statistics must come from training data, never recomputed
    from whatever is being scored later.
    """
    model_dir = out_dir / "final_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    for i, member in enumerate(members):
        torch.save(member.state_dict(), model_dir / f"member_{i}.pt")

    with open(model_dir / "standardisation.json", "w") as handle:
        json.dump({
            "features": FEATURE_COLS,
            "means": means.to_dict(),
            "sds": sds.to_dict(),
            "n_members": len(members),
            "hidden_dim": HIDDEN_DIM,
        }, handle, indent=2)

    print(f"  saved {len(members)}-member ensemble to {model_dir}")
    return model_dir


def load_final_model(model_dir):
    """Load a saved ensemble plus the standardisation stats it was trained with.

    Returns (members, means, sds) where means/sds are pandas Series indexed by
    FEATURE_COLS, ready to pass straight into BagDataset for new data.
    """
    model_dir = Path(model_dir)
    with open(model_dir / "standardisation.json") as handle:
        stats = json.load(handle)

    if stats["features"] != FEATURE_COLS:
        raise ValueError(
            f"saved model uses features {stats['features']}, "
            f"this script is configured for {FEATURE_COLS}"
        )

    means = pd.Series(stats["means"])[FEATURE_COLS]
    sds = pd.Series(stats["sds"])[FEATURE_COLS]

    members = []
    for i in range(stats["n_members"]):
        model = OutageModel(len(FEATURE_COLS), hidden_dim=stats["hidden_dim"])
        model.load_state_dict(torch.load(model_dir / f"member_{i}.pt", weights_only=True))
        model.eval()
        members.append(model)

    return members, means, sds


def fit_final_model(df, out_dir):
    """Refit on the complete dataset and save the ensemble for later scoring."""
    means, sds = standardisation_stats(df)
    data = BagDataset(df, means, sds)
    print(f"\nfinal model: {data.n_bags:,} bags")

    members, diagnostics, _ = fit_ensemble(data, N_MEMBERS, "final")
    diagnostics.to_csv(out_dir / "members_final.csv", index=False)
    in_sample = evaluate(data.y.numpy(), predict_ensemble(members, data))

    curve = {key: float(diagnostics[column].agg(stat))
             for key, (column, stat) in {
                 "a": ("a", "median"), "a_sd": ("a", lambda s: s.std(ddof=1)),
                 "b": ("b", "median"), "b_sd": ("b", lambda s: s.std(ddof=1)),
                 "d50": ("d50", "median"), "d50_sd": ("d50", lambda s: s.std(ddof=1)),
             }.items()}

    print(f"  a = {curve['a']:.3f} (SD {curve['a_sd']:.3f}), "
          f"b = {curve['b']:.3f} (SD {curve['b_sd']:.3f}), "
          f"d50 = {curve['d50']:.3f} (SD {curve['d50_sd']:.3f})")
    print(f"  in sample: R2 = {in_sample['r2']:.3f}, RMSE = {in_sample['rmse']:.3f}")

    response_curve_figure(members, data, out_dir / "response_curve.png")
    save_final_model(members, means, sds, out_dir)
    return curve, in_sample


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="pixel-level input table (CSV)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="directory for tables, figures and results.json")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="seed for fold assignment and initialisation")
    parser.add_argument("--threads", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                        help="CPU threads for torch (defaults to SLURM's allocation, else 1)")
    parser.add_argument("--final-only", action="store_true",
                        help="skip cross-validation and only fit + save the final ensemble "
                             "(use this once you already have CV results and just want the "
                             "trained model to score on unseen data)")
    # parse_known_args, not parse_args: inside Jupyter, sys.argv carries the
    # kernel's own "-f <connection file>.json" flag, which argparse would
    # otherwise reject as unrecognized. From a terminal this behaves the same
    # as parse_args unless you genuinely pass an unknown flag.
    args, _ = parser.parse_known_args()
    return args


def main():
    global SEED
    args = parse_args()
    SEED = args.seed
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"data:    {args.data}")
    print(f"out:     {args.out}")
    print(f"threads: {args.threads}")

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    df = load_data(args.data)

    if args.final_only:
        curve, in_sample = fit_final_model(df, args.out)
        with open(args.out / "final_model_summary.json", "w") as handle:
            json.dump({
                "final_model": {"response_curve": curve, "in_sample": in_sample},
                "configuration": {
                    "seed": SEED, "n_members": N_MEMBERS, "hidden_dim": HIDDEN_DIM,
                    "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
                    "max_epochs": N_EPOCHS, "early_stopping_fraction": ES_FRACTION,
                    "features": FEATURE_COLS,
                },
            }, handle, indent=2)
        return

    fold_results, fold_table, pooled = cross_validate(df, args.out)
    curve, in_sample = fit_final_model(df, args.out)

    with open(args.out / "results.json", "w") as handle:
        json.dump({
            "cross_validation": {
                "folds": fold_results,
                "mean_r2": float(fold_table["r2"].mean()),
                "sd_r2": float(fold_table["r2"].std(ddof=1)),
                "pooled": pooled,
            },
            "final_model": {"response_curve": curve, "in_sample": in_sample},
            "configuration": {
                "n_folds": N_FOLDS, "seed": SEED, "n_members": N_MEMBERS,
                "hidden_dim": HIDDEN_DIM, "learning_rate": LR,
                "weight_decay": WEIGHT_DECAY, "max_epochs": N_EPOCHS,
                "early_stopping_fraction": ES_FRACTION, "features": FEATURE_COLS,
            },
        }, handle, indent=2)


if __name__ == "__main__":
    main()
