# 04b Empirical Bayes Shrinkage
# The model
#   K(t+1) ~ Binomial(BF(t+1), p)
#   logit(p) = a[role]                         role-specific intercept (SP / RP)
#            + b · features(t)                 skill metrics from season t, standardized
#            + u[pitcher]                      optional pitcher effect (off: see USE_PITCHER_EFFECT)
#            + c * post2021                    era term: shift for forecasts of 2022 onward
#                                              (on by default; see USE_ERA_TERM)
#            + e[pitcher-season]               season-to-season talent change: e ~ Normal(0, sigma_season)
#
# Features (all known at the end of season t):
#   3-season empirical Bayes K% (from 04a, on the logit scale), whiff rate, BB%,
#   fastball velocity, velocity change, four-seam spin (+ a flag when missing), age, age squared
#
# Why the two random effects matter:
#   - u lets the model learn that some pitchers consistently beat (or miss) what their
#     metrics say, pooled toward zero when there's little evidence.
#   - e is the year-to-year noise in true talent. Without it, the predictive intervals
#     would only include binomial luck and would be too narrow.
#

import sqlite3
import sys
import time
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
try:
    import pymc as pm
except ImportError:
    print("PyMC is not installed. In Command Prompt run:  py -m pip install pymc")
    sys.exit(1)
    
def _patch_arviz():
    try:
        import arviz.data.base as az_base
    except Exception:
        return
    from collections.abc import Mapping
 
    def _flatten_with_path(structure):
        out = []
 
        def walk(node, path):
            if isinstance(node, Mapping):
                for key in sorted(node):
                    walk(node[key], path + (key,))
            else:
                out.append((path, node))
 
        walk(structure, ())
        return out
 
    az_base._flatten_with_path = _flatten_with_path

_patch_arviz()


SAMPLER = "nutpie"        
DRAWS = 1000
TUNE = 1000
CHAINS = 4
SEED = 42

# The pitcher effect (u) was dropped: with it, sigma_pitcher did not converge
# (R-hat 1.073 at 1,000 draws, 1.026 at 2,000), and its posterior sat near zero
# (94% interval 0.003-0.060). Set to True to reproduce that version.
USE_PITCHER_EFFECT = False

# Era term: Phase 5 found every method over-forecasts 2022+ seasons (returning pitchers
# lose more ground relative to the league than before 2020). This adds one coefficient
# for forecasts made after season ERA_START_SEASON or later (targets 2022+). It uses only
# information known at forecast time: which season it is. Set to False for the old model.
USE_ERA_TERM = True
ERA_START_SEASON = 2021

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "baseball.db"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
 
FEATURES = ["eb_k_logit", "whiff_pct", "bb_pct", "fb_velo", "velo_change",
            "ff_spin", "spin_missing", "age", "age_sq"]
FEATURE_LABELS = {
    "eb_k_logit": "3-season EB K%", "whiff_pct": "Whiff%", "bb_pct": "BB%",
    "fb_velo": "Fastball velocity", "velo_change": "Velocity change",
    "ff_spin": "Four-seam spin", "spin_missing": "No four-seamer",
    "age": "Age", "age_sq": "Age squared",
}
ROLES = ["RP", "SP"]
INTERVALS = [0.50, 0.80, 0.95]

# Same Chart Style
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
CHAIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
 
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 110,
})
 
 
def save(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")

# Scoring helpers (same definitions as Phases 3 and 4a)
def score(df, pred_col):
    err = df[pred_col] - df["next_k_pct"]
    w = df["next_bf"]
    return pd.Series({
        "rmse": np.sqrt(np.sum(w * err ** 2) / w.sum()),
        "mae": np.sum(w * err.abs()) / w.sum(),
        "bias": np.sum(w * err) / w.sum(),
        "n": len(df),
    })
 
 
def score_table(df, methods):
    t = pd.DataFrame({label: score(df, col) for label, col in methods.items()}).T
    t["vs_persistence"] = 1 - t["rmse"] / t.loc["Persistence", "rmse"]
    t["vs_marcel"] = 1 - t["rmse"] / t.loc["Marcel", "rmse"]
    return t
 
 
def show(t):
    out = t.copy()
    for c in ["rmse", "mae", "bias"]:
        out[c] = (out[c] * 100).map(lambda v: f"{v:.2f}")
    out["n"] = out["n"].astype(int)
    for c in ["vs_persistence", "vs_marcel"]:
        out[c] = out[c].map(lambda v: f"{v:+.1%}")
    out.columns = ["RMSE (pts)", "MAE (pts)", "Bias (pts)", "Pairs",
                   "vs persistence", "vs Marcel"]
    print(out.to_string())
 
 
def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# Data
def load_data():
    con = sqlite3.connect(DB_PATH)
    pairs = pd.read_sql("SELECT * FROM model_pairs WHERE involves_2020 = 0", con)
    base = pd.read_sql("SELECT player_id, season, pred_league, pred_persist, pred_marcel "
                       "FROM baseline_forecasts", con)
    eb = pd.read_sql("SELECT player_id, season, split, pred_eb1, pred_eb3, pred_eb3_tuned, "
                     "age_factor FROM eb_forecasts", con)
    con.close()
    df = (pairs.merge(base, on=["player_id", "season"], how="inner")
               .merge(eb, on=["player_id", "season"], how="inner"))
    df["target_season"] = df["season"] + 1
    df["next_k"] = df["next_k"].round().astype(int)
    df["next_bf"] = df["next_bf"].round().astype(int)
    return df
 
 
def build_features(df, train_mask):
    """Standardized feature matrix. Means/SDs and fill values come from TRAIN rows only."""
    f = pd.DataFrame(index=df.index)
    # EB 3-season K% without the age adjustment (age is its own feature here)
    f["eb_k_logit"] = logit(df["pred_eb3"] / df["age_factor"])
    f["whiff_pct"] = df["whiff_pct"]
    f["bb_pct"] = df["bb_pct"]
    f["fb_velo"] = df["fb_velo"]
    f["velo_change"] = df["velo_change"].fillna(0)        # no prior season -> no change
    f["spin_missing"] = df["ff_spin"].isna().astype(float)
    f["ff_spin"] = df["ff_spin"]
    f["age"] = df["age"]
 
    tr = f[train_mask]
    for col in f.columns:
        f[col] = f[col].fillna(tr[col].mean())
    mean, sd = f[train_mask].mean(), f[train_mask].std().replace(0, 1)
    z = (f - mean) / sd
    z["age_sq"] = z["age"] ** 2
    z["age_sq"] = (z["age_sq"] - z.loc[train_mask, "age_sq"].mean()) / z.loc[train_mask, "age_sq"].std()
    return z[FEATURES].to_numpy()


# Model
def sample_posterior(X, role_idx, pitcher_idx, n_pitchers, next_bf, next_k, era):
    """Fit the model; returns posterior draws shaped (chain, draw, ...) and the divergence count."""
    coords = {"role": ROLES, "feature": FEATURES,
              "pitcher": np.arange(n_pitchers), "obs": np.arange(len(next_k))}
    with pm.Model(coords=coords):
        a = pm.Normal("a_role", mu=logit(0.22), sigma=0.5, dims="role")
        b = pm.Normal("b", mu=0, sigma=0.5, dims="feature")
 
        sigma_season = pm.HalfNormal("sigma_season", sigma=0.3)
        z_season = pm.Normal("z_season", 0, 1, dims="obs")
        eta = a[role_idx] + pm.math.dot(X, b) + z_season * sigma_season

        if USE_ERA_TERM:
            era_effect = pm.Normal("era_effect", mu=0, sigma=0.3)
            eta = eta + era_effect * era

        if USE_PITCHER_EFFECT:
            sigma_pitcher = pm.HalfNormal("sigma_pitcher", sigma=0.3)
            z_pitcher = pm.Normal("z_pitcher", 0, 1, dims="pitcher")
            u = pm.Deterministic("u", z_pitcher * sigma_pitcher, dims="pitcher")
            eta = eta + u[pitcher_idx]
        pm.Binomial("next_k", n=next_bf, logit_p=eta, observed=next_k, dims="obs")
 
        kwargs = dict(draws=DRAWS, tune=TUNE, chains=CHAINS, target_accept=0.9,
                      random_seed=SEED)
        if SAMPLER == "nutpie":
            idata = pm.sample(nuts_sampler="nutpie", **kwargs)
        else:
            idata = pm.sample(cores=1, **kwargs)
 
    names = ["a_role", "b", "sigma_season"]
    if USE_ERA_TERM:
        names += ["era_effect"]
    if USE_PITCHER_EFFECT:
        names += ["u", "sigma_pitcher"]
    draws = {n: np.asarray(idata.posterior[n]) for n in names}
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    return draws, divergences
 
 
def flatten(draws):
    """(chain, draw, ...) -> (sample, ...)"""
    return {k: v.reshape(-1, *v.shape[2:]) for k, v in draws.items()}
 
 
def split_rhat(x):
    """Split R-hat for one parameter, x shaped (chain, draw). Values near 1.00 = converged."""
    half = x.shape[1] // 2
    chains = np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)
    n = chains.shape[1]
    w = chains.var(axis=1, ddof=1).mean()
    b = n * chains.mean(axis=1).var(ddof=1)
    var_hat = (n - 1) / n * w + b / n
    return float(np.sqrt(var_hat / w)) if w > 0 else np.nan
 
 
def predict(d, X, role_idx, pitcher_idx, next_bf, rng, era):
    """
    Posterior predictive forecasts.
      d            flattened posterior draws
      pitcher_idx  index into the training pitchers, or -1 for a pitcher the model hasn't seen
                   (his pitcher effect is drawn from the population distribution instead)
    Returns the point forecast (posterior mean of p) and simulated K% outcomes (sample x pair).
    The simulation uses the ACTUAL next-season batters faced, so intervals are conditional
    on playing time.
    """
    S, n = d["b"].shape[0], X.shape[0]
    eta = d["a_role"][:, role_idx] + d["b"] @ X.T
    if "era_effect" in d:
        eta = eta + d["era_effect"][:, None] * era[None, :]
    u = np.zeros((S, n))
    if "u" in d:
        known = pitcher_idx >= 0
        u[:, known] = d["u"][:, pitcher_idx[known]]
        u[:, ~known] = rng.normal(size=(S, (~known).sum())) * d["sigma_pitcher"][:, None]
    e = rng.normal(size=(S, n)) * d["sigma_season"][:, None]
    p = 1 / (1 + np.exp(-(eta + u + e)))
    sims = rng.binomial(next_bf[None, :], p) / next_bf[None, :]
    return p.mean(axis=0), sims

# Main
def main():
    rng = np.random.default_rng(SEED)
    df = load_data()
    train_mask = (df["split"] == "train").to_numpy()
    test_mask = ~train_mask
    print(f"Pairs: {len(df):,} (train {train_mask.sum():,}, test {test_mask.sum():,})")
 
    X = build_features(df, train_mask)
    role_idx = df["role"].map({r: i for i, r in enumerate(ROLES)}).to_numpy()
    era = (df["season"] >= ERA_START_SEASON).astype(float).to_numpy()
 
    train_pitchers = np.sort(df.loc[train_mask, "player_id"].unique())
    pid_to_idx = {p: i for i, p in enumerate(train_pitchers)}
    pitcher_idx = df["player_id"].map(pid_to_idx).fillna(-1).astype(int).to_numpy()
    seen = (pitcher_idx[test_mask] >= 0).mean()
    print(f"Training pitchers: {len(train_pitchers):,}; "
          f"{seen:.0%} of test pairs are pitchers the model has seen before")
 
    # fit
    print(f"\nSampling ({CHAINS} chains x {DRAWS} draws, sampler = {SAMPLER}) ...")
    t0 = time.time()
    draws, divergences = sample_posterior(
        X[train_mask], role_idx[train_mask], pitcher_idx[train_mask], len(train_pitchers),
        df.loc[train_mask, "next_bf"].to_numpy(), df.loc[train_mask, "next_k"].to_numpy(),
        era[train_mask])
    print(f"  done in {(time.time() - t0) / 60:.1f} minutes")
    d = flatten(draws)
 
    # diagnostics 
    print("\nConvergence diagnostics")
    rows = []
    for i, r in enumerate(ROLES):
        rows.append((f"a_role[{r}]", draws["a_role"][:, :, i]))
    for i, f in enumerate(FEATURES):
        rows.append((f"b[{FEATURE_LABELS[f]}]", draws["b"][:, :, i]))
    if "sigma_pitcher" in draws:
        rows.append(("sigma_pitcher", draws["sigma_pitcher"]))
    rows.append(("sigma_season", draws["sigma_season"]))
    if "era_effect" in draws:
        rows.append(("era_effect", draws["era_effect"]))
    summary = pd.DataFrame([{
        "parameter": name, "mean": x.mean(), "sd": x.std(),
        "3%": np.percentile(x, 3), "97%": np.percentile(x, 97), "r_hat": split_rhat(x),
    } for name, x in rows]).set_index("parameter")
    print(summary.round(3).to_string())
    u_rhat = 1.0
    if "u" in draws:
        u_rhat = max(split_rhat(draws["u"][:, :, j]) for j in range(draws["u"].shape[2]))
        print(f"\n  largest R-hat among pitcher effects: {u_rhat:.3f}")
    print(f"  divergent transitions: {divergences}")
    worst = max(summary["r_hat"].max(), u_rhat)
    if worst > 1.01 or divergences > 0:
        print("  WARNING: R-hat above 1.01 or divergences found. Try raising TUNE to 2000 and "
              "re-running; mention it in your write-up if it persists.")
    else:
        print("  All R-hat values <= 1.01 and no divergences: the chains agree.")
 
    print("\nHow to read the coefficients: each b is the change in log-odds of a strikeout "
          "for a 1-SD increase in that feature, holding the others fixed. Because the features "
          "are correlated (K%, whiff rate), look at them together, not one at a time.")
    if "sigma_pitcher" in d:
        print(f"  sigma_pitcher = {d['sigma_pitcher'].mean():.3f}: persistent pitcher "
              f"differences the metrics don't capture")
    print(f"  sigma_season  = {d['sigma_season'].mean():.3f}: season-to-season talent change")
    if "era_effect" in d:
        e_mean = d["era_effect"].mean()
        e_lo, e_hi = np.percentile(d["era_effect"], [3, 97])
        k = 0.22 * 0.78 * 100          # log-odds -> K% points near a 22% strikeout rate
        print(f"  era_effect    = {e_mean:.3f} (94% interval {e_lo:.3f} to {e_hi:.3f}): "
              f"forecasts of {ERA_START_SEASON + 1}+ shift by about {e_mean * k:+.2f} K% points")
 
    # forecasts
    pred, sims = predict(d, X, role_idx, pitcher_idx, df["next_bf"].to_numpy(), rng, era)
    df["pred_hier"] = pred
    for level in INTERVALS:
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        tag = int(level * 100)
        df[f"lo{tag}"] = np.quantile(sims, lo, axis=0)
        df[f"hi{tag}"] = np.quantile(sims, hi, axis=0)
 
    METHODS = {
        "League average": "pred_league", "Persistence": "pred_persist",
        "Marcel": "pred_marcel", "EB 3-season": "pred_eb3",
        "EB 3-season (tuned)": "pred_eb3_tuned", "Hierarchical": "pred_hier",
    }
    print("\nTRAIN pairs (in-sample for the hierarchical model)")
    show(score_table(df[train_mask], METHODS))
    print("\nTEST pairs (forecasting 2025 and 2026)")
    test_table = score_table(df[test_mask], METHODS)
    show(test_table)
    print("\nTest pairs by role")
    for role in ["SP", "RP"]:
        print(f"  {role}:")
        show(score_table(df[test_mask & (df["role"] == role).to_numpy()], METHODS))
 
    # interval calibration
    print("\nPredictive interval coverage on TEST pairs (should be close to the nominal level)")
    levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    test_sims = sims[:, test_mask]
    actual = df.loc[test_mask, "next_k_pct"].to_numpy()
    coverage = []
    for level in levels:
        lo = np.quantile(test_sims, (1 - level) / 2, axis=0)
        hi = np.quantile(test_sims, 1 - (1 - level) / 2, axis=0)
        coverage.append(np.mean((actual >= lo) & (actual <= hi)))
    for level, c in zip(levels, coverage):
        print(f"  {level:.0%} interval: covers {c:.1%} of actual outcomes")
    width80 = (df.loc[test_mask, "hi80"] - df.loc[test_mask, "lo80"]).median()
    print(f"  median 80% interval width: {width80 * 100:.1f} K% points "
          f"(intervals use actual next-season batters faced)")
 
    # save
    con = sqlite3.connect(DB_PATH)
    out = ["player_id", "name", "season", "target_season", "split", "role",
           "next_bf", "next_k_pct", "pred_hier"] + \
          [f"{s}{int(l * 100)}" for l in INTERVALS for s in ("lo", "hi")]
    df[out].to_sql("hier_forecasts", con, if_exists="replace", index=False)
    summary.reset_index().to_sql("hier_coefficients", con, if_exists="replace", index=False)
    con.close()
 
    # charts
    print("\nCharts")
 
    # 1) coefficients
    coef = summary.loc[[f"b[{FEATURE_LABELS[f]}]" for f in FEATURES]].copy()
    coef.index = [FEATURE_LABELS[f] for f in FEATURES]
    coef = coef.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(coef))
    ax.hlines(y, coef["3%"], coef["97%"], color=BLUE, linewidth=2)
    ax.plot(coef["mean"], y, "o", color=BLUE, markersize=8)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(y, coef.index)
    ax.set_xlabel("Effect on log-odds of a strikeout, per 1 SD (94% interval)")
    ax.set_title("What predicts next season's strikeout rate?")
    ax.grid(axis="y", visible=False)
    save(fig, "11_coefficients.png")
 
    # 2) trace plots for the key parameters
    trace_params = [("sigma_season", draws["sigma_season"]),
                    ("b[3-season EB K%]", draws["b"][:, :, 0]),
                    ("b[Whiff%]", draws["b"][:, :, 1]),
                    ("a_role[SP]", draws["a_role"][:, :, 1])]
    if "sigma_pitcher" in draws:
        trace_params.insert(0, ("sigma_pitcher", draws["sigma_pitcher"]))
    fig, axes = plt.subplots(len(trace_params), 1, figsize=(9, 8), sharex=True)
    for ax, (name, x) in zip(axes, trace_params):
        for c in range(x.shape[0]):
            ax.plot(x[c], color=CHAIN_COLORS[c % 4], linewidth=0.6, alpha=0.7,
                    label=f"Chain {c + 1}")
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
    axes[0].set_title("Trace plots: chains should overlap like fuzzy caterpillars")
    axes[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.15), fontsize=9)
    axes[-1].set_xlabel("Draw")
    save(fig, "12_trace.png")
 
    # 3) posterior predictive check on TRAIN pairs
    bins = np.arange(0, 0.55, 0.01)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    train_sims = sims[:, train_mask]
    for i, s in enumerate(rng.choice(train_sims.shape[0], 30, replace=False)):
        ax.hist(train_sims[s], bins=bins, histtype="step", color=BLUE_LIGHT, linewidth=1,
                alpha=0.6, label="Simulated from the model" if i == 0 else None)
    ax.hist(df.loc[train_mask, "next_k_pct"], bins=bins, histtype="step", color=INK,
            linewidth=2, label="Actual")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Next-season K%")
    ax.set_ylabel("Pitchers")
    ax.set_title("Posterior predictive check: simulated vs actual outcomes")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "13_posterior_predictive.png")
 
    # 4) interval calibration
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0.4, 1], [0.4, 1], color=INK_2, linewidth=1.2, linestyle="--")
    ax.text(0.98, 0.95, "Perfect calibration", color=INK_2, fontsize=9, ha="right")
    ax.plot(levels, coverage, color=BLUE, marker="o", markersize=8)
    for level, c in zip(levels, coverage):
        ax.text(level + 0.012, c - 0.02, f"{c:.0%}", fontsize=9, color=INK)
    ax.set_xlim(0.4, 1)
    ax.set_ylim(0.4, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Nominal interval level")
    ax.set_ylabel("Share of 2025–26 outcomes inside the interval")
    ax.set_title("Are the forecast intervals honest?")
    save(fig, "14_interval_calibration.png")
 
    # 5) test RMSE, every model
    t = test_table.sort_values("rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.barh(t.index, t["rmse"] * 100, color=BLUE, height=0.6)
    for bar, (lab, row) in zip(bars, t.iterrows()):
        note = "" if lab == "Marcel" else f"   ({row['vs_marcel']:+.1%} vs Marcel)"
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{row['rmse'] * 100:.2f}{note}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, t["rmse"].max() * 100 * 1.45)
    ax.set_xlabel("Weighted RMSE, K% points (lower is better)")
    ax.set_title("Test error, all models")
    ax.grid(axis="y", visible=False)
    save(fig, "15_test_rmse_all_models.png")
 
 
 
if __name__ == "__main__":     
    main()
