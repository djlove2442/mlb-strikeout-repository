# 04a Empirical Bayes Shrinkage
# The model (beta-binomial):
#   Each pitcher has a true strikeout rate p. Over BF batters, his strikeouts are
#   K ~ Binomial(BF, p). True rates across pitchers follow a Beta(alpha, beta) prior,
#   fit separately for starters and relievers, separately for each season.
#
#   Posterior mean of p = (K + alpha) / (BF + alpha + beta)
#
#   alpha + beta acts like "prior batters faced": a pitcher with far fewer BF than that
#   gets pulled most of the way to his role's average; a workhorse barely moves.
#
# Priors are fit by method of moments, using only season-t data, so every forecast
# uses only information available at the time.
#
# Three versions:
#   EB 1-season          - shrink season t alone
#   EB 3-season          - shrink seasons t, t-1, t-2 combined (weights 1, 0.8, 0.6,
#                          Marcel's 5/4/3 rescaled), plus Marcel's age adjustment
#   EB 3-season (tuned)  - same, with the prior strength scaled by one factor tuned on
#                          TRAIN pairs. Pure EB measures how spread out pitchers are
#                          right now; forecasting next year also has to cover talent
#                          changing between seasons, which may call for more shrinkage.

import sqlite3
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
 
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "baseball.db"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
 
PRIOR_MIN_BF = 100                     
HISTORY_WEIGHTS = {0: 1.0, 1: 0.8, 2: 0.6}
EXAMPLE_SEASON = 2025                  

# Same Chart Style
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
BLUE_DARK = "#0d3a75"
 
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

# Scoring helpers (same definitions as Phase 3)
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
 
 
# Load data
con = sqlite3.connect(DB_PATH)
seasons = pd.read_sql("SELECT * FROM pitcher_seasons", con)
base = pd.read_sql("SELECT * FROM baseline_forecasts", con)
settings = pd.read_sql("SELECT * FROM settings", con).set_index("setting")["value"]
use_age = settings["marcel_use_age"] == "True"
 
print(f"Pairs: {len(base):,} (train {sum(base['split'] == 'train'):,}, "
      f"test {sum(base['split'] == 'test'):,})")

# Step 1: fit a Beta prior for each season and role (method of moments)
#   mean        mu   = total K / total BF
#   observed variance of K% across pitchers = true-talent variance + binomial noise
#   true-talent variance  tau^2 = observed variance - average of mu(1-mu)/BF
#   prior strength  alpha + beta = mu(1-mu)/tau^2 - 1

def fit_beta_prior(d):
    mu = d["k"].sum() / d["bf"].sum()
    obs_var = np.mean((d["k_pct"] - mu) ** 2)
    noise_var = np.mean(mu * (1 - mu) / d["bf"])
    tau2 = max(obs_var - noise_var, 1e-5)
    strength = mu * (1 - mu) / tau2 - 1
    return pd.Series({"pitchers": len(d), "prior_mean": mu,
                      "talent_sd": np.sqrt(tau2), "prior_strength": strength,
                      "alpha": mu * strength, "beta": (1 - mu) * strength})
 
 
priors = (seasons[seasons["bf"] >= PRIOR_MIN_BF]
          .groupby(["season", "role"])
          .apply(fit_beta_prior, include_groups=False)
          .reset_index())
 
print("\nBeta priors by season and role")
show_priors = priors.copy()
show_priors["prior_mean"] = show_priors["prior_mean"].map("{:.1%}".format)
show_priors["talent_sd"] = (show_priors["talent_sd"] * 100).map("{:.1f} pts".format)
show_priors["prior_strength"] = show_priors["prior_strength"].map("{:.0f} BF".format)
show_priors["pitchers"] = show_priors["pitchers"].astype(int)
print(show_priors[["season", "role", "pitchers", "prior_mean", "talent_sd", "prior_strength"]]
      .to_string(index=False))
print("  prior_strength = how many batters faced of 'role average' each forecast starts with.")
print("  talent_sd = spread of TRUE strikeout skill, after removing small-sample noise.")

# Step 2: shrink each pitcher towards his role's prior
df = base.merge(priors[["season", "role", "alpha", "beta", "prior_mean", "prior_strength"]],
                on=["season", "role"], how="left")
cur = seasons[["player_id", "season", "k", "bf"]]
df = df.merge(cur.rename(columns={"bf": "bf_cur"}), on=["player_id", "season"], how="left")
 
# weighted three-season counts
wk = np.zeros(len(df))
wbf = np.zeros(len(df))
for lag, w in HISTORY_WEIGHTS.items():
    h = cur.copy()
    h["season"] = h["season"] + lag
    h = df[["player_id", "season"]].merge(h, on=["player_id", "season"], how="left")
    wk += w * h["k"].fillna(0).to_numpy()
    wbf += w * h["bf"].fillna(0).to_numpy()
df["wk"], df["wbf"] = wk, wbf
 
age = df["age"].fillna(29)
df["age_factor"] = np.where(age < 29, 1 + (29 - age) * 0.006, 1 + (29 - age) * 0.003)
if not use_age:
    df["age_factor"] = 1.0
 
 
def eb_three_season(d, scale=1.0):
    a, b = d["alpha"] * scale, d["beta"] * scale
    return (d["wk"] + a) / (d["wbf"] + a + b) * d["age_factor"]
 
 
df["pred_eb1"] = (df["k"] + df["alpha"]) / (df["bf_cur"] + df["alpha"] + df["beta"])
df["pred_eb3"] = eb_three_season(df)
 
# tune one prior-strength multiplier on TRAIN pairs
train = df[df["split"] == "train"]
grid = np.round(np.arange(0.25, 6.01, 0.05), 2)
rmse_by_scale = pd.Series({c: score(train.assign(p=eb_three_season(train, c)), "p")["rmse"]
                           for c in grid})
BEST_SCALE = float(rmse_by_scale.idxmin())
df["pred_eb3_tuned"] = eb_three_season(df, BEST_SCALE)
 
print(f"\nTuned prior-strength multiplier: {BEST_SCALE:.2f}")
print(f"  i.e. about {BEST_SCALE:.1f}x the pure empirical Bayes shrinkage "
      f"({'more' if BEST_SCALE > 1 else 'less'} shrinkage is better for forecasting next season)")
if BEST_SCALE in (grid.min(), grid.max()):
    print("  NOTE: best multiplier is at the edge of the grid; widen the grid and re-run.")

# Results
METHODS = {
    "League average": "pred_league",
    "Persistence": "pred_persist",
    "Marcel": "pred_marcel",
    "EB 1-season": "pred_eb1",
    "EB 3-season": "pred_eb3",
    "EB 3-season (tuned)": "pred_eb3_tuned",
}
 
print("\nTRAIN pairs (tuned version is in-sample here)")
show(score_table(df[df["split"] == "train"], METHODS))
 
print("\nTEST pairs (forecasting 2025 and 2026)")
test_table = score_table(df[df["split"] == "test"], METHODS)
show(test_table)
 
print("\nTest pairs by role")
for role in ["SP", "RP"]:
    print(f"  {role}:")
    show(score_table(df[(df["split"] == "test") & (df["role"] == role)], METHODS))
 
print("\nBias by forecast season (pts; positive = forecast too high)")
bias_by_year = (df.groupby("target_season")
                .apply(lambda d: pd.Series({lab: score(d, col)["bias"]
                                            for lab, col in METHODS.items()
                                            if lab != "League average"}),
                       include_groups=False))
print((bias_by_year * 100).round(2).to_string())

# Save
priors.to_sql("eb_priors", con, if_exists="replace", index=False)
keep = ["player_id", "season", "target_season", "split", "role", "prior_mean",
        "prior_strength", "wk", "wbf", "age_factor",
        "pred_eb1", "pred_eb3", "pred_eb3_tuned"]
df[keep].to_sql("eb_forecasts", con, if_exists="replace", index=False)
con.execute("DELETE FROM settings WHERE setting = 'eb_scale'")
pd.DataFrame({"setting": ["eb_scale"], "value": [str(BEST_SCALE)]}).to_sql(
    "settings", con, if_exists="append", index=False)
con.close()

# Charts
print("\nCharts")
 
# 1) What shrinkage does: raw K% vs shrunk K% for one season, colored by batters faced
ex = seasons[(seasons["season"] == EXAMPLE_SEASON) & (seasons["bf"] >= 20)]
ex = ex.merge(priors, on=["season", "role"], how="inner")
ex["shrunk"] = (ex["k"] + ex["alpha"]) / (ex["bf"] + ex["alpha"] + ex["beta"])
ex = ex.sort_values("bf")
 
ramp = LinearSegmentedColormap.from_list("bf", [BLUE_LIGHT, BLUE, BLUE_DARK])
fig, ax = plt.subplots(figsize=(7.5, 6.2))
sc = ax.scatter(ex["k_pct"], ex["shrunk"], c=np.log10(ex["bf"]), cmap=ramp,
                s=16, edgecolors="none")
lims = [0, max(0.5, ex["k_pct"].max() + 0.02)]
ax.plot(lims, lims, color=INK_2, linewidth=1.2, linestyle="--")
ax.text(lims[1] * 0.97, lims[1] * 0.93, "No shrinkage", color=INK_2, fontsize=9, ha="right")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ticks = [20, 50, 100, 250, 500, 900]
cbar = fig.colorbar(sc, ax=ax, shrink=0.8, ticks=np.log10(ticks))
cbar.ax.set_yticklabels([str(t) for t in ticks])
cbar.set_label("Batters faced")
ax.set_xlabel(f"Observed K%, {EXAMPLE_SEASON}")
ax.set_ylabel("Empirical Bayes estimate of true K%")
ax.set_title("Small samples get pulled hardest toward average")
save(fig, "09_shrinkage_example.png")
 
# 2) Test RMSE, all methods so far
t = test_table.sort_values("rmse", ascending=False)
fig, ax = plt.subplots(figsize=(8.5, 4.4))
bars = ax.barh(t.index, t["rmse"] * 100, color=BLUE, height=0.6)
for bar, (lab, row) in zip(bars, t.iterrows()):
    note = "" if lab == "Marcel" else f"   ({row['vs_marcel']:+.1%} vs Marcel)"
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{row['rmse'] * 100:.2f}{note}", va="center", fontsize=9, color=INK)
ax.set_xlim(0, t["rmse"].max() * 100 * 1.45)
ax.set_xlabel("Weighted RMSE, K% points (lower is better)")
ax.set_title("Test error: baselines vs empirical Bayes")
ax.grid(axis="y", visible=False)
save(fig, "10_test_rmse_with_eb.png")
