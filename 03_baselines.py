# 03 Baseline Forecasts

import sqlite3
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "baseball.db"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
 
TEST_SEASONS = [2024, 2025]          
MARCEL_WEIGHTS = {0: 5, 1: 4, 2: 3}

# Chart Style
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
 
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


# Scoring helpers
def score(df, pred_col):
    """Batters-faced-weighted RMSE, MAE, and bias, in K% points."""
    err = df[pred_col] - df["next_k_pct"]
    w = df["next_bf"]
    return pd.Series({
        "rmse": np.sqrt(np.sum(w * err ** 2) / w.sum()),
        "mae": np.sum(w * err.abs()) / w.sum(),
        "bias": np.sum(w * err) / w.sum(),
        "n": len(df),
    })
 
 
def score_table(df, methods):
    rows = {label: score(df, col) for label, col in methods.items()}
    t = pd.DataFrame(rows).T
    base = t.loc["Persistence", "rmse"]
    t["vs_persistence"] = 1 - t["rmse"] / base
    return t
 
 
def show(t):
    out = t.copy()
    for c in ["rmse", "mae", "bias"]:
        out[c] = (out[c] * 100).map(lambda v: f"{v:.2f}")
    out["n"] = out["n"].astype(int)
    out["vs_persistence"] = out["vs_persistence"].map(lambda v: f"{v:+.1%}")
    out.columns = ["RMSE (pts)", "MAE (pts)", "Bias (pts)", "Pairs", "RMSE vs persistence"]
    print(out.to_string())


# Load Data
con = sqlite3.connect(DB_PATH)
seasons = pd.read_sql("SELECT * FROM pitcher_seasons", con)
pairs = pd.read_sql("SELECT * FROM model_pairs", con)
 
pairs = pairs[pairs["involves_2020"] == 0].copy()
pairs["target_season"] = pairs["season"] + 1
pairs["split"] = np.where(pairs["season"].isin(TEST_SEASONS), "test", "train")
 
print(f"Scored pairs: {len(pairs):,} "
      f"(train {sum(pairs['split'] == 'train'):,}, test {sum(pairs['split'] == 'test'):,})")
 
# League K% by season, using every pitcher (batters-faced weighted)
league = seasons.groupby("season")[["k", "bf"]].sum()
league_k = league["k"] / league["bf"]


# Baseline 1: league average (last season's league K%; next season's is not known yet)
# Baseline 2: persistence
pairs["pred_league"] = pairs["season"].map(league_k)
pairs["pred_persist"] = pairs["k_pct"]

# Baseline 3: Marcel
#
#   weighted K   = 5*K(t)  + 4*K(t-1)  + 3*K(t-2)
#   weighted BF  = 5*BF(t) + 4*BF(t-1) + 3*BF(t-2)
#   league rate  = league K% over the same seasons, weighted the same way
#   forecast     = (weighted K + R * league rate) / (weighted BF + R)
#   then an age adjustment: +0.6% per year under 29, -0.3% per year over 29

hist = seasons[["player_id", "season", "k", "bf"]]
m = pairs[["player_id", "season"]].copy()
for lag in MARCEL_WEIGHTS:
    h = hist.copy()
    h["season"] = h["season"] + lag          # season s lines up with t = s + lag
    m = m.merge(h.rename(columns={"k": f"k_{lag}", "bf": f"bf_{lag}"}),
                on=["player_id", "season"], how="left")
    m[f"lg_{lag}"] = (m["season"] - lag).map(league_k)
 
wk = wbf = wlg = 0
for lag, w in MARCEL_WEIGHTS.items():
    k_l = m[f"k_{lag}"].fillna(0)
    bf_l = m[f"bf_{lag}"].fillna(0)
    lg_l = m[f"lg_{lag}"].fillna(0)
    wk = wk + w * k_l
    wbf = wbf + w * bf_l
    wlg = wlg + w * bf_l * lg_l
pairs["marcel_wk"] = wk.values
pairs["marcel_wbf"] = wbf.values
pairs["marcel_lg"] = (wlg / wbf).values
 
age = pairs["age"].fillna(29)
pairs["age_factor"] = np.where(age < 29, 1 + (29 - age) * 0.006, 1 + (29 - age) * 0.003)
 
 
def marcel(df, R, use_age=True):
    rate = (df["marcel_wk"] + R * df["marcel_lg"]) / (df["marcel_wbf"] + R)
    return rate * df["age_factor"] if use_age else rate
 
 
# tune R (and whether the age adjustment helps) on train pairs only 
print("\nTuning Marcel on train pairs ...")
train = pairs[pairs["split"] == "train"]
grid = np.arange(0, 8001, 100)
tuning = []
for use_age in (True, False):
    for R in grid:
        tuning.append({"R": R, "use_age": use_age,
                       "rmse": score(train.assign(p=marcel(train, R, use_age)), "p")["rmse"]})
tuning = pd.DataFrame(tuning)
best = tuning.loc[tuning["rmse"].idxmin()]
BEST_R, BEST_AGE = int(best["R"]), bool(best["use_age"])
 
print(f"  best R = {BEST_R:,} weighted BF  (about {BEST_R / 5:,.0f} batters faced of "
      f"league-average pitching at this-season weight)")
print(f"  age adjustment: {'helps, kept' if BEST_AGE else 'does not help, dropped'}")
for use_age in (True, False):
    t = tuning[tuning["use_age"] == use_age]
    r = t.loc[t["rmse"].idxmin()]
    print(f"    with age={use_age}: best train RMSE {r['rmse'] * 100:.3f} pts at R = {int(r['R']):,}")
if BEST_R == grid.max():
    print("  NOTE: best R is at the edge of the grid; widen the grid and re-run.")
 
pairs["pred_marcel"] = marcel(pairs, BEST_R, BEST_AGE)


# Results
METHODS = {"League average": "pred_league", "Persistence": "pred_persist",
           "Marcel": "pred_marcel"}
 
print("\nTRAIN pairs (Marcel was tuned here, so this is in-sample for Marcel)")
show(score_table(pairs[pairs["split"] == "train"], METHODS))
 
print("\nTEST pairs (forecasting 2025 and 2026; the honest comparison)")
test_table = score_table(pairs[pairs["split"] == "test"], METHODS)
show(test_table)
 
print("\nTest pairs by role")
for role in ["SP", "RP"]:
    print(f"  {role}:")
    show(score_table(pairs[(pairs["split"] == "test") & (pairs["role"] == role)], METHODS))
 
print("\nRMSE by forecast target season (pts)")
by_year = (pairs.groupby("target_season")
           .apply(lambda d: pd.Series({lab: score(d, col)["rmse"] for lab, col in METHODS.items()}),
                  include_groups=False))
print((by_year * 100).round(2).to_string())
 
# A few familiar names: biggest strikeout pitchers in the last test year
print("\nExample forecasts for 2026 (top 2025 K%, 400+ BF)")
ex = (pairs[(pairs["season"] == 2025) & (pairs["bf"] >= 400)]
      .nlargest(10, "k_pct")
      [["name", "role", "k_pct", "pred_marcel", "next_k_pct"]])
ex.columns = ["Pitcher", "Role", "2025 K% (persistence)", "Marcel forecast", "Actual 2026"]
print(ex.to_string(index=False, formatters={c: "{:.1%}".format for c in ex.columns[2:]}))


# Save forecasts
out_cols = ["player_id", "name", "season", "target_season", "split", "role", "age",
            "bf", "k_pct", "next_bf", "next_k_pct",
            "pred_league", "pred_persist", "pred_marcel"]
pairs[out_cols].to_sql("baseline_forecasts", con, if_exists="replace", index=False)
pd.DataFrame({"setting": ["marcel_R", "marcel_use_age", "test_seasons"],
              "value": [str(BEST_R), str(BEST_AGE), ",".join(map(str, TEST_SEASONS))]}
             ).to_sql("settings", con, if_exists="replace", index=False)
con.close()


# Charts
print("\nCharts")
 
# Test RMSE by method
t = test_table.sort_values("rmse", ascending=False)
fig, ax = plt.subplots(figsize=(8, 3.6))
bars = ax.barh(t.index, t["rmse"] * 100, color=BLUE, height=0.55)
for bar, (lab, row) in zip(bars, t.iterrows()):
    note = "" if lab == "Persistence" else f"   ({row['vs_persistence']:+.0%} vs persistence)"
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{row['rmse'] * 100:.2f}{note}", va="center", fontsize=9, color=INK)
ax.set_xlim(0, t["rmse"].max() * 100 * 1.45)
ax.set_xlabel("Weighted RMSE, K% points (lower is better)")
ax.set_title("Baseline forecast error on 2025–2026 test seasons")
ax.grid(axis="y", visible=False)
save(fig, "07_baseline_test_rmse.png")
 
# RMSE by target season
years = list(range(int(by_year.index.min()), int(by_year.index.max()) + 1))
by_year_full = by_year.reindex(years) * 100      # missing 2020/2021 targets show as gaps
fig, ax = plt.subplots(figsize=(9, 5))
for (lab, col), color in zip(METHODS.items(), [ORANGE, BLUE, AQUA]):
    s = by_year_full[lab]
    ax.plot(s.index, s.values, color=color, marker="o", markersize=7, label=lab)
    last = s.dropna()
    ax.text(last.index[-1] + 0.25, last.values[-1], lab, color=INK, fontsize=9, va="center")
first_test = min(TEST_SEASONS) + 1
ax.axvspan(first_test - 0.5, years[-1] + 0.5, color=GRID, alpha=0.5, zorder=0)
ax.text(first_test - 0.4, ax.get_ylim()[1], "Test", color=INK_2, fontsize=9, va="top")
ax.set_xticks(years)
ax.set_xlim(years[0] - 0.5, years[-1] + 2.2)
ax.set_xlabel("Season being forecast (2020 and 2021 excluded)")
ax.set_ylabel("Weighted RMSE, K% points")
ax.set_title("Forecast error by season")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3)
save(fig, "08_baseline_rmse_by_year.png")
 
