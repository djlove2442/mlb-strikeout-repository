# 05 Evalutaion and Error Analysis
# Questions this answers:
#   1. Is the hierarchical model's edge over Marcel real, or could it be luck?
#      (bootstrap that resamples PITCHERS, since most appear in both test seasons)
#   2. Why does every method over-forecast recent seasons?
#      (split the bias into "league environment changed" vs "these pitchers declined")
#   3. Are the point forecasts calibrated across the K% range, and do the intervals
#      hold up for starters and relievers separately?
#   4. Who did the model miss, and could anything known at forecast time have caught it?

import sqlite3
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "baseball.db"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
 
N_BOOT = 5000
SEED = 7
MISS_MIN_BF = 300

# Same Chart Style
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
 
 
def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))
 
 
def rmse(df, col):
    return np.sqrt(wmean((df[col] - df["next_k_pct"]) ** 2, df["next_bf"]))
 
 
def bias(df, col):
    return wmean(df[col] - df["next_k_pct"], df["next_bf"])
 
 
def pts(x):
    return f"{x * 100:+.2f}"
 
 
METHODS = {
    "Persistence": "pred_persist",
    "Marcel": "pred_marcel",
    "EB 3-season (tuned)": "pred_eb3_tuned",
    "Hierarchical": "pred_hier",
}

# Load everything into one table
con = sqlite3.connect(DB_PATH)
seasons = pd.read_sql("SELECT * FROM pitcher_seasons", con)
pairs = pd.read_sql("SELECT player_id, season, next_role, velo_change, fb_velo "
                    "FROM model_pairs WHERE involves_2020 = 0", con)
base = pd.read_sql("SELECT * FROM baseline_forecasts", con)
eb = pd.read_sql("SELECT player_id, season, pred_eb3_tuned FROM eb_forecasts", con)
hier = pd.read_sql("SELECT player_id, season, pred_hier, lo50, hi50, lo80, hi80, lo95, hi95 "
                   "FROM hier_forecasts", con)
 
df = (base.merge(eb, on=["player_id", "season"])
          .merge(hier, on=["player_id", "season"])
          .merge(pairs, on=["player_id", "season"], how="left"))
 
# next season's fastball velocity (NOT known at forecast time; used only to explain misses)
nxt_velo = seasons[["player_id", "season", "fb_velo"]].rename(columns={"fb_velo": "next_fb_velo"})
nxt_velo["season"] -= 1
df = df.merge(nxt_velo, on=["player_id", "season"], how="left")
df["next_velo_change"] = df["next_fb_velo"] - df["fb_velo"]
df["role_changed"] = df["next_role"].notna() & (df["next_role"] != df["role"])
 
league = seasons.groupby("season")[["k", "bf"]].sum()
league_k = league["k"] / league["bf"]
df["lg_t"] = df["season"].map(league_k)
df["lg_next"] = df["target_season"].map(league_k)
 
test = df[df["split"] == "test"].copy()
print(f"Test pairs: {len(test):,} from {test['player_id'].nunique():,} pitchers "
      f"(seasons forecast: {', '.join(str(int(s)) for s in sorted(test['target_season'].unique()))})")
 
summary_rows = []

# 1. Is the edge over Marcel real? Pitcher-level (cluster) bootstrap
print("\n1. Bootstrap: improvement in RMSE vs Marcel on TEST pairs")
print(f"   ({N_BOOT:,} resamples of pitchers; each resample keeps all of a pitcher's test pairs)")
 
rng = np.random.default_rng(SEED)
pitcher_codes, pitcher_idx = np.unique(test["player_id"], return_inverse=True)
n_p = len(pitcher_codes)
w = test["next_bf"].to_numpy()
 
# per-pitcher sums of weight and weighted squared error, so each resample is fast
w_by_p = np.bincount(pitcher_idx, weights=w, minlength=n_p)
se_by_p = {lab: np.bincount(pitcher_idx,
                            weights=w * (test[col] - test["next_k_pct"]).to_numpy() ** 2,
                            minlength=n_p)
           for lab, col in METHODS.items()}
 
counts = np.stack([np.bincount(rng.integers(0, n_p, n_p), minlength=n_p) for _ in range(N_BOOT)])
boot_w = counts @ w_by_p
boot_rmse = {lab: np.sqrt((counts @ se) / boot_w) for lab, se in se_by_p.items()}
 
boot_table = []
for lab in METHODS:
    if lab == "Marcel":
        continue
    imp = 1 - boot_rmse[lab] / boot_rmse["Marcel"]
    point = 1 - rmse(test, METHODS[lab]) / rmse(test, "pred_marcel")
    lo, hi = np.percentile(imp, [2.5, 97.5])
    boot_table.append({"model": lab, "improvement": point, "ci_low": lo, "ci_high": hi,
                       "p_better": float(np.mean(imp > 0))})
boot_table = pd.DataFrame(boot_table).set_index("model")
show = boot_table.copy()
for c in ["improvement", "ci_low", "ci_high"]:
    show[c] = show[c].map(lambda v: f"{v:+.1%}")
show["p_better"] = show["p_better"].map(lambda v: f"{v:.1%}")
show.columns = ["RMSE vs Marcel", "95% CI low", "95% CI high", "Share of resamples better"]
print(show.to_string())
 
h = boot_table.loc["Hierarchical"]
if h["ci_low"] > 0:
    verdict = "the interval is entirely above zero: the improvement is unlikely to be luck."
elif h["ci_high"] < 0:
    verdict = "the interval is entirely below zero: the model is reliably WORSE than Marcel."
else:
    verdict = "the interval includes zero: the difference could be luck at this sample size."
print(f"\n   Hierarchical vs Marcel: {h['improvement']:+.1%} "
      f"(95% CI {h['ci_low']:+.1%} to {h['ci_high']:+.1%}); {verdict}")
print("   Caveat: only two test seasons, so this shows uncertainty across pitchers,\n"
      "   not across seasons with different league conditions.")
 
summary_rows += [
    ("hier_vs_marcel_improvement", h["improvement"]),
    ("hier_vs_marcel_ci_low", h["ci_low"]),
    ("hier_vs_marcel_ci_high", h["ci_high"]),
    ("hier_vs_marcel_p_better", h["p_better"]),
]

# 2. Where does the bias come from
print("\n2. Forecast bias by season (pts; positive = forecast too high)")
print("   Note: the hierarchical model's training seasons (through 2024) are in-sample.")
 
rows = []
for yr, d in df.groupby("target_season"):
    wts = d["next_bf"]
    lg_part = float(d["lg_t"].iloc[0] - d["lg_next"].iloc[0])
    rel_part = wmean((d["k_pct"] - d["lg_t"]) - (d["next_k_pct"] - d["lg_next"]), wts)
    row = {"season": int(yr), "pairs": len(d)}
    for lab, col in METHODS.items():
        row[lab] = bias(d, col)
    row["league_change"] = float(d["lg_next"].iloc[0] - d["lg_t"].iloc[0])
    row["persist_from_league"] = lg_part
    row["persist_from_pitchers"] = rel_part
    rows.append(row)
by_season = pd.DataFrame(rows).set_index("season")
 
show = by_season.copy()
for c in show.columns:
    if c != "pairs":
        show[c] = show[c].map(pts)
show = show.rename(columns={"league_change": "League K% change",
                            "persist_from_league": "Persistence bias: league part",
                            "persist_from_pitchers": "Persistence bias: pitcher part"})
print(show.to_string())
print("\n   League part  = how much league K% fell (positive) or rose (negative) that year.")
print("   Pitcher part = how much returning pitchers lost ground relative to the league.")
 
pre = by_season.loc[by_season.index <= 2019]
post = by_season.loc[by_season.index >= 2022]
for label, d in [("2016-2019", pre), ("2022-2026", post)]:
    print(f"   {label}: persistence bias {pts(d['Persistence'].mean())} = "
          f"league {pts(d['persist_from_league'].mean())} + "
          f"pitchers {pts(d['persist_from_pitchers'].mean())}")
 
# What if the model had known next season's league K% in advance? (an oracle check, not a
# usable forecast; it shows how much error is league environment rather than pitchers)
test["pred_hier_oracle"] = test["pred_hier"] * test["lg_next"] / test["lg_t"]
r_h, r_o = rmse(test, "pred_hier"), rmse(test, "pred_hier_oracle")
print(f"\n   Oracle check: rescaling hierarchical forecasts by the ACTUAL league K% change")
print(f"   moves test RMSE {r_h * 100:.2f} -> {r_o * 100:.2f} and bias "
      f"{pts(bias(test, 'pred_hier'))} -> {pts(bias(test, 'pred_hier_oracle'))} pts.")
print("   (Uses future information, so it is a diagnostic, not a model.)")
 
summary_rows += [
    ("test_bias_hier", bias(test, "pred_hier")),
    ("test_bias_hier_oracle", bias(test, "pred_hier_oracle")),
    ("test_rmse_hier", r_h),
    ("test_rmse_hier_oracle", r_o),
    ("post2022_persist_bias_league_part", post["persist_from_league"].mean()),
    ("post2022_persist_bias_pitcher_part", post["persist_from_pitchers"].mean()),
]

# Calibration: point forecasts across the K% range, intervals by role
print("\n3a. Point-forecast calibration on TEST pairs (10 bins by forecast K%)")
test["bin"] = pd.qcut(test["pred_hier"], 10, labels=False, duplicates="drop")
calib = (test.groupby("bin")
         .apply(lambda d: pd.Series({
             "pairs": len(d),
             "forecast": wmean(d["pred_hier"], d["next_bf"]),
             "actual": wmean(d["next_k_pct"], d["next_bf"])}), include_groups=False))
calib["gap"] = calib["actual"] - calib["forecast"]
show = calib.copy()
show["pairs"] = show["pairs"].astype(int)
for c in ["forecast", "actual"]:
    show[c] = show[c].map("{:.1%}".format)
show["gap"] = show["gap"].map(pts)
print(show.to_string())
slope = np.polyfit(calib["forecast"], calib["actual"], 1)[0]
print(f"   Slope of actual on forecast across bins: {slope:.2f} "
      f"(1.00 = forecasts spread out exactly the right amount; "
      f"<1 = too spread out; >1 = too compressed)")
summary_rows.append(("calibration_slope", slope))
 
print("\n3b. Interval coverage on TEST pairs by role")
for role, d in test.groupby("role"):
    cov = {lvl: np.mean((d["next_k_pct"] >= d[f"lo{lvl}"]) & (d["next_k_pct"] <= d[f"hi{lvl}"]))
           for lvl in (50, 80, 95)}
    print(f"   {role}: 50% -> {cov[50]:.1%}   80% -> {cov[80]:.1%}   95% -> {cov[95]:.1%}"
          f"   (n = {len(d)})")

# 4. Who did the model miss?
print("\n4. Biggest misses on TEST pairs "
      f"({MISS_MIN_BF}+ batters faced the following season)")
test["error"] = test["pred_hier"] - test["next_k_pct"]
big = test[test["next_bf"] >= MISS_MIN_BF].copy()
cols = ["name", "target_season", "role", "next_role", "k_pct", "pred_hier", "next_k_pct",
        "error", "next_velo_change"]
fmt = {"k_pct": "{:.1%}".format, "pred_hier": "{:.1%}".format,
       "next_k_pct": "{:.1%}".format, "error": lambda v: f"{v * 100:+.1f}",
       "next_velo_change": lambda v: "n/a" if pd.isna(v) else f"{v:+.1f}"}
headers = ["Pitcher", "Season", "Role", "Next role", "Prior K%", "Forecast", "Actual",
           "Error (pts)", "Velo change (mph)"]
for title, part in [("   Forecast too HIGH (pitcher struck out fewer than expected)",
                     big.nlargest(10, "error")),
                    ("   Forecast too LOW (pitcher struck out more than expected)",
                     big.nsmallest(10, "error"))]:
    print(f"\n{title}")
    t = part[cols].copy()
    t.columns = headers
    print(t.to_string(index=False, formatters=dict(zip(headers, [fmt.get(c) for c in cols]))))
 
print("\n   Do things we could NOT know at forecast time explain the misses?")
d = test.dropna(subset=["next_velo_change"])
r_velo = np.corrcoef(d["next_velo_change"], -d["error"])[0, 1]
print(f"   Correlation between next-season velocity change and (actual - forecast): {r_velo:.2f}")
rc = test.groupby("role_changed").apply(
    lambda d: pd.Series({"pairs": len(d), "rmse": rmse(d, "pred_hier"),
                         "bias": bias(d, "pred_hier")}), include_groups=False)
for changed, row in rc.iterrows():
    label = "changed role" if changed else "same role   "
    print(f"   Pitchers who {label}: n = {int(row['pairs']):>3}, RMSE {row['rmse'] * 100:.2f}, "
          f"bias {pts(row['bias'])}")
summary_rows.append(("corr_next_velo_change_with_miss", r_velo))

# Summary
pd.DataFrame(summary_rows, columns=["metric", "value"]).to_sql(
    "evaluation_summary", con, if_exists="replace", index=False)
con.close()

# Charts
print("\nCharts")
 
# 1) bootstrap intervals
bt = boot_table.sort_values("improvement")
fig, ax = plt.subplots(figsize=(8, 3.6))
y = np.arange(len(bt))
ax.hlines(y, bt["ci_low"] * 100, bt["ci_high"] * 100, color=BLUE, linewidth=2.5)
ax.plot(bt["improvement"] * 100, y, "o", color=BLUE, markersize=9)
for yi, (lab, row) in zip(y, bt.iterrows()):
    ax.text(row["ci_high"] * 100 + 0.4, yi,
            f"{row['improvement']:+.1%}  ({row['ci_low']:+.1%} to {row['ci_high']:+.1%})",
            va="center", fontsize=9, color=INK)
ax.axvline(0, color=INK_2, linewidth=1)
ax.set_yticks(y, bt.index)
ax.set_ylim(-0.6, len(bt) - 0.4)
xmin = min(bt["ci_low"].min() * 100, 0) - 2
ax.set_xlim(xmin, bt["ci_high"].max() * 100 + 12)
ax.set_xlabel("RMSE improvement over Marcel, % (95% bootstrap interval)")
ax.set_title("Improvement over Marcel on 2025–26, with uncertainty")
ax.grid(axis="y", visible=False)
save(fig, "16_bootstrap_vs_marcel.png")
 
# 2) bias by season vs league change
years = list(range(int(by_season.index.min()), int(by_season.index.max()) + 1))
b = by_season.reindex(years) * 100
fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(0, color=INK_2, linewidth=1)
series = [("Hierarchical", "Hierarchical forecast bias", BLUE),
          ("Persistence", "Persistence forecast bias", AQUA),
          ("persist_from_league", "League K% drop (league part of bias)", ORANGE)]
for col, label, color in series:
    s = b[col]
    ax.plot(s.index, s.values, color=color, marker="o", markersize=7, label=label)
ax.set_xticks(years)
ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
ax.set_xlabel("Season being forecast (2020 and 2021 excluded)")
ax.set_ylabel("K% points (positive = forecast too high)")
ax.set_title("Forecast bias by season, next to the league K% change")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
save(fig, "17_bias_by_season.png")
 
# 3) point-forecast calibration
fig, ax = plt.subplots(figsize=(6, 5.6))
lo_lim = min(calib["forecast"].min(), calib["actual"].min()) - 0.01
hi_lim = max(calib["forecast"].max(), calib["actual"].max()) + 0.01
ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color=INK_2, linewidth=1.2, linestyle="--")
ax.text(hi_lim - 0.003, hi_lim - 0.012, "Perfect calibration", color=INK_2, fontsize=9,
        ha="right")
ax.plot(calib["forecast"], calib["actual"], color=BLUE, marker="o", markersize=8)
ax.set_xlim(lo_lim, hi_lim)
ax.set_ylim(lo_lim, hi_lim)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Forecast K% (average within bin)")
ax.set_ylabel("Actual K% (average within bin)")
ax.set_title("Forecast vs actual, 2025–26 test seasons")
save(fig, "18_forecast_calibration.png")
 
 
