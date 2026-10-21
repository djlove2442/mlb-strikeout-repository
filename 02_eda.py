# 02 Exploratory Data Analysis


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
 
MIN_BF = 100

# Chart Style

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#e8e7e3"
BLUE = "#2a78d6"      
ORANGE = "#eb6834"    

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "figure.dpi": 110,
})

def save(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")
 
 
def pct(x):
    return f"{x:.1%}"


# Load Data
con = sqlite3.connect(DB_PATH)
seasons = pd.read_sql("SELECT * FROM pitcher_seasons", con)
pairs = pd.read_sql("SELECT * FROM model_pairs", con)
con.close()
 
# Leave out pairs that touch the 60-game 2020 season for the EDA
pairs = pairs[pairs["involves_2020"] == 0].copy()
 
print(f"Pitcher-seasons: {len(seasons):,}")
print(f"Season pairs used for EDA (2020 excluded): {len(pairs):,}")
print(pairs[["bf", "k_pct", "bb_pct", "whiff_pct", "csw_pct", "fb_velo",
             "ff_spin", "fip", "next_k_pct"]].describe().round(3))


# 1. Year-to-year correlation (stickiness) of each metric
print("\n1. Year-to-year correlation (stickiness) by metric")
 
metrics = {
    "K%": "k_pct",
    "BB%": "bb_pct",
    "K-BB%": "k_minus_bb",
    "Whiff%": "whiff_pct",
    "CSW%": "csw_pct",
    "Fastball velocity": "fb_velo",
    "Four-seam spin": "ff_spin",
    "FIP": "fip",
}
 
nxt = seasons.copy()
nxt["season"] = nxt["season"] - 1
yoy = seasons.merge(nxt, on=["player_id", "season"], suffixes=("", "_next"))
yoy = yoy[(yoy["bf"] >= MIN_BF) & (yoy["bf_next"] >= MIN_BF)]
yoy = yoy[~yoy["season"].isin([2019, 2020])]
 
stick = []
for label, col in metrics.items():
    d = yoy[[col, col + "_next"]].dropna()
    stick.append({"metric": label, "r": d[col].corr(d[col + "_next"]), "n": len(d)})
stick = pd.DataFrame(stick).sort_values("r")
print(stick.sort_values("r", ascending=False).round(3).to_string(index=False))
 
fig, ax = plt.subplots(figsize=(8, 4.8))
bars = ax.barh(stick["metric"], stick["r"], color=BLUE, height=0.6)
for bar, r in zip(bars, stick["r"]):
    ax.text(r + 0.01, bar.get_y() + bar.get_height() / 2, f"{r:.2f}",
            va="center", fontsize=9, color=INK)
ax.set_xlim(0, 1.05)
ax.set_xlabel("Correlation between season t and season t+1")
ax.set_title("Which pitcher metrics carry over from year to year?")
ax.grid(axis="y", visible=False)
save(fig, "01_stickiness.png")


# 2. This year's K% vs next year's K%
print("\n2. K% this season vs next season")
 
r_k = pairs["k_pct"].corr(pairs["next_k_pct"])
slope, intercept = np.polyfit(pairs["k_pct"], pairs["next_k_pct"], 1)
print(f"  r = {r_k:.3f}; fitted line: next = {intercept:.3f} + {slope:.3f} * current")
print("  A slope below 1 is regression to the mean: extreme K% pitchers tend to move "
      "back toward average.")
 
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(pairs["k_pct"], pairs["next_k_pct"], s=12, alpha=0.3, color=BLUE,
           edgecolors="none")
lims = [0.05, 0.45]
ax.plot(lims, lims, color=INK_2, linewidth=1.2, linestyle="--")
xs = np.array(lims)
ax.plot(xs, intercept + slope * xs, color=ORANGE, linewidth=2)
ax.text(0.40, 0.425, "Next year = this year", color=INK_2, fontsize=9, ha="right")
ax.text(0.44, intercept + slope * 0.44 - 0.025, f"Fitted (slope {slope:.2f})",
        color=INK, fontsize=9, ha="right")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("K% in season t")
ax.set_ylabel("K% in season t+1")
ax.set_title(f"K% carries over, but regresses (r = {r_k:.2f})")
save(fig, "02_k_pct_year_to_year.png")



# 3. Noise vs sample size: why shrinkage is needed
print("\n3. Year-to-year change in K% by sample size")
 
pairs["k_change"] = pairs["next_k_pct"] - pairs["k_pct"]
bins = [100, 200, 300, 400, 500, 600, 700, 800, 1200]
pairs["bf_bin"] = pd.cut(pairs["bf"], bins=bins, right=False)
by_bin = (pairs.groupby("bf_bin", observed=True)
          .agg(pitchers=("k_change", "size"),
               mean_abs_change=("k_change", lambda s: s.abs().mean()),
               bf_mid=("bf", "median"))
          .reset_index())
print(by_bin.round(4).to_string(index=False))
 
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(pairs["bf"], pairs["k_change"], s=12, alpha=0.25, color=BLUE,
           edgecolors="none", label="Individual pitcher")
ax.plot(by_bin["bf_mid"], by_bin["mean_abs_change"], color=ORANGE, marker="o",
        markersize=7, label="Average size of change (absolute)")
ax.plot(by_bin["bf_mid"], -by_bin["mean_abs_change"], color=ORANGE, marker="o",
        markersize=7)
ax.axhline(0, color=INK_2, linewidth=1)
ax.set_xlim(90, 1100)
ax.set_ylim(-0.2, 0.2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
ax.set_xlabel("Batters faced in season t")
ax.set_ylabel("Change in K% (next season minus this season)")
ax.set_title("Small samples swing the most from year to year")
ax.legend(loc="upper right")
save(fig, "03_noise_vs_sample_size.png")


# 4. Starters vs Relievers
print("\n4. Starters vs relievers")
 
role_summary = (pairs.groupby("role")
                .agg(pitchers=("k_pct", "size"),
                     median_bf=("bf", "median"),
                     mean_k_pct=("k_pct", "mean"),
                     sd_k_pct=("k_pct", "std"),
                     yoy_r=("k_pct", lambda s: s.corr(pairs.loc[s.index, "next_k_pct"])))
                .round(3))
print(role_summary)
print("  Relievers usually strike out more batters on smaller samples, so they may need "
      "their own prior in Phase 4.")
 
fig, ax = plt.subplots(figsize=(8, 4.8))
k_bins = np.arange(0.05, 0.46, 0.01)
for role, color, label in [("SP", BLUE, "Starters"), ("RP", ORANGE, "Relievers")]:
    d = pairs.loc[pairs["role"] == role, "k_pct"]
    ax.hist(d, bins=k_bins, density=True, histtype="stepfilled", alpha=0.25, color=color)
    ax.hist(d, bins=k_bins, density=True, histtype="step", linewidth=2, color=color,
            label=f"{label} (n = {len(d):,}, mean {pct(d.mean())})")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("K% in season t")
ax.set_ylabel("Density")
ax.set_title("Relievers strike out more batters than starters")
ax.legend(loc="upper right")
ax.grid(axis="x", visible=False)
save(fig, "04_starters_vs_relievers.png")


# 5. Correlation matrix of candidate features
print("\n5. Feature correlation matrix")
 
features = {
    "K%": "k_pct", "BB%": "bb_pct", "Whiff%": "whiff_pct", "CSW%": "csw_pct",
    "FB velo": "fb_velo", "FF spin": "ff_spin", "Velo change": "velo_change",
    "Age": "age", "Next K%": "next_k_pct",
}
corr = pairs[list(features.values())].corr()
corr.index = corr.columns = list(features.keys())
print(corr.round(2))
 
diverging = LinearSegmentedColormap.from_list("div", [ORANGE, NEUTRAL, BLUE])
fig, ax = plt.subplots(figsize=(8, 6.8))
im = ax.imshow(corr.values, cmap=diverging, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.index)
ax.grid(False)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color=INK)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Correlation")
ax.set_title("How the candidate features relate")
save(fig, "05_feature_correlations.png")
 
high = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.index)
        for b in corr.index[i + 1:] if abs(corr.loc[a, b]) >= 0.7 and "Next" not in a + b]
if high:
    print("  Feature pairs with |r| >= 0.7 (watch for multicollinearity):")
    for a, b, r in high:
        print(f"    {a} & {b}: {r:.2f}")


# 6. League K% over time
print("\n6. League strikeout rate by season")
 
league = (seasons.groupby("season")
          .apply(lambda d: d["k"].sum() / d["bf"].sum(), include_groups=False)
          .rename("league_k_pct"))
print(league.map(pct).to_string())
 
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(league.index, league.values, color=BLUE, marker="o", markersize=7)
ax.axvline(2022.5, color=INK_2, linewidth=1, linestyle="--")
ax.text(2022.6, league.max(), "2023 rule changes", color=INK_2, fontsize=9, va="top")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
ax.set_xticks(league.index)
ax.tick_params(axis="x", rotation=45)
ax.set_xlabel("Season")
ax.set_ylabel("League K%")
ax.set_title("League strikeout rate by season")
save(fig, "06_league_k_pct.png")
 
 
