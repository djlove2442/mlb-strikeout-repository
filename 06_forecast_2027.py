# 06 2027 Strikeout Forecasts
# Refits the final model (hierarchical + era term, no pitcher effect) on EVERY season pair
# through 2025 -> 2026, then forecasts 2027 strikeout rate for every pitcher with 100+
# batters faced in 2026.
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
    """Same ArviZ compatibility patch as Phase 4b (fixes the '_TEXT_OR_BYTES' crash)."""
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
 
FORECAST_FROM = 2026          # last completed season
ERA_START_SEASON = 2021       # same era definition as Phase 4b
MIN_BF = 100                  # pitchers forecast: 100+ batters faced in 2026
HISTORY_WEIGHTS = {0: 1.0, 1: 0.8, 2: 0.6}
STUFF_FEATURES = ["whiff_pct", "fb_velo", "velo_change", "ff_spin", "spin_missing"]
AGE_FEATURES = ["age", "age_sq"]
STUFF_MIN_BF = {"SP": 300, "RP": 150}     
 
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

# Same Chart Styles
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
 
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
 
 
def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))
 
 
def pct(v):
    return f"{v:.1%}"
 
# Data
def load_training():
    """Every scored season pair (train + test), with the same raw features as Phase 4b."""
    con = sqlite3.connect(DB_PATH)
    pairs = pd.read_sql("SELECT * FROM model_pairs WHERE involves_2020 = 0", con)
    eb = pd.read_sql("SELECT player_id, season, pred_eb3, age_factor FROM eb_forecasts", con)
    con.close()
    df = pairs.merge(eb, on=["player_id", "season"], how="inner")
    df["eb_rate"] = df["pred_eb3"] / df["age_factor"]       # EB 3-season K%, no age adjustment
    df["next_k"] = df["next_k"].round().astype(int)
    df["next_bf"] = df["next_bf"].round().astype(int)
    return df
 
 
def load_forecast_pool():
    """Every pitcher with MIN_BF+ batters faced in FORECAST_FROM, with features built the
    same way as the training pairs (3-season EB K% uses that season's role prior)."""
    con = sqlite3.connect(DB_PATH)
    seasons = pd.read_sql("SELECT * FROM pitcher_seasons", con)
    priors = pd.read_sql("SELECT * FROM eb_priors", con)
    con.close()
 
    pool = seasons[(seasons["season"] == FORECAST_FROM) & (seasons["bf"] >= MIN_BF)].copy()
 
    prev = seasons[["player_id", "season", "fb_velo"]].rename(columns={"fb_velo": "prev_velo"})
    prev["season"] += 1
    pool = pool.merge(prev, on=["player_id", "season"], how="left")
    pool["velo_change"] = pool["fb_velo"] - pool["prev_velo"]
 
    wk = np.zeros(len(pool))
    wbf = np.zeros(len(pool))
    hist = seasons[["player_id", "season", "k", "bf"]]
    for lag, w in HISTORY_WEIGHTS.items():
        h = hist.copy()
        h["season"] += lag
        h = pool[["player_id", "season"]].merge(h, on=["player_id", "season"], how="left")
        wk += w * h["k"].fillna(0).to_numpy()
        wbf += w * h["bf"].fillna(0).to_numpy()
    pool = pool.merge(priors[["season", "role", "alpha", "beta"]], on=["season", "role"],
                      how="left")
    pool["eb_rate"] = (wk + pool["alpha"]) / (wbf + pool["alpha"] + pool["beta"])
    return pool, seasons
 
 
def raw_features(d):
    f = pd.DataFrame(index=d.index)
    f["eb_k_logit"] = logit(d["eb_rate"])
    f["whiff_pct"] = d["whiff_pct"]
    f["bb_pct"] = d["bb_pct"]
    f["fb_velo"] = d["fb_velo"]
    f["velo_change"] = d["velo_change"].fillna(0)
    f["spin_missing"] = d["ff_spin"].isna().astype(float)
    f["ff_spin"] = d["ff_spin"]
    f["age"] = d["age"]
    return f



def expected_k(seasons, pool):
    """xK%: the strikeout rate a pitcher's stuff would normally produce.
    Weighted regression of logit(K%) on whiff rate, fastball velocity and four-seam spin
    (plus a no-four-seamer flag), fit separately for starters and relievers on every
    pitcher-season with MIN_BF+ batters faced. Returns xK% for the pool and the R-squared."""
    cols = ["whiff_pct", "fb_velo", "ff_spin"]
    fit_rows = seasons[(seasons["bf"] >= MIN_BF) & (seasons["season"] <= FORECAST_FROM)]
    fit_rows = fit_rows.dropna(subset=["whiff_pct", "fb_velo", "k_pct"])
    xk = pd.Series(np.nan, index=pool.index)
    r2 = {}
    for role in ROLES:
        tr = fit_rows[fit_rows["role"] == role]
        means = tr[cols].mean()

        def design(d):
            return np.column_stack([np.ones(len(d)), d[cols].fillna(means).to_numpy(),
                                    d["ff_spin"].isna().astype(float).to_numpy()])

        w = np.sqrt(tr["bf"].to_numpy())
        coef, *_ = np.linalg.lstsq(design(tr) * w[:, None], logit(tr["k_pct"].to_numpy()) * w,
                                   rcond=None)
        fitted = 1 / (1 + np.exp(-(design(tr) @ coef)))
        r2[role] = np.corrcoef(fitted, tr["k_pct"])[0, 1] ** 2
        target = pool["role"] == role
        xk[target] = 1 / (1 + np.exp(-(design(pool[target]) @ coef)))
    return xk, r2 
 
def standardize(train_raw, other_raw):
    """Fill and scale both tables using TRAINING statistics only."""
    fill = train_raw.mean()
    tr, ot = train_raw.fillna(fill), other_raw.fillna(fill)
    mean, sd = tr.mean(), tr.std().replace(0, 1)
    tr, ot = (tr - mean) / sd, (ot - mean) / sd
    tr["age_sq"], ot["age_sq"] = tr["age"] ** 2, ot["age"] ** 2
    m2, s2 = tr["age_sq"].mean(), tr["age_sq"].std()
    tr["age_sq"], ot["age_sq"] = (tr["age_sq"] - m2) / s2, (ot["age_sq"] - m2) / s2
    return tr[FEATURES].to_numpy(), ot[FEATURES].to_numpy()
 
# Model (identical to final Phase 4b model
def sample_posterior(X, role_idx, next_bf, next_k, era):
    coords = {"role": ROLES, "feature": FEATURES, "obs": np.arange(len(next_k))}
    with pm.Model(coords=coords):
        a = pm.Normal("a_role", mu=logit(0.22), sigma=0.5, dims="role")
        b = pm.Normal("b", mu=0, sigma=0.5, dims="feature")
        sigma_season = pm.HalfNormal("sigma_season", sigma=0.3)
        z_season = pm.Normal("z_season", 0, 1, dims="obs")
        era_effect = pm.Normal("era_effect", mu=0, sigma=0.3)
        eta = (a[role_idx] + pm.math.dot(X, b) + era_effect * era
               + z_season * sigma_season)
        pm.Binomial("next_k", n=next_bf, logit_p=eta, observed=next_k, dims="obs")
 
        kwargs = dict(draws=DRAWS, tune=TUNE, chains=CHAINS, target_accept=0.9,
                      random_seed=SEED)
        names = ["a_role", "b", "sigma_season", "era_effect"]
        if SAMPLER == "nutpie":
            idata = pm.sample(nuts_sampler="nutpie", **kwargs)
            draws = {n: np.asarray(idata.posterior[n]) for n in names}
            divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
        else:
            trace = pm.sample(cores=1, return_inferencedata=False,
                              compute_convergence_checks=False, **kwargs)
            draws = {n: np.stack(trace.get_values(n, combine=False)) for n in names}
            divergences = int(sum(np.sum(x) for x in
                                  trace.get_sampler_stats("diverging", combine=False)))
    return draws, divergences
 
 
def split_rhat(x):
    half = x.shape[1] // 2
    chains = np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)
    n = chains.shape[1]
    w = chains.var(axis=1, ddof=1).mean()
    b = n * chains.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + b / n) / w)) if w > 0 else np.nan
 
# Main
def main():
    rng = np.random.default_rng(SEED)
    train = load_training()
    pool, seasons = load_forecast_pool()
    print(f"Training pairs: {len(train):,} (every scored pair through "
          f"{int(train['season'].max())} -> {int(train['season'].max()) + 1})")
    print(f"Pitchers to forecast: {len(pool):,} ({MIN_BF}+ batters faced in {FORECAST_FROM}; "
          f"{(pool['role'] == 'SP').sum()} starters, {(pool['role'] == 'RP').sum()} relievers)")
 
    X_train, X_pool = standardize(raw_features(train), raw_features(pool))
    role_map = {r: i for i, r in enumerate(ROLES)}
    era_train = (train["season"] >= ERA_START_SEASON).astype(float).to_numpy()
 
    print(f"\nSampling ({CHAINS} chains x {DRAWS} draws, sampler = {SAMPLER}) ...")
    t0 = time.time()
    draws, divergences = sample_posterior(
        X_train, train["role"].map(role_map).to_numpy(),
        train["next_bf"].to_numpy(), train["next_k"].to_numpy(), era_train)
    print(f"  done in {(time.time() - t0) / 60:.1f} minutes")
 
    # diagnostics 
    rhats = [split_rhat(draws["a_role"][:, :, i]) for i in range(len(ROLES))]
    rhats += [split_rhat(draws["b"][:, :, i]) for i in range(len(FEATURES))]
    rhats += [split_rhat(draws["sigma_season"]), split_rhat(draws["era_effect"])]
    print(f"\nConvergence: largest R-hat {max(rhats):.3f}, divergences {divergences}")
    if max(rhats) > 1.01 or divergences > 0:
        print("  WARNING: check convergence before trusting these forecasts "
              "(try TUNE = 2000).")
 
    d = {k: v.reshape(-1, *v.shape[2:]) for k, v in draws.items()}
    k_scale = 0.22 * 0.78 * 100
    print(f"  era effect: {d['era_effect'].mean() * k_scale:+.2f} K% points "
          f"(Phase 4b estimate was about -1.1)")
    coef = pd.Series(d["b"].mean(axis=0), index=[FEATURE_LABELS[f] for f in FEATURES])
    print("  coefficients, K% points per 1 SD:  " +
          ", ".join(f"{k} {v * k_scale:+.1f}" for k, v in coef.items()))
 
    # forecasts for 2027 
    role_idx = pool["role"].map(role_map).to_numpy()
    S, n = d["b"].shape[0], len(pool)
    eta = d["a_role"][:, role_idx] + d["b"] @ X_pool.T + d["era_effect"][:, None]   # era = 1
    e = rng.normal(size=(S, n)) * d["sigma_season"][:, None]
    p = 1 / (1 + np.exp(-(eta + e)))
    bf_proj = pool["bf"].to_numpy().astype(int)
    sims = rng.binomial(bf_proj[None, :], p) / bf_proj[None, :]
 
    # how much the skill metrics (everything except past K%) move each forecast
    eta_metrics = (d["b"][:, 1:] @ X_pool[:, 1:].T).mean(axis=0)
 
    pool["forecast_2027"] = p.mean(axis=0)
    pool["talent_lo80"], pool["talent_hi80"] = np.quantile(p, [0.1, 0.9], axis=0)
    pool["outcome_lo80"], pool["outcome_hi80"] = np.quantile(sims, [0.1, 0.9], axis=0)
    pool["change_vs_2026"] = pool["forecast_2027"] - pool["k_pct"]
    pool["metrics_adj_pts"] = eta_metrics * pool["forecast_2027"] * (1 - pool["forecast_2027"]) * 100

    # expected K% from stuff alone (xK%), and its gap to the actual 2026 K%
    xk, xk_r2 = expected_k(seasons, pool)
    pool["xk_pct"] = xk
    pool["stuff_gap"] = pool["xk_pct"] - pool["k_pct"]

    # tables 
    def table(frame, cols, headers, fmts):
        t = frame[cols].copy()
        t.columns = headers
        print(t.to_string(index=False, formatters=dict(zip(headers, fmts)), na_rep="n/a"))
 
    main_cols = ["name", "age", "bf", "k_pct", "forecast_2027", "talent_lo80", "talent_hi80"]
    main_heads = ["Pitcher", "Age", "2026 BF", "2026 K%", "2027 forecast",
                  "80% low", "80% high"]
    main_fmts = [None, lambda v: f"{v:.0f}", lambda v: f"{v:.0f}", pct, pct, pct, pct]
 
    print("\nTop 15 starters by 2027 forecast K% (400+ BF in 2026)")
    sp = pool[(pool["role"] == "SP") & (pool["bf"] >= 400)].nlargest(15, "forecast_2027")
    table(sp, main_cols, main_heads, main_fmts)
 
    print("\nTop 10 relievers by 2027 forecast K% (150+ BF in 2026)")
    rp = pool[(pool["role"] == "RP") & (pool["bf"] >= 150)].nlargest(10, "forecast_2027")
    table(rp, main_cols, main_heads, main_fmts)
 
    cand_cols = ["name", "role", "age", "k_pct", "forecast_2027", "change_vs_2026",
                 "metrics_adj_pts", "whiff_pct", "fb_velo"]
    cand_heads = ["Pitcher", "Role", "Age", "2026 K%", "2027 forecast", "Change (pts)",
                  "Metrics adj (pts)", "Whiff%", "FB velo"]
    cand_fmts = [None, None, lambda v: f"{v:.0f}", pct, pct, lambda v: f"{v * 100:+.1f}",
                 lambda v: f"{v:+.1f}", pct, lambda v: "n/a" if pd.isna(v) else f"{v:.1f}"]
    regulars = pool[pool["bf"] >= 300]
 
    print("\nBreakout candidates: biggest forecast INCREASE over 2026 K% (300+ BF)")
    table(regulars.nlargest(10, "change_vs_2026"), cand_cols, cand_heads, cand_fmts)
    print("\nRegression candidates: biggest forecast DECREASE from 2026 K% (300+ BF)")
    table(regulars.nsmallest(10, "change_vs_2026"), cand_cols, cand_heads, cand_fmts)
    print("\n  Metrics adj = how much whiff rate, velocity, spin, BB% and age move the forecast,")
    print("  beyond the pitcher's own shrunk strikeout history. Positive = his stuff says better")
    print("  than his recent results.")
    print("  Reminder from Phase 5: forecasts near 30%+ ran about 1 point LOW in testing, so")
    print("  elite strikeout pitchers on the regression list may not drop as far as shown.")

    # stuff vs results (xK%) 
    stuff_pool = pool[pool["bf"] >= pool["role"].map(STUFF_MIN_BF)]
    s_cols = ["name", "role", "age", "k_pct", "xk_pct", "stuff_gap", "forecast_2027",
              "whiff_pct", "fb_velo"]
    s_heads = ["Pitcher", "Role", "Age", "2026 K%", "xK% (stuff)", "Gap (pts)",
               "2027 forecast", "Whiff%", "FB velo"]
    s_fmts = [None, None, lambda v: f"{v:.0f}", pct, pct, lambda v: f"{v * 100:+.1f}", pct,
              pct, lambda v: f"{v:.1f}"]
    print(f"\nExpected K% from stuff (xK%): fit on whiff rate, fastball velocity and spin; "
          f"R-squared {xk_r2['SP']:.2f} for starters, {xk_r2['RP']:.2f} for relievers")
    print(f"  (starters {STUFF_MIN_BF['SP']}+ BF, relievers {STUFF_MIN_BF['RP']}+ BF in 2026)")
    print("\nStuff better than results: xK% most ABOVE actual 2026 K%")
    table(stuff_pool.nlargest(10, "stuff_gap"), s_cols, s_heads, s_fmts)
    print("\nResults better than stuff: xK% most BELOW actual 2026 K%")
    table(stuff_pool.nsmallest(10, "stuff_gap"), s_cols, s_heads, s_fmts)
    print("\n  xK% = the strikeout rate a typical pitcher with this whiff rate, velocity and spin")
    print("  would post. Gap = xK% minus actual 2026 K%. Positive = stuff says he should be")
    print("  striking out more batters; negative = his results are running ahead of his stuff.")

    # save 
    out_cols = ["player_id", "name", "role", "age", "bf", "k_pct", "whiff_pct", "fb_velo",
                "velo_change", "forecast_2027", "talent_lo80", "talent_hi80",
                "outcome_lo80", "outcome_hi80", "change_vs_2026", "metrics_adj_pts",
                "xk_pct", "stuff_gap"]
    out = pool[out_cols].sort_values("forecast_2027", ascending=False)
    con = sqlite3.connect(DB_PATH)
    out.to_sql("forecasts_2027", con, if_exists="replace", index=False)
    con.close()
    out.to_csv(DATA_DIR / "forecasts_2027.csv", index=False)
    print(f"\nSaved {len(out):,} forecasts to the 'forecasts_2027' table and "
          f"{DATA_DIR / 'forecasts_2027.csv'}")
 
    # charts 
    print("\nCharts")
 
    # 1) top 20 starters: forecast with 80% talent interval, 2026 K% for reference
    top = pool[(pool["role"] == "SP") & (pool["bf"] >= 400)].nlargest(20, "forecast_2027")
    top = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    y = np.arange(len(top))
    ax.hlines(y, top["talent_lo80"] * 100, top["talent_hi80"] * 100, color=BLUE,
              linewidth=2.5, alpha=0.45)
    ax.plot(top["forecast_2027"] * 100, y, "o", color=BLUE, markersize=8,
            label="2027 forecast (bar = 80% interval for true K%)")
    ax.plot(top["k_pct"] * 100, y, "o", markerfacecolor=SURFACE, markeredgecolor=INK_2,
            markeredgewidth=1.5, markersize=7, label="2026 actual K%")
    ax.set_yticks(y, top["name"])
    ax.set_xlabel("Strikeout rate, %")
    ax.set_title("Top 20 starters by 2027 forecast K%")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.08), ncol=2, fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "19_top_starters_2027.png")
 
    # 2) breakout and regression candidates: 2026 K% -> 2027 forecast
    up = regulars.nlargest(10, "change_vs_2026").iloc[::-1]
    down = regulars.nsmallest(10, "change_vs_2026").iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    for ax, part, title in [(axes[0], up, "Breakout candidates"),
                            (axes[1], down, "Regression candidates")]:
        y = np.arange(len(part))
        ax.hlines(y, part["k_pct"] * 100, part["forecast_2027"] * 100, color=GRID, linewidth=3)
        ax.plot(part["k_pct"] * 100, y, "o", markerfacecolor=SURFACE, markeredgecolor=INK_2,
                markeredgewidth=1.5, markersize=7, label="2026 actual")
        ax.plot(part["forecast_2027"] * 100, y, "o", color=BLUE, markersize=8,
                label="2027 forecast")
        ax.set_yticks(y, part["name"])
        ax.set_title(title)
        ax.set_xlabel("Strikeout rate, %")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.1), ncol=2, fontsize=9)
    save(fig, "20_breakout_regression_2027.png")
    
    # 3) stuff vs results: diverging bars
    hi = stuff_pool.nlargest(10, "stuff_gap")
    lo = stuff_pool.nsmallest(10, "stuff_gap")
    both = pd.concat([lo, hi]).sort_values("stuff_gap")
    gap = both["stuff_gap"] * 100
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    y = np.arange(len(both))
    colors = [BLUE if v > 0 else ORANGE for v in gap]
    ax.barh(y, gap, color=colors, height=0.65)
    for yi, v, (_, row) in zip(y, gap, both.iterrows()):
        ax.text(v + (0.15 if v > 0 else -0.15), yi,
                f"{v:+.1f}  (actual {row['k_pct']:.0%}, xK {row['xk_pct']:.0%})",
                va="center", ha="left" if v > 0 else "right", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(y, [f"{n} ({r})" for n, r in zip(both["name"], both["role"])])
    lim = gap.abs().max() * 2.1
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("xK% minus actual 2026 K%, percentage points")
    ax.set_title("Stuff vs results: whose metrics disagree with their strikeouts?")
    ax.text(0.99, 0.98, "Stuff better than results \u2192", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=INK_2)
    ax.text(0.01, 0.02, "\u2190 Results better than stuff", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, color=INK_2)
    ax.grid(axis="y", visible=False)
    save(fig, "21_stuff_vs_results_2027.png")
 
 
if __name__ == "__main__":     
    main()
 
