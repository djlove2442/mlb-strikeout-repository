import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pybaseball import statcast, cache

cache.enable()
pd.set_option("display.max_columns", 50)

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "baseball.db"

SEASONS = list(range(2015,2027))
MIN_BF = 100

KEEP_COLS = [
    "game_pk", "game_date", "game_year", "game_type", "inning", "inning_topbot",
    "at_bat_number", "pitch_number", "pitcher", "p_throws", "pitch_type", "release_speed",
    "release_spin_rate", "description", "events", "bat_score", "post_bat_score"
    ]
def month_ranges(season):
    starts = pd.date_range(f"{season}-03-01", f"{season}-10-01", freq="MS")
    for start in starts:
        end = start + pd.offsets.MonthEnd(0)
        yield start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def download_season(season):
    for start, end in month_ranges(season):
        out = RAW_DIR / f"statcast_{start[:7]}.parquet"
        if out.exists()and out.stat().st_size > 0:
            continue
        print(f"Downloading {start} to {end} ...")
        df = statcast(start_dt=start, end_dt=end, verbose=False)
        if df is None or df.empty:
            print(" (no game)")
            continue
        df = df[[c for c in KEEP_COLS if c in df.columns]]
        df = df[df["game_type"] == "R"]
        df.to_parquet(out, index=False)
        print(f" saved {len(df):,} pitches")
        time.sleep(2)

for season in SEASONS:
    download_season(season)


### 2. Aggregate pitches into pitcher-season stats
K_EVENTS = {"strikeout", "strikeout_double_play"}
BB_EVENTS = {"walk", "intent_walk"}
SWING_DESC = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
    "foul_bunt", "missed_bunt", "bunt_foul_tip"
    }
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}
OUTS_PER_EVENT = {
    "strikeout": 1, "field_out": 1, "force_out": 1, "sac_fly": 1, "sac_bunt": 1,
    "fielders_choice_out": 1, "other_out": 1,
    "double_play": 2, "grounded_into_double_play": 2, "strikeout_double_play": 2,
    "sac_fly_double_play": 2, "sac_bunt_double_play": 2,
    "triple_play": 3,
}

def load_season(season):
    files = sorted(RAW_DIR.glob(f"statcast_{season}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No downloaded files for {season}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)

def aggregate_season(pitches, season):
    p = pitches.copy()
    p = p.sort_values(["game_pk", "at_bat_number", "pitch_number"])


    # Pitch-level flags
    p["is_swing"] = p["description"].isin(SWING_DESC)
    p["is_whiff"] = p["description"].isin(WHIFF_DESC)
    p["is_called_strike"] = p["description"].eq("called_strike")
    p["is_fb"] = p["pitch_type"].isin(["FF", "SI"])
    p["is_ff"] = p["pitch_type"].eq("FF")
    p["runs_on_pitch"] = (p["post_bat_score"] - p["bat_score"]).clip(lower=0)

    pitch_agg = p.groupby("pitcher").agg(
        pitches=("description", "size"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        called_strikes=("is_called_strike", "sum"),
        runs_allowed=("runs_on_pitch", "sum"),
        throws=("p_throws", "first"),
    )
    fb = p[p["is_fb"]].groupby("pitcher")["release_speed"].mean().rename("fb_velo")
    ff_spin = p[p["is_ff"]].groupby("pitcher")["release_spin_rate"].mean().rename("ff_spin")


    # plate-appearance-level stats (one row per PA-ending pitch)
    pa = p[p["events"].notna() & (p["events"] != "truncated_pa")]
    pa = pa.assign(
        k=pa["events"].isin(K_EVENTS),
        bb=pa["events"].isin(BB_EVENTS),
        hbp=pa["events"].eq("hit_by_pitch"),
        hr=pa["events"].eq("home_run"),
        outs=pa["events"].map(OUTS_PER_EVENT).fillna(0),
    )
    pa_agg = pa.groupby("pitcher").agg(
        bf=("events", "size"), k=("k", "sum"), bb=("bb", "sum"),
        hbp=("hbp", "sum"), hr=("hr", "sum"), outs=("outs", "sum"),
    )


    # games and starts: a starter threw his team's first pitch of the game
    first_pitch = p.drop_duplicates(["game_pk", "inning_topbot"], keep="first")
    gs = first_pitch.groupby("pitcher").size().rename("gs")
    g = p.groupby("pitcher")["game_pk"].nunique().rename("g")
 
    s = pa_agg.join([pitch_agg, fb, ff_spin, gs, g], how="left")
    s["gs"] = s["gs"].fillna(0).astype(int)
    s["season"] = season

    # rate stats
    s["ip"] = s["outs"] / 3
    s["k_pct"] = s["k"] / s["bf"]
    s["bb_pct"] = s["bb"] / s["bf"]
    s["k_minus_bb"] = s["k_pct"] - s["bb_pct"]
    s["whiff_pct"] = s["whiffs"] / s["swings"]
    s["csw_pct"] = (s["called_strikes"] + s["whiffs"]) / s["pitches"]
    s["role"] = np.where(s["gs"] / s["g"] >= 0.5, "SP", "RP")

    # FIP, with the constant set from league totals (RA9-scaled)
    fip_core = 13 * s["hr"] + 3 * (s["bb"] + s["hbp"]) - 2 * s["k"]
    lg_ip = s["ip"].sum()
    lg_ra9 = 9 * s["runs_allowed"].sum() / lg_ip
    fip_constant = lg_ra9 - fip_core.sum() / lg_ip
    s["fip"] = np.where(s["ip"] > 0, fip_core / s["ip"] + fip_constant, np.nan)
 
    return s.reset_index().rename(columns={"pitcher": "player_id"})

season_tables = []
for season in SEASONS:
    print(f"Aggregating {season} ...")
    season_tables.append(aggregate_season(load_season(season), season))
 
pitcher_seasons = pd.concat(season_tables, ignore_index=True)
print(pitcher_seasons.head())


### 3. Add names and ages from the MLB Stats API

def fetch_people(player_ids, batch_size=150):
    rows = []
    ids = [int(i) for i in player_ids]
    for i in range(0, len(ids), batch_size):
        chunk = ",".join(map(str, ids[i:i + batch_size]))
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/people",
            params={"personIds": chunk}, timeout=30,
        )
        r.raise_for_status()
        for person in r.json().get("people", []):
            rows.append({
                "player_id": person["id"],
                "name": person.get("fullName"),
                "birth_date": person.get("birthDate"),
            })
        time.sleep(0.5)
    return pd.DataFrame(rows)

people = fetch_people(pitcher_seasons["player_id"].unique())
pitcher_seasons = pitcher_seasons.merge(people, on="player_id", how="left")
 
birth = pd.to_datetime(pitcher_seasons["birth_date"])
june30 = pd.to_datetime(pitcher_seasons["season"].astype(str) + "-06-30")
pitcher_seasons["age"] = ((june30 - birth).dt.days / 365.25).floordiv(1)
 
cols = ["player_id", "name", "season", "age", "throws", "role", "g", "gs", "bf", "ip",
        "k", "bb", "hbp", "hr", "k_pct", "bb_pct", "k_minus_bb", "whiff_pct", "csw_pct",
        "fb_velo", "ff_spin", "fip", "pitches"]
pitcher_seasons = pitcher_seasons[cols]
print(pitcher_seasons.sample(5, random_state=1))

### 4. Sanity checks
# League K% should sit around 20–23% in this era, and the leaderboard should look like real
league = pitcher_seasons.groupby("season").apply(
    lambda d: pd.Series({
        "pitchers": len(d),
        "league_k_pct": d["k"].sum() / d["bf"].sum(),
        "league_bb_pct": d["bb"].sum() / d["bf"].sum(),
        "avg_fb_velo": np.average(d["fb_velo"].dropna(), weights=d.loc[d["fb_velo"].notna(), "bf"]),
    }),
    include_groups=False,
)
print("\nLeague averages by season:")
print(league.round(3))
 
print("\n2026 strikeout leaders (500+ BF):")
print(pitcher_seasons
 .query("season == 2026 and bf >= 500")
 .nlargest(10, "k_pct")[["name", "role", "bf", "k_pct", "bb_pct", "whiff_pct", "fb_velo", "fip"]]
 .round(3))

### 5. Load into SQLite and build the modeling table with SQL

con = sqlite3.connect(DB_PATH)
pitcher_seasons.to_sql("pitcher_seasons", con, if_exists="replace", index=False)
 
PAIRS_SQL = f"""
SELECT
    cur.player_id,
    cur.name,
    cur.season,
    cur.age,
    cur.throws,
    cur.role,
    cur.bf,
    cur.k,
    cur.bb,
    cur.k_pct,
    cur.bb_pct,
    cur.k_minus_bb,
    cur.whiff_pct,
    cur.csw_pct,
    cur.fb_velo,
    cur.ff_spin,
    cur.fip,
    cur.fb_velo - prev.fb_velo              AS velo_change,
    nxt.bf                                  AS next_bf,
    nxt.k                                   AS next_k,
    nxt.k_pct                               AS next_k_pct,
    nxt.fip                                 AS next_fip,
    nxt.role                                AS next_role,
    CASE WHEN cur.season IN (2019, 2020) THEN 1 ELSE 0 END  AS involves_2020,
    CASE WHEN cur.season >= 2023 THEN 1 ELSE 0 END          AS post_2023_rules
FROM pitcher_seasons AS cur
JOIN pitcher_seasons AS nxt
  ON nxt.player_id = cur.player_id
 AND nxt.season    = cur.season + 1
LEFT JOIN pitcher_seasons AS prev
  ON prev.player_id = cur.player_id
 AND prev.season    = cur.season - 1
WHERE cur.bf >= {MIN_BF}
  AND nxt.bf >= {MIN_BF}
ORDER BY cur.season, cur.player_id
"""
 
pairs = pd.read_sql(PAIRS_SQL, con)
pairs.to_sql("model_pairs", con, if_exists="replace", index=False)
pairs.to_parquet(DATA_DIR / "model_pairs.parquet", index=False)
 
print(f"{len(pairs):,} pitcher-season pairs")
print(pairs.groupby("season").size().rename("pairs"))
 
# ## 6. How many pitchers drop out? (selection bias check)
# Pitchers who qualify in season *t* but not *t+1* are missing from the modeling table.
# Record this number for the Limitations section of your write-up.
 
dropout_sql = f"""
SELECT cur.season,
       COUNT(*)                                             AS qualified,
       SUM(CASE WHEN nxt.bf >= {MIN_BF} THEN 1 ELSE 0 END)  AS returned,
       ROUND(1.0 - AVG(CASE WHEN nxt.bf >= {MIN_BF} THEN 1.0 ELSE 0 END), 3) AS dropout_rate
FROM pitcher_seasons AS cur
LEFT JOIN pitcher_seasons AS nxt
  ON nxt.player_id = cur.player_id AND nxt.season = cur.season + 1
WHERE cur.bf >= {MIN_BF} AND cur.season < {max(SEASONS)}
GROUP BY cur.season
ORDER BY cur.season
"""
print("\nDropout between seasons:")
print(pd.read_sql(dropout_sql, con))
 
con.close()
 



















