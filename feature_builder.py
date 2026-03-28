"""
================================================================================
MODULE 2: FEATURE ENGINEERING (PRO EDITION)
File: feature_builder.py
================================================================================
"""
import logging
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

HISTORY_CSV = Path("./data/history.csv")
FEATURES_CSV = Path("./data/features.csv")

# ── Column Mapping ────────────────────────────────────────────────────────────
COL_MAP = {
    "date": "date",
    "course": "course",
    "off": "race_time",       
    "name": "horse",           
    "sp": "sp",              
    "position": "position",        
    "wt": "weight",          
    "jockey": "jockey",
    "trainer": "trainer",
    "age": "age",
    "dist": "distance",        
    "going": "going",
    "class": "race_class",
    "pattern": "pattern",
    "ts": "topspeed",
    "rpr": "rpr",
    "or": "or_rating" # NEW: Official Rating
}

COL_ALIASES = {
    "horse": ["name", "horse", "Name", "Horse"],
    "date": ["date", "Date"],
    "course": ["course", "Course", "venue"],
    "position": ["position", "pos", "Position", "Pos"],
    "sp": ["sp", "SP", "starting_price"],
    "weight": ["wt", "weight", "Weight", "lbs"],
    "jockey": ["jockey", "Jockey", "jock"],
    "trainer": ["trainer", "Trainer"], # NEW
    "distance": ["dist", "distance", "Distance", "trip"],
    "going": ["going", "Going"],
    "age": ["age", "Age"],
    "rpr": ["rpr", "RPR", "rating"], # NEW
    "or": ["or", "OR", "official_rating"] # NEW
}


def _resolve_col(df: pd.DataFrame, key: str) -> str | None:
    candidates = COL_ALIASES.get(key, [key])
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _parse_sp(sp_str) -> float:
    if pd.isna(sp_str):
        return np.nan
    s = str(sp_str).strip().upper()
    if s in ("EVS", "EVENS", "1/1"):
        return 0.5
    m = re.match(r"(\d+\.?\d*)\s*/\s*(\d+\.?\d*)", s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if num + den == 0:
            return np.nan
        decimal = (num / den) + 1.0
        return 1.0 / decimal
    try:
        dec = float(s)
        return 1.0 / dec if dec > 0 else np.nan
    except ValueError:
        return np.nan

def _parse_weight(wt_str) -> float | None:
    if pd.isna(wt_str):
        return np.nan
    s = str(wt_str).strip()
    m = re.match(r"(\d+)-(\d+)", s)
    if m:
        return int(m.group(1)) * 14 + int(m.group(2))
    try:
        return float(s)
    except ValueError:
        return np.nan

def _parse_position(pos_str) -> int | None:
    if pd.isna(pos_str):
        return np.nan
    s = str(pos_str).strip().upper()
    if s in ("F", "PU", "UR", "BD", "RO", "SU", "REF", "DSQ", "WO", "CO", ""):
        return np.nan
    try:
        return int(float(s))
    except ValueError:
        return np.nan

def _parse_distance(dist_str) -> float | None:
    if pd.isna(dist_str):
        return np.nan
    s = str(dist_str).strip().lower().replace(" ", "")
    furlongs = 0.0
    m_miles = re.search(r"(\d+\.?\d*)m", s)
    if m_miles:
        furlongs += float(m_miles.group(1)) * 8
    m_fur = re.search(r"(\d+\.?\d*)f", s)
    if m_fur:
        furlongs += float(m_fur.group(1))
    m_yds = re.search(r"(\d+)y", s)
    if m_yds:
        furlongs += int(m_yds.group(1)) / 220
    if furlongs == 0.0:
        try:
            return float(s)
        except ValueError:
            return np.nan
    return round(furlongs, 2)

def _win_rate_last_n(group: pd.DataFrame, n: int = 5) -> pd.Series:
    won = (group["position_int"] == 1).astype(float)
    return won.shift(1).rolling(n, min_periods=1).mean()

def _avg_position_last_n(group: pd.DataFrame, n: int = 3) -> pd.Series:
    pos = group["position_int"]
    return pos.shift(1).rolling(n, min_periods=1).mean()

# ── Main Feature Builder ──────────────────────────────────────────────────────

def build_features(
    history_path: Path = HISTORY_CSV,
    output_path: Path = FEATURES_CSV,
) -> pd.DataFrame:
    log.info(f"Loading raw data from {history_path} …")
    if not history_path.exists():
        raise FileNotFoundError("history.csv not found.")

    df = pd.read_csv(history_path, low_memory=False)

    col = {key: _resolve_col(df, key) for key in COL_ALIASES}
    rename_map = {v: k for k, v in col.items() if v and v != k}
    df.rename(columns=rename_map, inplace=True)

    for key in COL_ALIASES:
        if key not in df.columns:
            df[key] = np.nan

    log.info("Parsing core data...")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["position_int"] = df["position"].apply(_parse_position)
    df["weight_lbs"] = df["weight"].apply(_parse_weight)
    df["sp_prob"] = df["sp"].apply(_parse_sp)
    df["dist_f"] = df["distance"].apply(_parse_distance)

    # NEW: Safely parse Ratings
    df["rpr"] = pd.to_numeric(df["rpr"], errors="coerce")
    df["or"] = pd.to_numeric(df["or"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df["won"] = (df["position_int"] == 1).astype(int)
    df.sort_values("date", inplace=True, ignore_index=True)

    # Rolling Stats
    log.info("Engineering Rolling Stats (Horse, Jockey, Trainer) …")
    df["horse_win_rate_5"] = df.groupby("horse", group_keys=False).apply(_win_rate_last_n, n=5).reset_index(level=0, drop=True)
    df["jockey_win_rate_5"] = df.groupby("jockey", group_keys=False).apply(_win_rate_last_n, n=5).reset_index(level=0, drop=True)
    df["trainer_win_rate_5"] = df.groupby("trainer", group_keys=False).apply(_win_rate_last_n, n=5).reset_index(level=0, drop=True) # NEW

    df["days_since_last_run"] = df.groupby("horse")["date"].transform(lambda s: s.diff().dt.days)

    def _last_win_weight(group: pd.DataFrame) -> pd.Series:
        weights = group["weight_lbs"].copy()
        wins = group["won"].copy()
        result = pd.Series(np.nan, index=group.index)
        last_win_wt = np.nan
        for i in group.index:
            result.at[i] = last_win_wt
            if wins.at[i] == 1:
                last_win_wt = weights.at[i]
        return result

    df["last_win_weight"] = df.groupby("horse", group_keys=False).apply(_last_win_weight).reset_index(level=0, drop=True)
    df["weight_diff"] = df["weight_lbs"] - df["last_win_weight"]

    def _course_dist_win(group: pd.DataFrame) -> pd.Series:
        result = pd.Series(0, index=group.index)
        win_set: set = set()
        for i in group.index:
            key = (group.at[i, "course"], round(group.at[i, "dist_f"] or 0, 1))
            result.at[i] = int(key in win_set)
            if group.at[i, "won"] == 1:
                win_set.add(key)
        return result

    df["course_dist_win"] = df.groupby("horse", group_keys=False).apply(_course_dist_win).reset_index(level=0, drop=True)
    df["avg_pos_last3"] = df.groupby("horse", group_keys=False).apply(_avg_position_last_n, n=3).reset_index(level=0, drop=True)

    # ── Finalise ──────────────────────────────────────────────────────────────
    log.info("Cleaning and finalising …")
    
    # WE MUST EXPLICITLY KEEP 'dec' IF IT EXISTS FOR THE BACKTESTER TO WORK LATER
    if 'dec' not in df.columns:
         df['dec'] = np.nan

    FEATURE_COLS = [
        "date", "course", "horse", "jockey", "trainer", "dist_f", "going", "dec",
        "horse_win_rate_5", "jockey_win_rate_5", "trainer_win_rate_5", # NEW
        "weight_diff", "days_since_last_run", "course_dist_win", "avg_pos_last3",
        "sp_prob", "age", "weight_lbs", 
        "rpr", "or", # NEW
        "won"
    ]

    final_cols = [c for c in FEATURE_COLS if c in df.columns]
    features = df[final_cols].copy()

    features["days_since_last_run"] = features["days_since_last_run"].clip(0, 365)
    features["weight_diff"] = features["weight_diff"].clip(-30, 30)
    features["age"] = features["age"].clip(2, 15)

    ML_COLS = [
        "horse_win_rate_5", "jockey_win_rate_5", "trainer_win_rate_5", 
        "weight_diff", "days_since_last_run", "course_dist_win", "avg_pos_last3",
        "sp_prob", "age", "weight_lbs", "rpr", "or"
    ]
    for c in ML_COLS:
        if c in features.columns:
            median = features[c].median()
            features[c] = features[c].fillna(median)

    features.to_csv(output_path, index=False)
    log.info(f"\n✅ features.csv → {output_path.resolve()} | {len(features):,} rows")
    return features


def build_today_features(today_cards: list[dict]) -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError("features.csv not found.")
    hist = pd.read_csv(FEATURES_CSV, low_memory=False)

    horse_wr = hist.groupby("horse")["won"].mean().to_dict()
    jockey_wr = hist.groupby("jockey")["won"].mean().to_dict()
    trainer_wr = hist.groupby("trainer")["won"].mean().to_dict() # NEW

    horse_last_win_wt = hist[hist["won"] == 1].sort_values("date").groupby("horse")["weight_lbs"].last().to_dict()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    horse_last_date = hist.sort_values("date").groupby("horse")["date"].last().to_dict()

    cd_wins = hist[hist["won"] == 1].assign(cd=lambda x: x["course"] + "_" + x["dist_f"].round(1).astype(str)).groupby("horse")["cd"].apply(set).to_dict()

    rows = []
    today = pd.Timestamp("today").normalize()

    for race in today_cards:
        course = race.get("course", "Unknown")
        dist_f = _parse_distance(race.get("distance", ""))
        race_id = race.get("race_id", "")

        for runner in race.get("runners", []):
            horse = runner.get("horse", "")
            jockey = runner.get("jockey", "")
            trainer = runner.get("trainer", "")
            wt_lbs = _parse_weight(runner.get("weight", ""))
            
            # Since today's live cards might not have RPR or OR, we will fill them with historic medians later
            rpr = np.nan 
            or_rating = np.nan 

            last_win_wt = horse_last_win_wt.get(horse, np.nan)
            wt_diff = (wt_lbs - last_win_wt) if (not np.isnan(wt_lbs) and not np.isnan(last_win_wt)) else np.nan

            last_date = horse_last_date.get(horse)
            days_off = (today - last_date).days if last_date else np.nan

            cd_key = f"{course}_{round(dist_f or 0, 1)}"
            cd_win = int(cd_key in cd_wins.get(horse, set()))

            rows.append({
                "race_id": race_id,
                "course": course,
                "horse": horse,
                "jockey": jockey,
                "trainer": trainer,
                "dist_f": dist_f,
                "horse_win_rate_5": horse_wr.get(horse, 0.0),
                "jockey_win_rate_5": jockey_wr.get(jockey, 0.0),
                "trainer_win_rate_5":trainer_wr.get(trainer, 0.0), # NEW
                "weight_diff": wt_diff,
                "days_since_last_run": days_off,
                "course_dist_win": cd_win,
                "avg_pos_last3": np.nan,
                "sp_prob": _parse_sp(runner.get("odds", "")),
                "age": float(runner.get("age", np.nan)) if runner.get("age") else np.nan,
                "weight_lbs": wt_lbs,
                "rpr": rpr, # NEW
                "or": or_rating # NEW
            })

    today_df = pd.DataFrame(rows)

    hist_medians = hist[[
        "horse_win_rate_5","jockey_win_rate_5","trainer_win_rate_5","weight_diff",
        "days_since_last_run","avg_pos_last3","sp_prob","age","weight_lbs", "rpr", "or"
    ]].median()

    for col in hist_medians.index:
        if col in today_df.columns:
            today_df[col] = today_df[col].fillna(hist_medians[col])

    today_df["course_dist_win"] = today_df["course_dist_win"].fillna(0)
    return today_df

if __name__ == "__main__":
    build_features()
