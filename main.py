"""
================================================================================
MODULE 5: DAILY PREDICTOR (UK PREDICTOR EDITION)
Developer: @el_bitzo | Discord: @hustlekiddo
================================================================================
"""

import argparse
import json
import logging
import sys
import os
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Enable colors in Windows CMD
os.system("")

# ── ANSI Terminal Colors ──
C_CYAN = "\033[96m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_MAGENTA = "\033[95m"
C_RED = "\033[91m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"

logging.basicConfig(
    level=logging.INFO,
    format=f"{C_CYAN}%(asctime)s{C_RESET} [{C_MAGENTA}%(levelname)s{C_RESET}] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("./data")
HISTORY_CSV = DATA_DIR / "history.csv"
FEATURES_CSV = DATA_DIR / "features.csv"
TODAY_CARDS_JSON = DATA_DIR / "today_cards.json"
ODDS_JSON = DATA_DIR / "today_cards_with_odds.json"
MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / "xgb_racing_model.json"

ML_FEATURES = [
    "horse_win_rate_5", "jockey_win_rate_5", "trainer_win_rate_5",
    "weight_diff", "days_since_last_run", "course_dist_win",
    "avg_pos_last3", "sp_prob", "age", "weight_lbs", "rpr", "or"
]

BANNER = f"""{C_MAGENTA}{C_BOLD}
  ██╗   ██╗██╗  ██╗    ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗██████╗ ██████╗ 
  ██║   ██║██║ ██╔╝    ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝██╔══██╗██╔══██╗
  ██║   ██║█████╔╝     ██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   ██║  ██║██████╔╝
  ██║   ██║██╔═██╗     ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   ██║  ██║██╔══██╗
  ╚██████╔╝██║  ██╗    ██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   ██████╔╝██║  ██║
   ╚═════╝ ╚═╝  ╚═╝    ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝
{C_CYAN}  > Developer: {C_YELLOW}@el_bitzo{C_CYAN} | Discord: {C_YELLOW}@hustlekiddo{C_RESET}
{C_MAGENTA}========================================================================================{C_RESET}
"""

def step_acquire_data():
    from data_loader import scrape_historical_data, fetch_todays_racecards
    if not HISTORY_CSV.exists():
        scrape_historical_data()
    fetch_todays_racecards()

def step_build_features():
    from feature_builder import build_features
    build_features()

def step_fetch_odds():
    from odds_tracker import fetch_and_merge_odds
    fetch_and_merge_odds()

def step_train_model():
    from model import load_features, backtest, train_final_model, save_model
    df = load_features()
    backtest(df, n_splits=5)
    final_model, importance = train_final_model(df)
    save_model(final_model, importance)

def step_predict() -> pd.DataFrame:
    log.info("\n" + f"{C_CYAN}─{C_RESET}" * 60)
    log.info(f" STEP 5/5 — Generating Top AI Predictions (Direct Merge Engine){C_RESET}")
    log.info(f"{C_CYAN}─{C_RESET}" * 60)

    cards_path = ODDS_JSON if ODDS_JSON.exists() else TODAY_CARDS_JSON
    if not cards_path.exists():
        raise FileNotFoundError("No live racecards found. Please run fetch_real.py")
    
    with open(cards_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    races = raw if isinstance(raw, list) else raw.get("races", [])

    live_runners = []
    for race in races:
        r_time = str(race.get("time", "??:??")).upper()
        r_course = str(race.get("course", "Unknown")).title()
        
        for runner in race.get("runners", []):
            raw_name = str(runner.get("horse", ""))
            
            display_name = re.sub(r'\([A-Za-z]+\)', '', raw_name)
            display_name = re.sub(r'^\d+[\.\s]*', '', display_name).strip().title()
            match_name = re.sub(r'[^a-z0-9]', '', display_name.lower())
            
            live_runners.append({
                "match_name": match_name,
                "display_horse": display_name,
                "display_course": r_course,
                "display_time": r_time
            })
            
    live_df = pd.DataFrame(live_runners)
    if live_df.empty:
        log.warning(f"{C_RED}⚠️ No horses found in live racecards.{C_RESET}")
        return pd.DataFrame()

    # ── THE CLONE KILLER & GARBAGE FILTER ──
    # 1. Kill duplicate horses
    live_df.drop_duplicates(subset=['match_name'], inplace=True)
    
    # 2. Kill fake race times (must contain a colon OR the word LATER)
    live_df = live_df[live_df['display_time'].str.contains(r':|LATER', na=False, regex=True)]

    if not FEATURES_CSV.exists():
        log.error(f"{C_RED}⚠️ features.csv missing! Run --retrain to generate it.{C_RESET}")
        return pd.DataFrame()
        
    feat_df = pd.read_csv(FEATURES_CSV)
    
    feat_df['match_name'] = feat_df['horse'].astype(str).apply(
        lambda x: re.sub(r'[^a-z0-9]', '', re.sub(r'\([A-Za-z]+\)', '', x).lower())
    )
    
    if 'date' in feat_df.columns:
        feat_df['date'] = pd.to_datetime(feat_df['date'], errors='coerce')
        feat_df.sort_values('date', ascending=True, inplace=True)
        
    latest_stats_df = feat_df.drop_duplicates(subset=['match_name'], keep='last').copy()

    today_df = pd.merge(live_df, latest_stats_df, on='match_name', how='inner')

    if today_df.empty:
        log.warning(f"{C_RED}⚠️ 0 live horses matched the history. Check database integrity.{C_RESET}")
        return pd.DataFrame()

    from model import load_model
    model = load_model()

    avail_features = [c for c in ML_FEATURES if c in today_df.columns]
    X_today = today_df[avail_features].fillna(0)

    today_df["win_prob"] = model.predict_proba(X_today)[:, 1]

    today_df.sort_values("win_prob", ascending=False, inplace=True)
    today_df.reset_index(drop=True, inplace=True)

    return today_df

def print_predictions(picks: pd.DataFrame) -> None:
    top_3_picks = picks.head(3)
    today = date.today().strftime("%A %d %B %Y")

    print(f"\n{C_MAGENTA}══════════════════════════════════════════════════════════════════════{C_RESET}")
    print(f"{C_BOLD}{C_CYAN} 🏇 UK PREDICTOR — {today}{C_RESET}")
    print(f"{C_MAGENTA}══════════════════════════════════════════════════════════════════════{C_RESET}")
    print(f"  {C_BOLD}{'RANK':<4} {'TIME':<6} {'COURSE':<18} {'HORSE':<24} {'PROB':<6} {'EDGE'}{C_RESET}")
    print(f"{C_CYAN}──────────────────────────────────────────────────────────────────────{C_RESET}")

    rank = 1
    for _, row in top_3_picks.iterrows():
        time_s = str(row.get("display_time", "??:??"))[:5]
        course = str(row.get("display_course", "Unknown"))[:18]
        horse = str(row.get("display_horse", "Unknown"))[:22]
        ai_p = row.get("win_prob", 0.0)
        
        if ai_p >= 0.35:
            edge = f"{C_GREEN}⭐️⭐️⭐️ STRONG{C_RESET}"
        elif ai_p >= 0.20:
            edge = f"{C_YELLOW}⭐️⭐️   GOOD{C_RESET}"
        else:
            edge = f"{C_RED}⭐️     FAIR{C_RESET}"
            
        print(f"  {rank:<4} {time_s:<6} {course:<18} {horse:<24} {C_GREEN}{ai_p*100:>4.1f}%{C_RESET}  {edge}")
        rank += 1

    print(f"{C_MAGENTA}══════════════════════════════════════════════════════════════════════{C_RESET}")
    print(f"  Total verified UK runners matched to database: {C_YELLOW}{len(picks)}{C_RESET}")
    
    if not top_3_picks.empty:
        best = top_3_picks.iloc[0]
        print(f"  #1 Selection: {C_GREEN}{best['display_horse']}{C_RESET} @ {C_GREEN}{best['win_prob']*100:.1f}%{C_RESET} confidence\n")

    print(f"{C_CYAN}──────────────────────────────────────────────────────────────────────{C_RESET}")
    print(f" {C_YELLOW}⚠️  DISCLAIMER: This is a research/ML tool built by @el_bitzo.{C_RESET}")
    print(f" {C_YELLOW}   Always gamble responsibly. 18+ only.{C_RESET}")
    print(f"{C_CYAN}──────────────────────────────────────────────────────────────────────{C_RESET}\n")
    
    picks.drop(picks.index[3:], inplace=True)

def save_predictions(picks: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"predictions_{date.today()}.csv"
    
    save_df = picks[['display_time', 'display_course', 'display_horse', 'win_prob']].copy()
    save_df.columns = ['Time', 'Course', 'Horse', 'Win Probability']
    save_df.to_csv(out, index=False)
    
    log.info(f"Predictions saved → {C_GREEN}{out}{C_RESET}")

def parse_args():
    p = argparse.ArgumentParser(description="UK Predictor by @el_bitzo")
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--skip-odds", action="store_true")
    p.add_argument("--retrain", action="store_true")
    return p.parse_args()

def main():
    print(BANNER)
    args = parse_args()

    try:
        if not args.predict_only:
            step_acquire_data()
            step_build_features()
            if not args.skip_odds:
                step_fetch_odds()
            if not MODEL_PATH.exists() or args.retrain:
                step_train_model()

        if args.train_only:
            return

        picks = step_predict()

        if picks.empty:
            return

        print_predictions(picks)
        save_predictions(picks)

    except Exception as exc:
        log.exception(f"\n{C_RED}❌ Unexpected error: {exc}{C_RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()