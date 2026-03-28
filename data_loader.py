"""
================================================================================
MODULE 1: DATA ACQUISITION SETUP
File: data_loader.py
================================================================================
"""

import subprocess
import json
import logging
import sys
import time
from pathlib import Path
from datetime import date, datetime

import pandas as pd

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
RPSCRAPE_DIR = Path("./rpscrape")
RPSCRAPE_SCRIPT = RPSCRAPE_DIR / "scripts" / "rpscrape.py"
RACECARDS_SCRIPT= RPSCRAPE_DIR / "scripts" / "racecards.py"

OUTPUT_DIR = Path("./data")
HISTORY_CSV = OUTPUT_DIR / "history.csv"
TODAY_CARDS_JSON= OUTPUT_DIR / "today_cards.json"

SCRAPE_START = "2026/02/01"
SCRAPE_END = "2026/02/28"
RACE_TYPES = ["flat", "jumps"]
REGIONS = ["gb"]
SCRAPE_DELAY = 1  


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _validate_rpscrape():
    if not RPSCRAPE_DIR.exists():
        raise FileNotFoundError("rpscrape folder not found.")
    if not RPSCRAPE_SCRIPT.exists():
        raise FileNotFoundError("rpscrape.py script missing.")

def _run(cmd: list, cwd=None) -> subprocess.CompletedProcess:
    log.info("CMD: " + " ".join(str(c) for c in cmd))
    log.info("--- Scraper Output Beginning (Please wait, this may take minutes) ---")
    
    # By removing capture_output entirely, we force rpscrape's built-in progress 
    # bar to render directly in your command prompt in real-time.
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    
    log.info("--- Scraper Output Complete ---")
    if r.returncode != 0:
        raise RuntimeError(f"Exit {r.returncode}")
    return r

# ── Public Functions ──────────────────────────────────────────────────────────

def scrape_historical_data(start: str = SCRAPE_START, end: str = SCRAPE_END) -> Path:
    _validate_rpscrape()
    _ensure_dirs()

    # Generate a list of the last day of each month
    import calendar
    s = datetime.strptime(start, "%Y/%m/%d")
    e = datetime.strptime(end, "%Y/%m/%d")
   
    date_ranges = []
    cur = s.replace(day=1)
    while cur <= e:
        last_day = calendar.monthrange(cur.year, cur.month)[1]
        month_end = cur.replace(day=last_day)
        if month_end > e:
            month_end = e
        # Format as YYYY/MM/DD-YYYY/MM/DD
        rng = f"{cur.strftime('%Y/%m/%d')}-{month_end.strftime('%Y/%m/%d')}"
        date_ranges.append(rng)
       
        # Advance to next month
        cur = cur.replace(day=1)
        cur = cur.replace(month=cur.month % 12 + 1, year=cur.year + (1 if cur.month == 12 else 0))

    total = len(date_ranges) * len(REGIONS) * len(RACE_TYPES)
    log.info(f"Planning {total} scrape calls in monthly chunks to avoid timeouts.")

    frames = []
    done = 0

    import os
    env = os.environ.copy()
    env["RPSCRAPE_NO_UPDATE"] = "1"

    for region in REGIONS:
        for rtype in RACE_TYPES:
            for rng in date_ranges:
                done += 1
                log.info(f"[{done}/{total}] {region}/{rtype}/{rng}")
                try:
                    _run(
                        [
                            sys.executable, str(RPSCRAPE_SCRIPT.resolve()),
                            "-r", region,
                            "-d", rng,
                            "-t", rtype
                        ],
                        cwd=RPSCRAPE_DIR / "scripts",
                    )
                except RuntimeError as exc:
                    log.warning(f" Scrape failed — {exc}")
                    time.sleep(2)
                    continue

                # Load any newly created CSVs
                data_folder = RPSCRAPE_DIR / "data" / rtype / region
                if data_folder.exists():
                    for chunk in data_folder.glob("*.csv"):
                        try:
                            # Keep only chunks we haven't already processed
                            if chunk.name not in [f.name for f in frames if hasattr(f, 'name')]:
                                df = pd.read_csv(chunk, low_memory=False)
                                df["race_type"] = rtype
                                df["region"] = region
                                df.name = chunk.name # tag it so we don't load it twice
                                frames.append(df)
                                log.info(f" ✓ Added {len(df):,} rows from {chunk.name}")
                        except Exception as e:
                            pass
               
                time.sleep(3) # Pause to respect the server

    if not frames:
        raise RuntimeError("No data collected. Check rpscrape is working.")

    history = pd.concat(frames, ignore_index=True)
    history.drop_duplicates(inplace=True)
    if "date" in history.columns:
        history.sort_values("date", inplace=True, ignore_index=True)

    history.to_csv(HISTORY_CSV, index=False)
    log.info(f"\n✅ history.csv → {HISTORY_CSV.resolve()} | {len(history):,} rows")
    return HISTORY_CSV

def _stub_cards() -> Path:
    stub = {
        "_meta": {
            "status": "STUB — populate manually or fix racecards.py",
            "date": str(date.today())
        },
        "races": []
    }
    TODAY_CARDS_JSON.write_text(json.dumps(stub, indent=2), encoding="utf-8")
    return TODAY_CARDS_JSON


def fetch_todays_racecards(region: str = "gb") -> Path:
    _ensure_dirs()

    if not RACECARDS_SCRIPT.exists():
        log.warning("racecards.py not found — writing stub.")
        return _stub_cards()

    try:
        # We use capture_output=True here because we NEED to read the JSON it outputs
        r = subprocess.run(
            [
                sys.executable, str(RACECARDS_SCRIPT.resolve()),
                "--day", "1", "--region", region
            ],
            cwd=RPSCRAPE_DIR / "scripts",
            capture_output=True, text=True
        )
        raw = r.stdout.strip()
        if not raw:
            log.warning("No output from racecards.py — writing stub.")
            return _stub_cards()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Non-JSON output — saving raw.")
            data = {"raw": raw, "date": str(date.today())}

        TODAY_CARDS_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info(f"✅ today_cards.json → {TODAY_CARDS_JSON.resolve()}")
        return TODAY_CARDS_JSON

    except RuntimeError as exc:
        log.warning(f"racecards.py failed: {exc} — writing stub.")
        return _stub_cards()

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info(" MODULE 1 — Data Acquisition")
    log.info("=" * 60)

    log.info("\n[1/2] Scraping historical data …")
    scrape_historical_data()

    log.info("\n[2/2] Fetching today's racecards …")
    fetch_todays_racecards()

    log.info("\n✅ Module 1 complete. Run feature_builder.py next.")