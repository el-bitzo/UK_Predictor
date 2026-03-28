import subprocess
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import os

def update_db():
    # 1. Automatically calculate yesterday's date
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y/%m/%d")
    print(f"=========================================")
    print(f" SCRAPING MISSING RESULTS FOR: {yesterday}")
    print(f"=========================================")

    # 2. Scrape yesterday's Flat and Jumps races
    # (Historical scraping usually bypasses Cloudflare without issues)
    subprocess.run(["python", "rpscrape/scripts/rpscrape.py", "-r", "gb", "-d", yesterday, "-t", "flat"])
    subprocess.run(["python", "rpscrape/scripts/rpscrape.py", "-r", "gb", "-d", yesterday, "-t", "jumps"])

    # 3. Re-merge all the newly downloaded CSVs into your master history.csv
    print("\nMerging new data into your master database...")
    data_dir = Path("./rpscrape/data")
    all_csvs = list(data_dir.rglob("*.csv"))
    
    frames = []
    for file in all_csvs:
        try:
            df = pd.read_csv(file, low_memory=False)
            frames.append(df)
        except Exception as e:
            pass
            
    if frames:
        history = pd.concat(frames, ignore_index=True)
        history.drop_duplicates(inplace=True)
        if "date" in history.columns:
            history.sort_values("date", inplace=True, ignore_index=True)
            
        out_dir = Path("./data")
        out_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(out_dir / "history.csv", index=False)
        print(f"✅ SUCCESS! Database updated to {len(history)} total races.")
    else:
        print("❌ Could not find any data to merge.")

if __name__ == "__main__":
    # Prevent rpscrape from trying to update via git
    os.environ["RPSCRAPE_NO_UPDATE"] = "1"
    update_db()