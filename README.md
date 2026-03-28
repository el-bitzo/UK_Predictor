# 🏇 UK Horse Racing Prediction Bot
### 100% Free | AI/ML Powered | XGBoost | Selenium | BeautifulSoup

---

## What This Does

This bot scrapes UK horse racing data, trains an XGBoost ML model, and
outputs ranked daily predictions. Every morning you run ONE command and it
tells you the **top horse per race**, sorted from strongest to weakest edge.

---

## System Requirements

- **Python 3.11+** (3.13 recommended)
- **Google Chrome** installed (for Selenium odds scraping)
- **Git** installed
- **Windows / macOS / Linux** — all supported

---

## STEP-BY-STEP SETUP (Do This Once)

### Step 1 — Clone rpscrape

rpscrape is the free tool that downloads horse racing results.

```bash
git clone https://github.com/rpscrape-team/rpscrape.git
```

> This creates a `rpscrape/` folder in the same directory as these scripts.
> Your folder structure should look like:
> ```
> your-folder/
> ├── rpscrape/          ← cloned repo
> ├── data_loader.py
> ├── feature_builder.py
> ├── odds_tracker.py
> ├── model.py
> ├── main.py
> ├── requirements.txt
> └── README.md
> ```

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

Or if you use Python 3 explicitly:

```bash
pip3 install -r requirements.txt
```

### Step 3 — Scrape Historical Data (One-Time, Takes ~10–20 Minutes)

This scrapes every UK flat + jumps race from Jan 2025 to today.

```bash
python data_loader.py
```

This creates `data/history.csv` (all historical results) and
`data/today_cards.json` (today's races).

> **First-time scraping takes time.** rpscrape requests data month-by-month
> with a 1-second delay between calls. ~15 months of data = ~30 requests.

### Step 4 — Build ML Features

```bash
python feature_builder.py
```

Creates `data/features.csv` with engineered ML features.

### Step 5 — Train the Model + Backtest

```bash
python model.py
```

This:
- Runs 5-fold time-series cross-validation and prints Log Loss, AUC, Pick Accuracy
- Trains the final XGBoost model on all data
- Saves the model to `model/xgb_racing_model.json`

---

## DAILY USAGE (Run Every Morning)

```bash
python main.py
```

This automatically:
1. Fetches today's racecards
2. Scrapes live odds from Oddschecker
3. Loads the trained model
4. Outputs top horse per race, sorted best → worst edge

### Example Output:

```
═══════════════════════════════════════════════════════════════════
  🏇  UK RACING PREDICTIONS — Wednesday 25 March 2026
═══════════════════════════════════════════════════════════════════
  RANK  TIME   COURSE             HORSE                   PROB   EDGE
─────────────────────────────────────────────────────────────────────
  1     13:45  Cheltenham         Jonbon                 38.2%  ⭐⭐⭐  STRONG
  2     14:20  Sandown            Nashwa                 31.5%  ⭐⭐    GOOD
  3     15:00  Newbury            Audience               28.1%  ⭐⭐    GOOD
  4     14:55  Kempton            Kinross                22.4%  ⭐     FAIR
  ...
═══════════════════════════════════════════════════════════════════
```

---

## Command Line Options

| Command | Description |
|---|---|
| `python main.py` | Full pipeline: scrape + predict |
| `python main.py --predict-only` | Use existing data, just predict |
| `python main.py --skip-odds` | Skip live odds scraping (faster) |
| `python main.py --retrain` | Force model retraining |
| `python main.py --train-only` | Retrain without outputting predictions |

---

## File Structure After Setup

```
your-folder/
├── rpscrape/                  ← rpscrape clone (git clone separately)
├── data/
│   ├── history.csv            ← all historical race results
│   ├── features.csv           ← engineered ML features
│   ├── today_cards.json       ← today's racecards
│   ├── today_cards_with_odds.json  ← racecards + live odds
│   └── predictions_YYYY-MM-DD.csv ← saved daily predictions
├── model/
│   ├── xgb_racing_model.json  ← trained XGBoost model
│   ├── feature_importance.csv ← what the model values most
│   └── backtest_metrics.json  ← cross-validation results
├── data_loader.py             ← Module 1: Data scraping
├── feature_builder.py         ← Module 2: Feature engineering
├── odds_tracker.py            ← Module 3: Live odds
├── model.py                   ← Module 4: ML + backtesting
├── main.py                    ← Module 5: Daily predictions
└── requirements.txt
```

---

## ML Features Used

| Feature | Description |
|---|---|
| `horse_win_rate_5` | Win rate in last 5 races |
| `jockey_win_rate_5` | Jockey win rate in last 5 rides |
| `weight_diff` | Current weight vs last winning weight (lbs) |
| `days_since_last_run` | Rest period in days |
| `course_dist_win` | Has this horse won here at this distance? (0/1) |
| `avg_pos_last3` | Average finishing position last 3 runs |
| `sp_prob` | Starting price implied probability |
| `age` | Horse age in years |
| `weight_lbs` | Absolute weight carried |

---

## Troubleshooting

**"rpscrape not found"**
→ Make sure you ran `git clone https://github.com/rpscrape-team/rpscrape.git`
  in the same folder as these scripts.

**"No races in today_cards.json"**
→ racecards.py in rpscrape may need a Racing API key. As a workaround,
  manually populate `data/today_cards.json` using the schema in the stub file.
  Alternatively run with `--skip-odds` and populate cards from Racing Post manually.

**"ChromeDriver not found" (Selenium error)**
→ `pip install webdriver-manager` and ensure Chrome is installed.
  The bot uses `webdriver-manager` to auto-download the right ChromeDriver.

**"No data scraped"**
→ rpscrape depends on Racing Post's website. If Racing Post changes their
  layout, rpscrape may need an update. Check the rpscrape GitHub for issues.

**Low Pick Accuracy in backtest**
→ Horse racing is inherently difficult to predict. Pick accuracy of 20–30%
  for a 10-runner field is meaningful. Add more data (more years) to improve.

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**.
It does not guarantee profits. Horse racing involves significant uncertainty.
Always gamble responsibly. Must be 18+ to bet. Know your limits.

---

## Architecture

```
data_loader.py ──► history.csv ──► feature_builder.py ──► features.csv
                                                                  │
today_cards.json ──► odds_tracker.py ──► today_cards_with_odds.json
                                                                  │
                                         features.csv ──► model.py ──► model/
                                                                         │
                              today_cards_with_odds.json ──► main.py ◄──┘
                                                                  │
                                                            predictions
```
