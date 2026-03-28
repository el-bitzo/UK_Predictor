"""
================================================================================
TRUE AI BACKTESTER: +EV VALUE BETTING (BULLETPROOF)
File: pnl_chart_ev.py
================================================================================
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from pathlib import Path

print("1. Loading AI Model and Historical Features...")

features_path = Path("data/features.csv")
history_path = Path("data/history.csv")
model_path = Path("model/xgb_racing_model.json")

if not features_path.exists() or not history_path.exists() or not model_path.exists():
    print("❌ Missing required files. Make sure you have run the full pipeline.")
    exit()

features = pd.read_csv(features_path, low_memory=False)
history = pd.read_csv(history_path, low_memory=False)

ML_FEATURES = [
    "horse_win_rate_5",
    "jockey_win_rate_5",
    "trainer_win_rate_5",  # <--- NEW
    "weight_diff",
    "days_since_last_run",
    "course_dist_win",
    "avg_pos_last3",
    "sp_prob",
    "age",
    "weight_lbs",
    "rpr",                 # <--- NEW
    "or",                  # <--- NEW
]

# Load your trained XGBoost Brain
model = xgb.XGBClassifier()
model.load_model(model_path)

# Predict the AI's Win Probability
avail_features = [c for c in ML_FEATURES if c in features.columns]
X = features[avail_features].fillna(0)
features['ai_prob'] = model.predict_proba(X)[:, 1]

print("2. Merging AI Predictions with Bookmaker Odds...")

# ── THE FIX: Remove overlapping columns so Pandas doesn't rename them to won_x / won_y ──
for col in ['dec', 'pos', 'won']:
    if col in features.columns:
        features.drop(columns=[col], inplace=True)
        
hist_subset = history[['date', 'course', 'horse', 'dec', 'pos']].copy()
hist_subset['dec'] = pd.to_numeric(hist_subset['dec'], errors='coerce').fillna(3.0)

# Create a clean 'won' column
extracted_pos = hist_subset['pos'].astype(str).str.extract(r'(\d+)', expand=False)
hist_subset['won'] = pd.to_numeric(extracted_pos, errors='coerce') == 1

# Drop 'pos' before merge to keep it clean
hist_subset.drop(columns=['pos'], inplace=True)

# Merge the clean files
backtest_df = pd.merge(features, hist_subset, on=['date', 'course', 'horse'], how='inner')
backtest_df.drop_duplicates(subset=['date', 'course', 'horse'], inplace=True)
backtest_df.sort_values('date', inplace=True)

print("3. Hunting for Value (+EV)...")
# Calculate Bookie Implied Probability
backtest_df['bookie_prob'] = np.where(backtest_df['dec'] > 1.0, 1.0 / backtest_df['dec'], 0.99)

# Calculate your AI's Edge
backtest_df['edge'] = backtest_df['ai_prob'] - backtest_df['bookie_prob']

# Demand a 15% edge AND refuse to bet on horses with decimal odds higher than 15.0
bets = backtest_df[(backtest_df['edge'] > 0.15) & (backtest_df['dec'] <= 15.0)].copy()

print(f"Found {len(bets):,} Value Bets out of {len(backtest_df):,} total races.")

if len(bets) == 0:
    print("❌ No value bets found.")
    exit()

# Calculate Profit/Loss for a $1 Flat Bet
bets['profit'] = np.where(bets['won'], (1.0 * bets['dec']) - 1.0, -1.0)
bets['bankroll'] = bets['profit'].cumsum()

total_bets = len(bets)
total_profit = bets['profit'].sum()
win_rate = (bets['won'].sum() / total_bets) * 100 if total_bets > 0 else 0
roi = (total_profit / total_bets) * 100

print(f"\n📊 +EV STRATEGY RESULTS:")
print(f"Total Bets Placed: {total_bets:,}")
print(f"Win Rate:          {win_rate:.2f}%")
print(f"Total Profit/Loss: ${total_profit:.2f}")
print(f"Return on Invest.: {roi:.2f}%")

print("4. Generating Profit Chart...")
plt.figure(figsize=(12, 6))

line_color = '#2ecc71' if total_profit > 0 else '#e74c3c'

plt.plot(range(total_bets), bets['bankroll'], color=line_color, linewidth=2)
plt.axhline(0, color='black', linestyle='--', linewidth=1) 
plt.fill_between(range(total_bets), bets['bankroll'], 0, color=line_color, alpha=0.1)

plt.title('Cumulative Profit/Loss (+EV Value Betting Strategy)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Bets Over Time', fontsize=12)
plt.ylabel('Bankroll Profit ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
chart_path = Path("pnl_ev_chart.png")
plt.savefig(chart_path)
print(f"\n✅ Chart generated successfully! Open {chart_path.resolve()} to view it.")