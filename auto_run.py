import os
import sys
import time

C_CYAN = "\033[96m"
C_GREEN = "\033[92m"
C_MAGENTA = "\033[95m"
C_RED = "\033[91m"
C_RESET = "\033[0m"

os.system("") # Enable colors

def run_pipeline():
    print(f"\n{C_MAGENTA}======================================================================{C_RESET}")
    print(f"{C_CYAN} 🚀 TRACKSNIPER PRO - ONE CLICK PIPELINE{C_RESET}")
    print(f"{C_MAGENTA}======================================================================{C_RESET}")

    # Step 1: Fetch Live Horses
    print(f"\n{C_CYAN}[1/2] Launching Chrome to bypass Cloudflare and scrape today's races...{C_RESET}")
    time.sleep(1)
    exit_code = os.system(f"{sys.executable} fetch_real.py")
    
    if exit_code != 0:
        print(f"\n{C_RED}❌ ERROR: Scraper failed or was closed early. Pipeline stopped.{C_RESET}")
        sys.exit(1)

    # Step 2: Generate Predictions
    print(f"\n{C_CYAN}[2/2] Feeding live data to the AI Prediction Engine...{C_RESET}")
    time.sleep(1)
    exit_code = os.system(f"{sys.executable} main.py --predict-only")

    if exit_code != 0:
        print(f"\n{C_RED}❌ ERROR: AI Prediction Engine failed. Pipeline stopped.{C_RESET}")
        sys.exit(1)
        
    print(f"\n{C_GREEN}✅ PIPELINE COMPLETE! Predictions saved to the /data folder.{C_RESET}\n")

if __name__ == "__main__":
    run_pipeline()