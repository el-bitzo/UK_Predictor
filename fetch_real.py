"""
================================================================================
TODAY'S RACECARDS (SELENIUM CHROME) - SMART PARSER
File: fetch_real.py
================================================================================
"""
import sys
import json
import time
from pathlib import Path

try:
    import undetected_chromedriver as uc
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"\n❌ THE EXACT ERROR: {e}")
    sys.exit(1)

def get_decimal(odds_str):
    try:
        if odds_str.upper() in ["EVS", "EVENS", "1/1"]: return 2.0
        n, d = odds_str.split('/')
        return (float(n) / float(d)) + 1.0
    except:
        return 0.0

def main():
    data_dir = Path("data").resolve()
    data_dir.mkdir(exist_ok=True)

    print("🚀 Opening Chrome to clear Cloudflare...")
    options = uc.ChromeOptions()
    
    try:
        driver = uc.Chrome(options=options, use_subprocess=True, version_main=146)
    except Exception as e:
        print(f"❌ Failed to launch Chrome. Error: {e}")
        sys.exit(1)
    
    print("Navigating to Racing Post...")
    driver.get("https://www.racingpost.com/racecards/")
    
    print("\n" + "="*65)
    print(" 🛑 ACTION REQUIRED:")
    print(" 1. Look at the Chrome window that just opened.")
    print(" 2. If asked, click the 'I am human' box to pass Cloudflare.")
    print(" 3. Wait until you see the list of today's courses and races.")
    print("="*65)
    
    input("\nPress ENTER here in the console once the racecards have loaded... ")
    
    print("\n🔍 Extracting race links from the main page...")
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    full_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/racecards/" in href and len(href.split("/")) > 4 and "results" not in href:
            link = "https://www.racingpost.com" + href if href.startswith("/") else href
            if link not in full_links:
                full_links.append(link)
                
    if not full_links:
        print("❌ Could not find any races. Racing Post layout might have changed.")
        try: driver.quit() 
        except: pass
        sys.exit(1)
        
    print(f"✅ Found {len(full_links)} races! Scraping them now...\n")
    
    real_races = []
    
    for i, link in enumerate(full_links):
        print(f"  Scraping race {i+1}/{len(full_links)}...")
        driver.get(link)
        
        # SMART WAIT: Give the page time to actually render the horses
        time.sleep(2.5) 
        
        race_soup = BeautifulSoup(driver.page_source, "html.parser")
        
        parts = link.split("/")
        try:
            course = parts[5].replace("-", " ").title()
            race_id = parts[7]
        except:
            course = "Unknown"
            race_id = "0000"
            
        time_str = "00:00"
        time_elem = race_soup.find(attrs={"data-test-selector": "RC-courseHeader__time"})
        if not time_elem: time_elem = race_soup.find(class_=lambda c: c and 'time' in c.lower())
        if time_elem: time_str = time_elem.text.strip()
        
        runners = []
        
        # SMART SELECTOR: Look for multiple different possible Row classes
        rows = race_soup.find_all(attrs={"data-test-selector": "RC-runnerRow"})
        if not rows: 
            rows = race_soup.find_all(class_=lambda c: c and 'runnerrow' in c.lower().replace('-', '').replace('_', ''))
        if not rows:
            rows = race_soup.find_all("tr", class_=lambda c: c and 'runner' in c.lower())
        
        for row in rows:
            horse = "Unknown"
            jockey = "Unknown"
            odds = "SP"
            
            # Find Horse Name
            h_elem = row.find(attrs={"data-test-selector": "RC-runnerName"})
            if not h_elem: h_elem = row.find(class_=lambda c: c and 'runnername' in c.lower().replace('-', ''))
            if not h_elem: h_elem = row.find("a", class_=lambda c: c and 'profilelink' in c.lower())
            if h_elem: horse = h_elem.text.strip().split('\n')[0] # Clean up newlines
            
            # Find Jockey
            j_elem = row.find(attrs={"data-test-selector": "RC-runnerJockey"})
            if not j_elem: j_elem = row.find(class_=lambda c: c and 'jockey' in c.lower())
            if j_elem: jockey = j_elem.text.strip().replace('J: ', '')
            
            # Find Odds
            o_elem = row.find(attrs={"data-test-selector": "RC-runnerPrice"})
            if not o_elem: o_elem = row.find(class_=lambda c: c and 'price' in c.lower())
            if not o_elem: o_elem = row.find(class_=lambda c: c and 'odds' in c.lower())
            if o_elem: odds = o_elem.text.strip()
            
            if horse != "Unknown":
                dec = get_decimal(odds)
                runners.append({
                    "horse": horse,
                    "jockey": jockey,
                    "odds": odds,
                    "live_odds_frac": odds,
                    "live_odds_dec": dec,
                    "live_prob": 1.0 / dec if dec > 0 else 0.05,
                    "open_odds_frac": odds,
                    "odds_momentum": 0.0,
                    "weight": "", 
                    "age": ""     
                })
            
        if runners:
            real_races.append({
                "course": course,
                "time": time_str,
                "race_id": race_id,
                "runners": runners
            })
            
    try:
        driver.quit()
    except:
        pass
    
    if not real_races:
        print("\n❌ Scraper finished but extracted 0 runners. Racing Post layout completely changed.")
        sys.exit(1)
        
    cards_file = data_dir / "today_cards.json"
    odds_file = data_dir / "today_cards_with_odds.json"
    
    cards_file.write_text(json.dumps(real_races, indent=2), encoding="utf-8")
    odds_file.write_text(json.dumps(real_races, indent=2), encoding="utf-8")
    
    print("\n" + "="*65)
    print("✅ SUCCESS! Real UK race data saved automatically.")
    print(f"Loaded {sum(len(r['runners']) for r in real_races)} real horses across {len(real_races)} races.")
    print("="*65)

if __name__ == "__main__":
    main()