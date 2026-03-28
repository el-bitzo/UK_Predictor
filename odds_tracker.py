"""
================================================================================
MODULE 3: LIVE ODDS TRACKER
File: odds_tracker.py

Scrapes current live fractional odds for today's UK races from Oddschecker
(free public aggregator). Converts fractional → implied probability, computes
Odds Momentum (opening vs live), and merges into today_cards.json.

Output: data/today_cards_with_odds.json

Dependencies: selenium, webdriver-manager, beautifulsoup4, requests
================================================================================
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from datetime import date

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

TODAY_CARDS_JSON  = Path("./data/today_cards.json")
ODDS_OUTPUT_JSON  = Path("./data/today_cards_with_odds.json")

# Oddschecker base URL (free, public)
OC_BASE = "https://www.oddschecker.com"
OC_RACING_PATH = "/horse-racing"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

REQUEST_DELAY = 2   # seconds between page requests


# ── Odds Parsing ──────────────────────────────────────────────────────────────

def fractional_to_decimal(frac_str: str) -> float | None:
    """
    '5/1'  → 6.0
    '11/4' → 3.75
    'EVS'  → 2.0
    '1/2'  → 1.5
    """
    if not frac_str:
        return None
    s = str(frac_str).strip().upper()
    if s in ("EVS", "EVENS", "1/1"):
        return 2.0
    m = re.match(r"(\d+\.?\d*)\s*/\s*(\d+\.?\d*)", s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den == 0:
            return None
        return round(num / den + 1.0, 4)
    try:
        return float(s)
    except ValueError:
        return None


def decimal_to_implied_prob(decimal: float | None) -> float | None:
    """Decimal odds → implied probability [0–1]."""
    if decimal is None or decimal <= 0:
        return None
    return round(1.0 / decimal, 4)


def fractional_to_prob(frac_str: str) -> float | None:
    return decimal_to_implied_prob(fractional_to_decimal(frac_str))


# ── Oddschecker Scraper (requests + BeautifulSoup) ───────────────────────────

class OddscheckerScraper:
    """
    Scrapes today's UK horse racing odds from Oddschecker.
    Uses requests + BeautifulSoup (no Selenium needed for static pages).
    Falls back to Selenium for JavaScript-heavy pages.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._driver = None

    # ── Requests path ─────────────────────────────────────────────────────────

    def _get(self, url: str) -> BeautifulSoup | None:
        try:
            log.info(f"GET {url}")
            r = self.session.get(url, timeout=20)
            r.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return BeautifulSoup(r.text, "lxml")
        except Exception as exc:
            log.warning(f"Request failed for {url}: {exc}")
            return None

    def fetch_todays_meetings(self) -> list[dict]:
        """
        Scrape the Oddschecker horse racing hub to get today's meeting links.
        Returns list of { 'meeting': str, 'url': str }
        """
        soup = self._get(f"{OC_BASE}{OC_RACING_PATH}")
        if soup is None:
            return []

        meetings = []
        # Oddschecker lists meetings as links under a racing section
        for a in soup.select("a[href*='/horse-racing/']"):
            href = a.get("href", "")
            text = a.get_text(strip=True)
            # Filter out non-race links
            if (href.startswith("/horse-racing/")
                    and "ante-post" not in href
                    and "specials" not in href
                    and text):
                full_url = OC_BASE + href
                if full_url not in [m["url"] for m in meetings]:
                    meetings.append({"meeting": text, "url": full_url})

        log.info(f"Found {len(meetings)} meeting links on Oddschecker")
        return meetings[:20]   # cap at 20 meetings


    def fetch_race_odds(self, race_url: str) -> dict[str, dict]:
        """
        Scrape a single race page and return:
        { 'Horse Name': { 'odds_frac': '5/1', 'odds_dec': 6.0,
                          'prob': 0.167, 'best_odds_frac': '11/2' } }
        """
        soup = self._get(race_url)
        if soup is None:
            return {}

        results = {}

        # ── Strategy 1: Standard Oddschecker table ─────────────────────────
        # Each horse row has class 'diff-row' or similar
        rows = soup.select("tr.diff-row, tr[data-bname]")
        for row in rows:
            name_el = (row.select_one("td.name a")
                       or row.select_one("td.horse-name")
                       or row.select_one("[data-bname]"))
            if not name_el:
                continue
            horse_name = name_el.get_text(strip=True)
            if not horse_name:
                continue

            # Best odds cell is usually the first odds column
            odds_cells = row.select("td.bc, td[data-odig]")
            raw_odds = None
            if odds_cells:
                raw_odds = odds_cells[0].get_text(strip=True)

            # Opening odds
            open_el = row.select_one("td.open-price, [class*='opening']")
            open_odds = open_el.get_text(strip=True) if open_el else None

            dec   = fractional_to_decimal(raw_odds)
            prob  = decimal_to_implied_prob(dec)
            o_dec = fractional_to_decimal(open_odds)

            results[horse_name] = {
                "odds_frac":    raw_odds,
                "odds_dec":     dec,
                "prob":         prob,
                "open_odds_frac": open_odds,
                "open_odds_dec":  o_dec,
                "odds_momentum":  _calc_momentum(o_dec, dec),
            }

        # ── Strategy 2: JSON-LD / embedded data ───────────────────────────
        if not results:
            results = self._parse_json_ld(soup)

        log.info(f"  {race_url.split('/')[-1]} → {len(results)} runners with odds")
        return results


    def _parse_json_ld(self, soup: BeautifulSoup) -> dict[str, dict]:
        """Try extracting odds from embedded JSON-LD or script tags."""
        results = {}
        for script in soup.find_all("script", type="application/json"):
            try:
                data = json.loads(script.string or "")
                # Look for runners/horses arrays in the blob
                runners = (data.get("runners")
                           or data.get("horses")
                           or data.get("selections", []))
                for r in runners:
                    name = r.get("name") or r.get("horseName") or r.get("horse")
                    odds = (r.get("odds") or r.get("price")
                            or r.get("bestOdds") or r.get("sp"))
                    if name:
                        dec  = fractional_to_decimal(str(odds))
                        prob = decimal_to_implied_prob(dec)
                        results[name] = {
                            "odds_frac":     str(odds),
                            "odds_dec":      dec,
                            "prob":          prob,
                            "open_odds_frac": None,
                            "open_odds_dec":  None,
                            "odds_momentum":  None,
                        }
            except (json.JSONDecodeError, AttributeError):
                continue
        return results


    # ── Selenium fallback ─────────────────────────────────────────────────────

    def _get_driver(self):
        """Lazily initialise headless Chrome via webdriver-manager."""
        if self._driver:
            return self._driver
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager

            opts = Options()
            opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument("--disable-gpu")
            opts.add_argument(f"user-agent={HEADERS['User-Agent']}")

            self._driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=opts,
            )
            return self._driver
        except Exception as exc:
            log.warning(f"Selenium driver setup failed: {exc}")
            return None


    def fetch_with_selenium(self, url: str) -> dict[str, dict]:
        """Use headless Chrome for JS-heavy pages."""
        driver = self._get_driver()
        if not driver:
            return {}
        try:
            driver.get(url)
            time.sleep(4)   # wait for JS to render
            soup = BeautifulSoup(driver.page_source, "lxml")
            return self.fetch_race_odds.__wrapped__(self, soup) if hasattr(
                self.fetch_race_odds, "__wrapped__") else self._parse_soup(soup)
        except Exception as exc:
            log.warning(f"Selenium fetch failed: {exc}")
            return {}

    def _parse_soup(self, soup: BeautifulSoup) -> dict[str, dict]:
        """Same parsing logic, takes a pre-loaded soup."""
        return self.fetch_race_odds.__func__(self, "") if False else (
            self._parse_json_ld(soup)
        )

    def close(self):
        if self._driver:
            self._driver.quit()
            self._driver = None


# ── Odds Momentum Calculator ──────────────────────────────────────────────────

def _calc_momentum(open_dec: float | None, live_dec: float | None) -> float | None:
    """
    Odds Momentum = (opening_prob - live_prob).
    Positive → drifting out (support fading).
    Negative → shortening (support growing — GOOD signal).
    """
    if open_dec is None or live_dec is None:
        return None
    open_prob = decimal_to_implied_prob(open_dec)
    live_prob = decimal_to_implied_prob(live_dec)
    if open_prob is None or live_prob is None:
        return None
    return round(open_prob - live_prob, 4)


# ── Racing Post Fallback Scraper ──────────────────────────────────────────────

class RacingPostScraper:
    """
    Lightweight fallback — scrapes odds from Racing Post free race pages.
    Less structured than Oddschecker but useful as a backup.
    """

    BASE = "https://www.racingpost.com"
    HEADERS = HEADERS

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def fetch_todays_odds(self) -> dict[str, float]:
        """
        Returns { 'Horse Name': implied_probability } from today's cards page.
        """
        today_str = date.today().strftime("%Y-%m-%d")
        url = f"{self.BASE}/racecards/{today_str}/"
        log.info(f"RacingPost fallback: {url}")

        try:
            r = self.session.get(url, timeout=20)
            r.raise_for_status()
        except Exception as exc:
            log.warning(f"RacingPost request failed: {exc}")
            return {}

        soup  = BeautifulSoup(r.text, "lxml")
        probs: dict[str, float] = {}

        for runner_div in soup.select("[class*='RC-runnerName'], [data-test-selector='RC-runnerName']"):
            name = runner_div.get_text(strip=True)
            # Find sibling odds element
            parent = runner_div.find_parent()
            if not parent:
                continue
            odds_el = parent.select_one(
                "[class*='RC-runnerOdds'], [data-test-selector*='odds']"
            )
            if odds_el:
                p = fractional_to_prob(odds_el.get_text(strip=True))
                if name and p:
                    probs[name] = p

        log.info(f"RacingPost: {len(probs)} odds found")
        return probs


# ── Main Merge Function ───────────────────────────────────────────────────────

def fetch_and_merge_odds(
    cards_path:  Path = TODAY_CARDS_JSON,
    output_path: Path = ODDS_OUTPUT_JSON,
) -> list[dict]:
    """
    1. Load today_cards.json
    2. Scrape live odds from Oddschecker (with RacingPost fallback)
    3. Merge odds into each runner entry
    4. Write today_cards_with_odds.json
    Returns the enriched race list.
    """
    if not cards_path.exists():
        raise FileNotFoundError(
            f"{cards_path} not found. Run data_loader.py first."
        )

    raw = json.loads(cards_path.read_text(encoding="utf-8"))

    # Normalise: handle both list-of-races and {'races': [...]} formats
    if isinstance(raw, list):
        races = raw
    elif isinstance(raw, dict) and "races" in raw:
        races = raw["races"]
    else:
        races = []

    if not races:
        log.warning("today_cards.json has no races. Saving as-is.")
        output_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        return races

    scraper = OddscheckerScraper()
    rp      = RacingPostScraper()
    rp_odds = {}

    try:
        # ── Primary: Oddschecker ──────────────────────────────────────────────
        meetings = scraper.fetch_todays_meetings()
        meeting_odds: dict[str, dict[str, dict]] = {}

        for meeting in meetings:
            race_data = scraper.fetch_race_odds(meeting["url"])
            if not race_data:
                log.info(f"Trying Selenium for {meeting['url']}")
                race_data = scraper.fetch_with_selenium(meeting["url"])
            # Key by meeting name (we'll fuzzy-match to course names)
            meeting_odds[meeting["meeting"].lower()] = race_data

        # ── Fallback: Racing Post ─────────────────────────────────────────────
        if not any(meeting_odds.values()):
            log.info("Oddschecker returned no odds — trying RacingPost …")
            rp_odds = rp.fetch_todays_odds()

        # ── Merge into race cards ─────────────────────────────────────────────
        for race in races:
            course_lower = race.get("course", "").lower()

            # Find best matching meeting odds dict
            race_odds_dict: dict[str, dict] = {}
            for key, odds_dict in meeting_odds.items():
                if course_lower in key or key in course_lower:
                    race_odds_dict = odds_dict
                    break

            for runner in race.get("runners", []):
                horse = runner.get("horse", "")

                # Try exact match first, then case-insensitive
                odds_entry = (race_odds_dict.get(horse)
                              or _fuzzy_match(horse, race_odds_dict))

                if odds_entry:
                    runner["live_odds_frac"]   = odds_entry.get("odds_frac")
                    runner["live_odds_dec"]     = odds_entry.get("odds_dec")
                    runner["live_prob"]         = odds_entry.get("prob")
                    runner["open_odds_frac"]    = odds_entry.get("open_odds_frac")
                    runner["odds_momentum"]     = odds_entry.get("odds_momentum")
                elif horse in rp_odds:
                    runner["live_prob"]         = rp_odds[horse]
                    runner["live_odds_frac"]    = None
                    runner["live_odds_dec"]     = None
                    runner["open_odds_frac"]    = None
                    runner["odds_momentum"]     = None
                else:
                    # Use odds from racecard if live not available
                    card_odds = runner.get("odds", "")
                    runner["live_odds_frac"]    = card_odds or None
                    runner["live_odds_dec"]     = fractional_to_decimal(card_odds)
                    runner["live_prob"]         = fractional_to_prob(card_odds)
                    runner["open_odds_frac"]    = card_odds or None
                    runner["odds_momentum"]     = 0.0

    finally:
        scraper.close()

    # Save
    output = raw if isinstance(raw, dict) else races
    if isinstance(output, dict):
        output["races"] = races
    else:
        output = races

    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    log.info(f"\n✅ today_cards_with_odds.json → {output_path.resolve()}")
    return races


def _fuzzy_match(name: str, odds_dict: dict) -> dict | None:
    """Case-insensitive partial match for horse name."""
    name_lower = name.lower()
    for key, val in odds_dict.items():
        if name_lower in key.lower() or key.lower() in name_lower:
            return val
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  MODULE 3 — Live Odds Tracker")
    log.info("=" * 60)
    fetch_and_merge_odds()
    log.info("\n✅ Module 3 complete. Run model.py next.")
