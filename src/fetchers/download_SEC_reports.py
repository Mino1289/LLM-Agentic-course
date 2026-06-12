#!/usr/bin/env python3
"""Téléchargement des rapports SEC (10-K, 10-Q, 8-K, 20-F, 6-K) via l'API EDGAR."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

from src.config import TRACKED_TICKERS
from src.paths import DATA_DIR, ENV_FILE, PROJECT_ROOT, SEC_FILINGS_METADATA, ensure_dir

load_dotenv(ENV_FILE)

USER_AGENT = os.getenv("SEC_USER_AGENT", "").strip()
if not USER_AGENT:
    raise ValueError("SEC_USER_AGENT is required in .env (example: MyFinanceRAG your-email@example.com)")

HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

SEC_TICKERS = tuple(t for t in TRACKED_TICKERS if not t.endswith(".PA"))
SKIPPED_TICKERS = tuple(t for t in TRACKED_TICKERS if t.endswith(".PA"))
TICKERS = list(SEC_TICKERS)

_SEC_DEFAULT_INTER_TICKER_SLEEP = 0.15
_SEC_MIN_INTER_TICKER_SLEEP = 0.10
_SEC_HARD_CAP_INTER_TICKER_SLEEP = 0.05


def _get_inter_ticker_sleep() -> float:
    raw = os.getenv("SEC_INTER_TICKER_SLEEP", str(_SEC_DEFAULT_INTER_TICKER_SLEEP))
    try:
        value = float(raw)
    except ValueError:
        return _SEC_DEFAULT_INTER_TICKER_SLEEP
    if value < _SEC_HARD_CAP_INTER_TICKER_SLEEP:
        return _SEC_HARD_CAP_INTER_TICKER_SLEEP
    if value < _SEC_MIN_INTER_TICKER_SLEEP:
        return _SEC_MIN_INTER_TICKER_SLEEP
    return value


_INTER_TICKER_SLEEP = _get_inter_ticker_sleep()


def rate_limit_sleep():
    time.sleep(_INTER_TICKER_SLEEP)


def get_cik_mapping() -> dict[str, str]:
    print("--- Récupération de la table des correspondances Tickers/CIK ---")
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    rate_limit_sleep()
    mapping: dict[str, str] = {}
    for entry in response.json().values():
        ticker = entry["ticker"].upper()
        if ticker in TICKERS:
            mapping[ticker] = str(entry["cik_str"]).zfill(10)
    return mapping


def get_filings(ticker: str, cik: str, min_year: int, max_year: int) -> list[dict]:
    print(f"\nRecherche des rapports pour {ticker} (CIK: {cik})...")
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Impossible de récupérer les données pour {ticker}")
        return []
    rate_limit_sleep()
    data = response.json()
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return []
    filings_found: list[dict] = []
    for i in range(len(recent["form"])):
        form = recent["form"][i]
        filing_date_str = recent["filingDate"][i]
        filing_year = int(filing_date_str.split("-")[0])
        if not (min_year <= filing_year <= max_year):
            continue
        is_wanted = False
        if form == "8-K":
            items_raw = (recent.get("items") or [])[i] if i < len(recent.get("items") or []) else ""
            items = items_raw if isinstance(items_raw, list) else str(items_raw).split(",")
            items = [item.strip() for item in items]
            if "2.02" in items:
                is_wanted = True
        elif form in {"10-K", "10-Q"}:
            is_wanted = True
        elif form in {"20-F", "6-K"}:
            is_wanted = True
        if is_wanted:
            accession_num = recent["accessionNumber"][i]
            accession_clean = accession_num.replace("-", "")
            primary_doc = recent["primaryDocument"][i]
            cik_int = int(cik)
            filings_found.append({
                "ticker": ticker, "form": form, "date": filing_date_str,
                "accessionNumber": accession_num,
                "indexUrl": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/{accession_num}-index.htm",
                "documentUrl": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/{primary_doc}",
            })
    return filings_found


def download_filing(ticker: str, form: str, date: str, url: str) -> str | None:
    ensure_dir(DATA_DIR)
    ext = url.split(".")[-1]
    filename = DATA_DIR / f"{ticker.lower()}-{form.lower()}_{date}.{ext}"
    portable_path = filename.relative_to(PROJECT_ROOT).as_posix()
    if filename.exists():
        return portable_path
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        filename.write_bytes(response.content)
        rate_limit_sleep()
        return portable_path
    except Exception as e:
        print(f"  [Erreur] Impossible de télécharger {url} : {e}")
        return None


def parse_args() -> argparse.Namespace:
    current_year = datetime.now().year
    parser = argparse.ArgumentParser(description="Télécharger les rapports SEC de l'univers debug.")
    parser.add_argument("--min-year", type=int, default=int(os.getenv("BOOTSTRAP_MIN_YEAR", str(current_year - 2))))
    parser.add_argument("--max-year", type=int, default=int(os.getenv("BOOTSTRAP_MAX_YEAR", str(current_year))))
    args = parser.parse_args()
    if args.min_year > args.max_year:
        parser.error("--min-year doit être inférieur ou égal à --max-year")
    return args


def main():
    args = parse_args()
    if SKIPPED_TICKERS:
        print(f"⏭  {len(SKIPPED_TICKERS)} ticker(s) sans CIK SEC (Euronext), skip auto : {', '.join(SKIPPED_TICKERS)}")
    print(f"⏱  Pacing SEC : {_INTER_TICKER_SLEEP}s entre tickers (SEC_INTER_TICKER_SLEEP={os.getenv('SEC_INTER_TICKER_SLEEP', 'default')})")
    print(f"📋 Univers SEC : {len(TICKERS)} tickers ({len(SKIPPED_TICKERS)} skip)\n")
    cik_map = get_cik_mapping()
    results: dict[str, list[dict]] = {}
    for ticker in TICKERS:
        if ticker in cik_map:
            cik = cik_map[ticker]
            filings = get_filings(ticker, cik, args.min_year, args.max_year)
            results[ticker] = filings
            print(f"-> {len(filings)} documents trouvés (8-K 2.02, 10-K, 10-Q) pour {ticker} entre {args.min_year} et {args.max_year}.")
            for f in filings:
                local_path = download_filing(f["ticker"], f["form"], f["date"], f["documentUrl"])
                if local_path:
                    f["localPath"] = local_path
        else:
            print(f"Ticker {ticker} non trouvé dans le registre de la SEC.")
    SEC_FILINGS_METADATA.write_text(json.dumps(results, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Terminé] Métadonnées : {SEC_FILINGS_METADATA}")
    print(f"Fichiers HTML : {DATA_DIR}/")


if __name__ == "__main__":
    main()
