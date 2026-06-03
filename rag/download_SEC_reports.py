import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
import json
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

from rag.config import TRACKED_TICKERS
from rag.paths import DATA_DIR, ENV_FILE, PROJECT_ROOT, SEC_FILINGS_METADATA, ensure_dir

load_dotenv(ENV_FILE)

# ---------------------------------------------------------------------------
# CONFIGURATION OBLIGATOIRE (Règle SEC)
# Changez ces valeurs avec vos propres informations pour éviter le blocage IP.
# ---------------------------------------------------------------------------
USER_AGENT = os.getenv("SEC_USER_AGENT", "").strip()
if not USER_AGENT:
    raise ValueError(
        "SEC_USER_AGENT is required in .env (example: MyFinanceRAG your-email@example.com)"
    )
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate"
}

# Liste des tickers ciblés (univers debug volontairement limité).
TICKERS = list(TRACKED_TICKERS)

def rate_limit_sleep():
    """Garantit le respect de la limite de la SEC (max 10 req/sec)"""
    time.sleep(0.15)

def get_cik_mapping():
    """Récupère le dictionnaire officiel Ticker -> CIK de la SEC"""
    print("--- Récupération de la table des correspondances Tickers/CIK ---")
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    rate_limit_sleep()
    
    mapping = {}
    data = response.json()
    for entry in data.values():
        ticker = entry["ticker"].upper()
        if ticker in TICKERS:
            # Formate le CIK sur 10 chiffres avec des zéros au début
            mapping[ticker] = str(entry["cik_str"]).zfill(10)
    return mapping

def get_filings(ticker, cik, min_year, max_year):
    """Cherche les formulaires 8-K (Item 2.02), 10-K et 10-Q de la plage demandée."""
    print(f"\nRecherche des rapports pour {ticker} (CIK: {cik})...")
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Impossible de récupérer les données pour {ticker}")
        return []
    
    rate_limit_sleep()
    data = response.json()
    recent_filings = data.get("filings", {}).get("recent", {})
    
    if not recent_filings:
        return []
    
    filings_found = []
    # Parcourt les documents soumis
    for i in range(len(recent_filings["form"])):
        form = recent_filings["form"][i]
        filing_date_str = recent_filings["filingDate"][i]
        filing_year = int(filing_date_str.split("-")[0])
        
        if not (min_year <= filing_year <= max_year):
            continue

        is_wanted = False
        if form == "8-K":
            # Récupère la liste des items associés au 8-K (ex: "2.02", "9.01")
            filing_items = recent_filings.get("items", [])
            items_raw = filing_items[i] if i < len(filing_items) else ""
            items = items_raw if isinstance(items_raw, list) else str(items_raw).split(",")
            items = [item.strip() for item in items]
            
            # 2.02 = Communiqué de presse sur les résultats financiers
            if "2.02" in items:
                is_wanted = True
        elif form in {"10-K", "10-Q"}:
            is_wanted = True
            
        if is_wanted:
            accession_num = recent_filings["accessionNumber"][i]
            accession_clean = accession_num.replace("-", "")
            primary_doc = recent_filings["primaryDocument"][i]
            
            # Reconstruction de l'URL vers la page d'index du dépôt
            index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{accession_num}-index.htm"
            
            # Reconstruction du lien direct vers le corps principal
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{primary_doc}"
            
            filings_found.append({
                "ticker": ticker,
                "form": form,
                "date": filing_date_str,
                "accessionNumber": accession_num,
                "indexUrl": index_url,
                "documentUrl": doc_url
            })
                
    return filings_found

def download_filing(ticker, form, date, url):
    """Télécharge le document et le sauvegarde dans data/ à la racine du projet."""
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

def parse_args():
    current_year = datetime.now().year
    parser = argparse.ArgumentParser(description="Télécharger les rapports SEC de l'univers debug.")
    parser.add_argument(
        "--min-year",
        type=int,
        default=int(os.getenv("BOOTSTRAP_MIN_YEAR", str(current_year - 2))),
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=int(os.getenv("BOOTSTRAP_MAX_YEAR", str(current_year))),
    )
    args = parser.parse_args()
    if args.min_year > args.max_year:
        parser.error("--min-year doit être inférieur ou égal à --max-year")
    return args


def main():
    args = parse_args()
    # 1. Obtenir les identifiants SEC (CIK) des entreprises
    cik_map = get_cik_mapping()
    
    results = {}
    
    # 2. Extraction des documents pour chaque entreprise
    for ticker in TICKERS:
        if ticker in cik_map:
            cik = cik_map[ticker]
            filings = get_filings(ticker, cik, args.min_year, args.max_year)
            results[ticker] = filings
            print(
                f"-> {len(filings)} documents trouvés (8-K 2.02, 10-K, 10-Q) "
                f"pour {ticker} entre {args.min_year} et {args.max_year}."
            )
            
            # 3. Téléchargement effectif des fichiers
            for f in filings:
                local_path = download_filing(f["ticker"], f["form"], f["date"], f["documentUrl"])
                if local_path:
                    f["localPath"] = local_path
        else:
            print(f"Ticker {ticker} non trouvé dans le registre de la SEC.")
            
    # 4. Sauvegarde des métadonnées et des URLs dans un fichier JSON
    SEC_FILINGS_METADATA.write_text(
        json.dumps(results, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n[Terminé] Métadonnées : {SEC_FILINGS_METADATA}")
    print(f"Fichiers HTML : {DATA_DIR}/")

if __name__ == "__main__":
    main()
