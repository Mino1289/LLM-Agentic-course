import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from datetime import date, datetime

import yfinance as yf

from rag.config import TRACKED_TICKERS
from rag.paths import DATA_DIR, ensure_dir

# Liste des entreprises demandées avec leurs tickers officiels du NASDAQ / NYSE
supported_companies = {
    "nvidia": "NVDA",
    "amd": "AMD",
    "microsoft": "MSFT",
}
entreprises = {
    name: ticker for name, ticker in supported_companies.items() if ticker in TRACKED_TICKERS
}

min_year = int(os.getenv("BOOTSTRAP_MIN_YEAR", "2024"))
max_year = int(os.getenv("BOOTSTRAP_MAX_YEAR", str(datetime.today().year)))
if min_year > max_year:
    raise ValueError("BOOTSTRAP_MIN_YEAR doit être inférieur ou égal à BOOTSTRAP_MAX_YEAR")
date_debut = date(min_year, 1, 1).isoformat()
date_fin = min(date(max_year + 1, 1, 1), datetime.today().date()).isoformat()

ensure_dir(DATA_DIR)

print(f"Début du téléchargement des données du {date_debut} au {date_fin}...")
print(f"Dossier de sortie : {DATA_DIR}\n")

for nom, ticker in entreprises.items():
    print(f"Téléchargement de {nom.upper()} ({ticker})...")
    try:
        df = yf.download(ticker, start=date_debut, end=date_fin)

        if df.empty:
            print(f"⚠️ Aucune donnée trouvée pour {ticker}")
            continue

        if isinstance(df.columns, type(df.columns)):
            if hasattr(df.columns, "get_level_values"):
                df.columns = df.columns.get_level_values(0)

        nom_fichier = DATA_DIR / f"historique_prix_{nom}.csv"
        df.to_csv(nom_fichier)
        print(f"✅ Sauvegardé : {nom_fichier}\n")

    except Exception as e:
        print(f"❌ Erreur lors du traitement de {ticker} : {e}\n")

print(f"Terminé — fichiers CSV dans {DATA_DIR}/")
