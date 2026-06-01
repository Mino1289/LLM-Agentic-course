import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime

import yfinance as yf

from rag.paths import DATA_DIR, ensure_dir

# Liste des entreprises demandées avec leurs tickers officiels du NASDAQ / NYSE
entreprises = {
    "nvidia": "NVDA",
    "intel": "INTC",
    "amd": "AMD",
    "palantir": "PLTR",
    "google": "GOOGL",
    "meta": "META",
    "amazon": "AMZN",
    "microsoft": "MSFT",
    "broadcom": "AVGO",
    "oracle": "ORCL",
}

date_debut = "2021-01-01"
date_fin = datetime.today().strftime("%Y-%m-%d")

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
