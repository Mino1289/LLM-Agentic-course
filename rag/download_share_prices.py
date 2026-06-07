import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import time
from datetime import date, datetime
from typing import Any, Optional

import yfinance as yf

from rag.config import TRACKED_TICKERS
from rag.embedding_pipeline import BackoffConfig, with_exponential_backoff
from rag.paths import DATA_DIR, ensure_dir

# Liste des entreprises demandées avec leurs tickers officiels.
# Mode test/debug: seules les valeurs présentes dans TRACKED_TICKERS sont téléchargées.
# Le nom (slug) sert à nommer le CSV historique_prix_<slug>.csv.
supported_companies = {
    "nvidia": "NVDA",
    "asml": "ASML",
    "tsm": "TSM",
    "amd": "AMD",
    "broadcom": "AVGO",
    "arm": "ARM",
    "microsoft": "MSFT",
    "apple": "AAPL",
    "intel": "INTC",
    "qualcomm": "QCOM",
    "lvmh": "MC.PA",
    "hermes": "RMS.PA",
    "kering": "KER.PA",
    "airbus": "AIR.PA",
    "totalenergies": "TTE.PA",
    "berkshire": "BRK-B",
    "jpmorgan": "JPM",
    "caterpillar": "CAT",
    "nike": "NKE",
    "exxonmobil": "XOM",
}
entreprises = {
    name: ticker for name, ticker in supported_companies.items() if ticker in TRACKED_TICKERS
}


def _resolve_window(min_year: int, max_year: int) -> tuple[str, str]:
    if min_year > max_year:
        raise ValueError("BOOTSTRAP_MIN_YEAR doit être inférieur ou égal à BOOTSTRAP_MAX_YEAR")
    date_debut = date(min_year, 1, 1).isoformat()
    date_fin = min(date(max_year + 1, 1, 1), datetime.today().date()).isoformat()
    return date_debut, date_fin


def download_with_retry(
    ticker: str,
    date_debut: str,
    date_fin: str,
    *,
    max_retries: int = 3,
    backoff_config: Optional[BackoffConfig] = None,
) -> Any:
    """Download a single ticker with exponential backoff on transient errors.

    Permanent errors (auth, invalid symbol) fail-fast via
    :func:`rag.embedding_pipeline.is_permanent_error`.
    """
    config = backoff_config or BackoffConfig(max_retries=max_retries)

    def _call() -> Any:
        df = yf.download(ticker, start=date_debut, end=date_fin)
        if isinstance(df.columns, type(df.columns)) and hasattr(df.columns, "get_level_values"):
            df.columns = df.columns.get_level_values(0)
        return df

    return with_exponential_backoff(_call, config=config)


def download_all_entreprises(
    date_debut: str,
    date_fin: str,
    *,
    inter_ticker_sleep: float = 1.0,
    backoff_config: Optional[BackoffConfig] = None,
    out_dir: Optional[Path] = None,
) -> dict[str, str]:
    """Download all tracked tickers with polite pacing and a reliability summary.

    Returns a ``{ticker: status}`` map (``"ok"``, ``"empty"``, ``"error:<msg>"``).
    """
    ensure_dir(out_dir or DATA_DIR)
    summary: dict[str, str] = {}
    for index, (nom, ticker) in enumerate(entreprises.items()):
        print(f"Téléchargement de {nom.upper()} ({ticker})...")
        try:
            df = download_with_retry(
                ticker,
                date_debut,
                date_fin,
                backoff_config=backoff_config,
            )
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {ticker} : {e}\n")
            summary[ticker] = f"error:{e}"
            continue

        if df is None or df.empty:
            print(f"⚠️ Aucune donnée trouvée pour {ticker}")
            summary[ticker] = "empty"
            continue

        nom_fichier = (out_dir or DATA_DIR) / f"historique_prix_{nom}.csv"
        df.to_csv(nom_fichier)
        print(f"✅ Sauvegardé : {nom_fichier}")
        summary[ticker] = "ok"

        if inter_ticker_sleep > 0 and index < len(entreprises) - 1:
            time.sleep(inter_ticker_sleep)
    return summary


def _print_summary(summary: dict[str, str]) -> None:
    if not summary:
        return
    ok = sum(1 for v in summary.values() if v == "ok")
    empty = sum(1 for v in summary.values() if v == "empty")
    failed = sum(1 for v in summary.values() if v.startswith("error"))
    print(
        f"\nRésumé : {ok} ok, {empty} sans données, {failed} erreur(s) — "
        f"fichiers CSV dans {DATA_DIR}/"
    )


if __name__ == "__main__":
    min_year = int(os.getenv("BOOTSTRAP_MIN_YEAR", "2024"))
    max_year = int(os.getenv("BOOTSTRAP_MAX_YEAR", str(datetime.today().year)))
    date_debut, date_fin = _resolve_window(min_year, max_year)

    print(f"Début du téléchargement des données du {date_debut} au {date_fin}...")
    print(f"Dossier de sortie : {DATA_DIR}\n")
    print(f"📋 Univers : {len(entreprises)} tickers (yfinance accepte tout, dont .PA)")

    inter_sleep = float(os.getenv("YFINANCE_INTER_TICKER_SLEEP", "1.0"))
    summary = download_all_entreprises(
        date_debut,
        date_fin,
        inter_ticker_sleep=inter_sleep,
        backoff_config=BackoffConfig.from_env(),
    )
    _print_summary(summary)
