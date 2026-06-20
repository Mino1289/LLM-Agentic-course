#!/usr/bin/env python3
"""Téléchargement des transcripts d'earnings calls (prepared remarks + Q&A).

Les transcripts ne sont PAS sur EDGAR (le 8-K item 2.02 n'est que le communiqué
de presse). On passe par Alpha Vantage (endpoint EARNINGS_CALL_TRANSCRIPT, clé
gratuite `ALPHAVANTAGE_API_KEY`). Les fichiers sont écrits dans `data/` sous le
nom `{ticker}-earnings_call_{YYYY}q{Q}.txt` — contrat de nommage imposé par le
pipeline existant (extract_ticker_from_filename / extract_year_from_filename /
is_earnings_call_filename) — puis ramassés tels quels par le preprocess et
l'indexation (file_type=EARNINGS_CALL, section=earnings_call).

Sans clé API, le script warn et sort proprement (la pipeline reste fonctionnelle).

Note free tier Alpha Vantage : 25 requêtes/jour, 5/min. Pour 10 tickers × N années
× 4 trimestres, restreins la plage (--min-year/--max-year) ou étale sur plusieurs
jours ; le script ignore proprement les réponses "rate limit".
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

from src.config import TRACKED_TICKERS
from src.paths import DATA_DIR, ENV_FILE, PROJECT_ROOT, ensure_dir

load_dotenv(ENV_FILE)

EARNINGS_CALLS_METADATA = DATA_DIR / "earnings_calls_metadata.json"

# Longueur minimale d'un transcript pour être conservé (le preprocess re-vérifie
# aussi un seuil de 200 caractères côté src/preprocess/cli.py).
_MIN_TRANSCRIPT_CHARS = 200

_DEFAULT_INTER_CALL_SLEEP = 0.25
_MIN_INTER_CALL_SLEEP = 0.05

_AV_BASE_URL = "https://www.alphavantage.co/query"


def _get_inter_call_sleep() -> float:
    raw = os.getenv("EARNINGS_INTER_CALL_SLEEP", str(_DEFAULT_INTER_CALL_SLEEP))
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_INTER_CALL_SLEEP
    return max(value, _MIN_INTER_CALL_SLEEP)


_INTER_CALL_SLEEP = _get_inter_call_sleep()


def rate_limit_sleep() -> None:
    time.sleep(_INTER_CALL_SLEEP)


def _api_key() -> str:
    return os.getenv("ALPHAVANTAGE_API_KEY", "").strip()


def fetch_transcript(
    ticker: str, year: int, quarter: int, api_key: str
) -> str | None:
    """Retourne le texte du transcript pour un call, ou None si absent.

    Réseau isolé ici pour faciliter le mock dans les tests.
    """
    params = {
        "function": "EARNINGS_CALL_TRANSCRIPT",
        "symbol": ticker.upper(),
        "quarter": f"{year}Q{quarter}",
        "apikey": api_key,
    }
    try:
        response = requests.get(_AV_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
    except Exception as e:  # noqa: BLE001 — on log et on continue
        print(f"  [Erreur] {ticker} {year}Q{quarter} : {e}")
        return None
    rate_limit_sleep()
    return parse_transcript_payload(response.json())


def parse_transcript_payload(payload: object) -> str | None:
    """Extrait le texte du transcript d'une réponse Alpha Vantage.

    None si vide/trop court, ou si Alpha Vantage renvoie un message
    d'erreur / de rate limit (clés "Information", "Note", "Error Message").
    """
    if not isinstance(payload, dict):
        return None
    # Alpha Vantage signale clé invalide / quota via ces champs (pas de "transcript").
    for flag in ("Information", "Note", "Error Message"):
        if payload.get(flag):
            print(f"  [API] {payload[flag]}")
            return None
    turns = payload.get("transcript")
    if not isinstance(turns, list) or not turns:
        return None
    lines: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        speaker = str(turn.get("speaker") or "").strip()
        title = str(turn.get("title") or "").strip()
        header = speaker + (f" ({title})" if title else "") if speaker else ""
        lines.append(f"{header}: {content}" if header else content)
    text = "\n\n".join(lines).strip()
    if len(text) < _MIN_TRANSCRIPT_CHARS:
        return None
    return text


def save_transcript(ticker: str, year: int, quarter: int, content: str) -> str | None:
    ensure_dir(DATA_DIR)
    filename = DATA_DIR / f"{ticker.lower()}-earnings_call_{year}q{quarter}.txt"
    portable_path = filename.relative_to(PROJECT_ROOT).as_posix()
    if filename.exists():
        return portable_path
    filename.write_text(content, encoding="utf-8")
    return portable_path


def parse_args() -> argparse.Namespace:
    current_year = datetime.now().year
    parser = argparse.ArgumentParser(
        description="Télécharger les transcripts d'earnings calls de l'univers suivi."
    )
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


def main() -> int:
    args = parse_args()
    api_key = _api_key()
    if not api_key:
        print(
            "⏭  ALPHAVANTAGE_API_KEY absente — étape earnings calls ignorée "
            "(la pipeline reste fonctionnelle sans transcripts)."
        )
        return 0

    print(
        f"⏱  Pacing earnings : {_INTER_CALL_SLEEP}s entre appels "
        f"(EARNINGS_INTER_CALL_SLEEP={os.getenv('EARNINGS_INTER_CALL_SLEEP', 'default')})"
    )
    print(
        f"📋 Earnings calls : {len(TRACKED_TICKERS)} tickers, "
        f"{args.min_year}-{args.max_year}\n"
    )

    results: dict[str, list[dict]] = {}
    total = 0
    for ticker in TRACKED_TICKERS:
        per_ticker: list[dict] = []
        for year in range(args.min_year, args.max_year + 1):
            for quarter in range(1, 5):
                content = fetch_transcript(ticker, year, quarter, api_key)
                if not content:
                    continue
                local_path = save_transcript(ticker, year, quarter, content)
                if local_path:
                    per_ticker.append(
                        {
                            "ticker": ticker,
                            "year": year,
                            "quarter": quarter,
                            "localPath": local_path,
                        }
                    )
                    total += 1
        results[ticker] = per_ticker
        print(f"-> {len(per_ticker)} transcript(s) pour {ticker}.")

    EARNINGS_CALLS_METADATA.write_text(
        json.dumps(results, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[Terminé] {total} transcript(s). Métadonnées : {EARNINGS_CALLS_METADATA}")
    print(f"Fichiers : {DATA_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
