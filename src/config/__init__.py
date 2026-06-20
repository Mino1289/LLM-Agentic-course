"""Configuration globale du projet."""

from __future__ import annotations

# Univers suivi pour l'indexation RAG.
# Focus semi-conducteurs / IA. Issuers US -> 10-K/10-Q/8-K ; foreign issuers
# (ASML, ARM, TSM) -> 20-F/6-K. Le downloader SEC détecte la forme via EDGAR,
# aucun mapping par ticker n'est nécessaire ici.
TRACKED_TICKERS = (
    "NVDA",  # Nvidia (US, SEC)
    "ASML",  # ASML Holding (NL, foreign issuer, SEC)
    "AMD",  # AMD (US, SEC)
    "ARM",  # Arm Holdings (UK, foreign issuer, SEC)
    "MSFT",  # Microsoft (US, SEC)
    "TSM",  # TSMC (Taïwan, foreign issuer, SEC -> 20-F/6-K)
    "AVGO",  # Broadcom (US, SEC)
    "INTC",  # Intel (US, SEC)
    "QCOM",  # Qualcomm (US, SEC)
    "MU",  # Micron Technology (US, SEC)
)
