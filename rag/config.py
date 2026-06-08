from __future__ import annotations


# Univers suivi pour l'indexation RAG.
# Mode test/debug: 5 entreprises avec rapports SEC / foreign issuer.
TRACKED_TICKERS = (
    "NVDA",    # Nvidia (US, SEC)
    "ASML",    # ASML Holding (NL, foreign issuer, SEC)
    "AMD",     # AMD (US, SEC)
    "ARM",     # Arm Holdings (UK, foreign issuer, SEC)
    "MSFT",    # Microsoft (US, SEC)
)
