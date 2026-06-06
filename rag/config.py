from __future__ import annotations


# Univers suivi pour l'indexation RAG.
# 20 entreprises sélectionnées pour l'audit TOON volume réel.
# 14 ont des rapports SEC (US/EU/foreign issuer), 6 sont Euronext Paris (yfinance prix seuls).
TRACKED_TICKERS = (
    "NVDA",    # Nvidia (US, SEC)
    "ASML",    # ASML Holding (NL, foreign issuer, SEC)
    "TSM",     # Taiwan Semi (TW, ADR, SEC)
    "AMD",     # AMD (US, SEC)
    "AVGO",    # Broadcom (US, SEC)
    "ARM",     # Arm Holdings (UK, foreign issuer, SEC)
    "MSFT",    # Microsoft (US, SEC)
    "AAPL",    # Apple (US, SEC)
    "INTC",    # Intel (US, SEC)
    "QCOM",    # Qualcomm (US, SEC)
    "MC.PA",   # LVMH (FR, Euronext)
    "RMS.PA",  # Hermès (FR, Euronext)
    "KER.PA",  # Kering (FR, Euronext)
    "AIR.PA",  # Airbus (FR, Euronext)
    "TTE.PA",  # TotalEnergies (FR, Euronext)
    "BRK-B",   # Berkshire Hathaway B (US, SEC)
    "JPM",     # JPMorgan Chase (US, SEC)
    "CAT",     # Caterpillar (US, SEC)
    "NKE",     # Nike (US, SEC)
    "XOM",     # ExxonMobil (US, SEC)
)
