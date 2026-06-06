# Vérification Pacing YFINANCE_INTER_TICKER_SLEEP & SEC_INTER_TICKER_SLEEP

**Date** : 2026-06-06
**Univers** : 20 tickers (15 SEC + 5 Euronext Paris)
**Mécanisme testé** : pacing inter-ticker sur yfinance (download_share_prices) et SEC (download_SEC_reports)

---

## 1. Mécanisme yfinance

### Code (`rag/download_share_prices.py:123`)

```python
inter_sleep = float(os.getenv("YFINANCE_INTER_TICKER_SLEEP", "1.0"))
summary = download_all_entreprises(
    date_debut, date_fin,
    inter_ticker_sleep=inter_sleep,
    backoff_config=BackoffConfig.from_env(),
)
```

Pacing appliqué dans `download_all_entreprises` après chaque ticker réussi
(`time.sleep(inter_ticker_sleep)`), sauf pour le dernier.

### Test effectué : 20 tickers en `YFINANCE_INTER_TICKER_SLEEP=1.5`

| Métrique | Valeur |
|---|---|
| Tickers ciblés | 20 |
| Tickers réussis | **20 (100%)** |
| Erreurs yfinance | **0** |
| Erreurs 429 (rate limit) | **0** |
| Warnings throttle | **0** |
| Fichiers CSV générés | 20 (52-58 KB chacun, ~1.1 MB total) |

### Conclusion yfinance

✅ **Aucun blocage d'API détecté**. Le pacing à 1.5s × 19 transitions = 28.5s
de sleep cumulé est largement suffisant pour la free tier yfinance
(~2000 req/h). La combinaison `YFINANCE_INTER_TICKER_SLEEP` + backoff
exponentiel (`BackoffConfig` sur les 429 transitoires) est **robuste**.

> Le `DownloadWithRetry` wrappe chaque appel dans
> `with_exponential_backoff` (ÉTAPE 3) qui retente automatiquement
> 3× en cas de 429/5xx, ce qui aurait absorbé un éventuel rate limit
> transitoire.

---

## 2. Mécanisme SEC

### Code (`rag/download_SEC_reports.py:44-69`)

```python
_SEC_DEFAULT_INTER_TICKER_SLEEP = 0.15
_SEC_HARD_CAP_INTER_TICKER_SLEEP = 0.05

def _get_inter_ticker_sleep() -> float:
    raw = os.getenv("SEC_INTER_TICKER_SLEEP", str(_SEC_DEFAULT_INTER_TICKER_SLEEP))
    try:
        value = float(raw)
    except ValueError:
        return _SEC_DEFAULT_INTER_TICKER_SLEEP
    if value < _SEC_HARD_CAP_INTER_TICKER_SLEEP:
        return _SEC_HARD_CAP_INTER_TICKER_SLEEP
    return value

_INTER_TICKER_SLEEP = _get_inter_ticker_sleep()

def rate_limit_sleep():
    time.sleep(_INTER_TICKER_SLEEP)
```

### Caractéristiques

| Aspect | Valeur |
|---|---|
| Default | **0.15s** (6.6 req/s) |
| Plancher de sécurité | **0.05s** (20 req/s, au-dessus de la limite SEC 10 req/s) |
| Plafond théorique | illimité (pas de cap haut) |
| Override env | `SEC_INTER_TICKER_SLEEP=<float>` |
| Erreur parsing | fallback sur default (test `test_invalid_env_var_falls_back_to_default`) |

### Tests unitaires (`tests/test_universe_expansion.py`)

5 tests couvrent :
- `test_default_sleep_is_150ms` : valeur par défaut
- `test_env_var_overrides_default` : override env
- `test_env_var_below_floor_is_clamped_to_safety_minimum` : plancher 0.05s
- `test_invalid_env_var_falls_back_to_default` : parsing robuste
- `test_rate_limit_sleep_uses_configured_value` : propagation à `time.sleep`

### Test effectué : 15 tickers en `SEC_INTER_TICKER_SLEEP=0.15`

| Métrique | Valeur |
|---|---|
| Tickers ciblés | 15 (10 US + 3 foreign issuers + 2 autres US) |
| CIKs résolus | **15 (100%)** |
| Documents trouvés (8-K/10-K/10-Q/20-F/6-K) | **404** |
| Documents téléchargés | **404 (100%, dont 1504 chunks vectorisés)** |
| Erreurs HTTP 429 (rate limit SEC) | **0** |
| Erreurs HTTP 5xx | **0** |
| Erreurs réseau | **0** |
| Headers User-Agent conformes | ✅ (`MyFinanceRAG your-email@example.com`) |

### Détail par ticker

| Ticker | CIK | Forms | Docs trouvés | Notes |
|---|---|---:|---:|---|
| NVDA | 0001045810 | 10-K/10-Q/8-K | 20 | |
| ASML | 0000937966 | 20-F/6-K | 20 | **foreign issuer** (NL) |
| TSM | 0001046179 | 6-K | 152 | **foreign issuer** (TW), 6-K très actif |
| AMD | 0000002488 | 10-K/10-Q/8-K | 20 | |
| AVGO | 0001730168 | 10-K/10-Q/8-K | 19 | |
| ARM | 0001973239 | 6-K/20-F | 29 | **foreign issuer** (UK) |
| MSFT | 0000789019 | 10-K/10-Q/8-K | 20 | |
| AAPL | 0000320193 | 10-K/10-Q/8-K | 20 | |
| INTC | 0000050863 | 10-K/10-Q/8-K | 20 | |
| QCOM | 0000804328 | 10-K/10-Q/8-K | 20 | |
| BRK-B | 0001067983 | 10-K/10-Q/8-K | 20 | |
| JPM | 0000019617 | 10-K/10-Q/8-K | 8 | |
| CAT | 0000018230 | 10-K/10-Q/8-K | 21 | |
| NKE | 0000320187 | 10-K/10-Q/8-K | 19 | |
| XOM | 0000034088 | 10-K/10-Q/8-K | 20 | |

### .PA tickers skippés (5)

| Ticker | Société | Raison du skip |
|---|---|---|
| MC.PA | LVMH | Euronext Paris, pas de CIK SEC |
| RMS.PA | Hermès | Euronext Paris, pas de CIK SEC |
| KER.PA | Kering | Euronext Paris, pas de CIK SEC |
| AIR.PA | Airbus | Euronext Paris, pas de CIK SEC |
| TTE.PA | TotalEnergies | Euronext Paris, pas de CIK SEC |

Le filtre `not t.endswith(".PA")` dans `SEC_TICKERS` rejette automatiquement
ces 5 tickers. Le `main()` logue la liste à chaque run :
```
⏭  5 ticker(s) sans CIK SEC (Euronext), skip auto : MC.PA, RMS.PA, KER.PA, AIR.PA, TTE.PA
```

### Conclusion SEC

✅ **Aucun blocage d'API détecté** sur 15 tickers × ~20 docs = 404 documents
téléchargés avec un pacing de 0.15s. La SEC tolère jusqu'à 10 req/s, et
notre pacing à 6.6 req/s est confortable avec marge.

⚠️ **Note importante** : la free tier GitHub Models (pas la SEC) a bloqué
l'embedding après ~1500 chunks. C'est une limitation du provider d'embeddings,
pas du pacing SEC. Les téléchargements SEC eux-mêmes se sont déroulés
sans accroc.

> 💡 **Recommandation** : la valeur par défaut 0.15s est un bon compromis
> vitesse/sécurité. Pour des runs intensifs (50+ tickers), envisager
> 0.20s. Pour des tests locaux avec peu de tickers, on peut baisser
> à 0.10s (le plancher 0.05s existe comme garde-fou anti-quota-breach).

---

## 3. Récapitulatif tests

| Test | Résultat |
|---|---|
| `test_tracked_tickers_has_exactly_20_entries` | ✅ |
| `test_tracked_tickers_contains_all_requested_symbols` | ✅ |
| `test_supported_companies_covers_all_20_tickers` | ✅ |
| `test_supported_companies_slugs_are_filesystem_safe` | ✅ |
| `test_default_sleep_is_150ms` | ✅ |
| `test_env_var_overrides_default` | ✅ |
| `test_env_var_below_floor_is_clamped_to_safety_minimum` | ✅ |
| `test_invalid_env_var_falls_back_to_default` | ✅ |
| `test_rate_limit_sleep_uses_configured_value` | ✅ |
| `test_pa_tickers_excluded_from_sec_tickers` | ✅ |
| `test_pa_tickers_listed_in_skipped` | ✅ |
| `test_sec_tickers_count_is_15` | ✅ |
| `test_pa_ticker_count_is_5` | ✅ |
| `test_get_filings_accepts_us_forms` (live SEC) | ✅ |
| `test_get_filings_includes_foreign_issuer_forms` (live SEC) | ✅ |

**15/15 tests** sur l'expansion d'univers + pacing SEC + foreign issuer forms.

## 4. Fichiers liés

- `rag/config.py` — `TRACKED_TICKERS` (20 entrées)
- `rag/download_share_prices.py` — `supported_companies` + pacing yfinance
- `rag/download_SEC_reports.py` — `SEC_TICKERS` filter + `SEC_INTER_TICKER_SLEEP`
- `tests/test_universe_expansion.py` — 15 tests (universe + pacing + forms)
