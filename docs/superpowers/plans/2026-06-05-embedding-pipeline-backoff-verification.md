# Étape 3 — Verification Report

**Date** : 2026-06-05
**Branche** : `etape2`
**Plan** : `docs/superpowers/plans/2026-06-05-embedding-pipeline-backoff.md`

## Suite de tests

| Suite | Tests | Statut |
|-------|-------|--------|
| `BackoffRetryTests` | 5 | ✅ OK |
| `QuotaStateTests` | 3 | ✅ OK |
| `YFinanceRetryTests` | 2 | ✅ OK |
| `BackoffConfigEnvTests` | 2 | ✅ OK |
| Régressions existantes | 34 | ✅ OK |
| **Total** | **46** | **✅ 46/46** |

## Smoke tests

| Test | Résultat |
|------|----------|
| `from rag.embedding_pipeline import QuotaState, BackoffConfig, with_exponential_backoff, is_permanent_error` | ✅ |
| `from rag.hybrid_rag import HybridRAG, EmbeddingPlan, BackoffConfig` | ✅ |
| `from rag.download_share_prices import download_with_retry, download_all_entreprises` | ✅ |
| `python -m py_compile rag/{hybrid_rag,download_share_prices,embedding_pipeline}.py` | ✅ |
| `python -m py_compile ui/app_rag.py` | ✅ |
| `python rag/hybrid_rag.py --help` (CLI args visibles : `--quota-state`) | ✅ |
| `python rag/hybrid_rag.py --plan` (auto-reprise quota) → "♻️ Reprise quota depuis l'état persisté : 0" | ✅ |
| `import rag.download_share_prices` (side-effect free) | ✅ |
| `QuotaState` round-trip (write → new instance → read) | ✅ |
| `is_permanent_error` (`dim mismatch` → True ; `ConnectionError` → False) | ✅ |

## Couverture PRD §2.3

| Exigence | Statut | Implémentation |
|----------|--------|----------------|
| Backoff exponentiel natif | ✅ | `with_exponential_backoff` + `_compute_sleep` (full jitter) |
| Suivi granulaire `quota-used` à chaud | ✅ | `QuotaState.update()` après chaque batch |
| Reprise entre runs | ✅ | `data/embedding_quota_state.json` lu si `--quota-used` omis |
| « chacun dans son module » | ✅ | `rag/embedding_pipeline.py` (helpers), `hybrid_rag.py` (consommateur), `download_share_prices.py` (consommateur) |
| Tests pour les deux chemins (§5.2) | ✅ | `YFinanceRetryTests` + intégration `BackoffRetryTests` dans `hybrid_rag` |

## Critères additionnels

- ✅ Fail-fast sur erreurs permanentes (auth, dim mismatch) — `is_permanent_error`
- ✅ Cap relevé de 8s à 60s (configurable via `EMBEDDING_BACKOFF_CAP_SEC`)
- ✅ Jitter full-jitter (AWS best practice)
- ✅ Cap `time.sleep` (lazy lookup) → tests mockables
- ✅ `--quota-state` CLI arg + `EMBEDDING_QUOTA_STATE_PATH` env var
- ✅ Message « Reprise quota depuis l'état persisté » au démarrage
- ✅ `last_error` sticky dans le state JSON (debug-friendly)
- ✅ `download_all_entreprises` retourne un résumé `{ticker: status}`
- ✅ Pacing inter-tickers (1s par défaut, `YFINANCE_INTER_TICKER_SLEEP` configurable)

## Commits

```
afcb16a feat(etape3): yfinance retry + polite pacing
099ec7d test(etape3): fix jitter test expected sleep count (5, not 6)
072e96c refactor(etape3): wire backoff + granular quota tracking in hybrid_rag
80c0223 feat(etape3): exponential backoff + quota state helpers
7e1eef8 test(etape3): red tests for backoff + quota tracking
```

5 commits + 1 correctif de test (5 → 6 au total, conforme au plan + 1 ajustement TDD).

## Hors scope (volontairement)

- ❌ Reprise des chunks partiels au-delà du compteur journalier (chunk-level checkpointing)
- ❌ SQLite pour state (over-engineering vs. JSON)
- ❌ Télémétrie Prometheus
- ❌ Tests d'intégration end-to-end (nécessitent clé API + ChromaDB peuplé)
