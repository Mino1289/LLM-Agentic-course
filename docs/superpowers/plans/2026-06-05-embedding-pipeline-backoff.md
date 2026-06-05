# Étape 3 — Backoff exponentiel ingestion (PRD §2.3)

**Branche** : `etape2`
**Date** : 2026-06-05
**PRD** : `PRD.md` §2.3, §5.2

## Objectif

1. **Backoff exponentiel natif** sur les appels API (ChromaDB embeddings + yfinance).
2. **Suivi granulaire de `quota-used` à chaud** avec persistance JSON pour reprise entre runs.
3. Fail-fast sur les erreurs permanentes (auth, dim mismatch).

## Décisions validées (plan mode)

| # | Décision | Implémentation |
|---|----------|----------------|
| 1 | Classification erreurs | Blacklist explicite dans `is_permanent_error` (tokens type + message) |
| 2 | Jitter | Full jitter (`random.uniform(0, min(cap, base * 2**attempt))`) |
| 3 | Format state file | JSON enrichi `{date, quota_used, last_batch_size, last_error, last_updated}` dans `data/embedding_quota_state.json` |
| 4 | Scope | Complet : `hybrid_rag.py` + `download_share_prices.py` |

## Chaîne de commits (5 commits, TDD strict)

| # | Type | Fichier(s) | Commit |
|---|------|------------|--------|
| **E1** | 🔴 red | `tests/test_embedding_backoff.py` | `test(etape3): red tests for backoff + quota tracking` |
| **E2** | 🟢 impl | `rag/embedding_pipeline.py` | `feat(etape3): exponential backoff + quota state helpers` |
| **E3** | 🟢 impl | `rag/hybrid_rag.py` | `refactor(etape3): wire backoff + granular quota tracking in hybrid_rag` |
| **E4** | 🟢 impl | `rag/download_share_prices.py` | `feat(etape3): yfinance retry + polite pacing` |
| **E5** | 🔵 verify | rapport | `chore(etape3): verification report` |

## Tests (12 nouveaux)

### `BackoffRetryTests` (5)
- `test_succeeds_after_transient_retries` : mock fail 2× puis success → 3 appels, 2 sleeps bornés
- `test_gives_up_after_max_retries` : toujours fail → raise après `max_retries+1` appels
- `test_does_not_retry_on_dim_mismatch` : `ValueError("dim mismatch")` → 1 appel, fail-fast
- `test_does_not_retry_on_auth_error` : type `Authentication` → 1 appel
- `test_full_jitter_sleeps_in_bounds` : 6 attempts → 5 sleeps ∈ `[0, min(cap, base*2^attempt)]`

### `QuotaStateTests` (3)
- `test_persists_across_instances` : write → new instance → read cohérent
- `test_update_increments_and_records_error` : incréments + last_error sticky
- `test_load_returns_defaults_when_file_missing` : payload `{0, None, "", today}`

### `YFinanceRetryTests` (2)
- `test_download_with_retry_recovers_from_transient_error` : mock fail 1× → success
- `test_download_with_retry_gives_up_after_max` : mock always fail → raise

### `BackoffConfigEnvTests` (2)
- `test_from_env_reads_overrides` : env vars lus
- `test_from_env_falls_back_to_defaults` : 1.0/60.0/"full"/3

## Variables d'environnement (E3)

| Var | Default | Usage |
|-----|---------|-------|
| `EMBEDDING_BACKOFF_BASE_SEC` | 1.0 | Base du backoff exponentiel |
| `EMBEDDING_BACKOFF_CAP_SEC` | 60.0 | Cap (sleep max) |
| `EMBEDDING_BACKOFF_JITTER` | full | full / equal / none |
| `EMBEDDING_BACKOFF_MAX_RETRIES` | 3 | Tentatives max |
| `EMBEDDING_QUOTA_STATE_PATH` | `data/embedding_quota_state.json` | Fichier d'état |

## Critères PRD satisfaits

- ✅ §2.3 « backoff exponentiel natif » : helper `_compute_sleep(config, attempt, rng)` avec full jitter
- ✅ §2.3 « suivi granulaire de `quota-used` à chaud » : `QuotaState.update()` après chaque batch, JSON persisté
- ✅ §2.3 « reprise entre runs » : `data/embedding_quota_state.json` lu au démarrage si `--quota-used` non fourni
- ✅ §2.3 « chacun dans son module » : `rag/embedding_pipeline.py` (helpers), `rag/hybrid_rag.py` (utilisateur), `rag/download_share_prices.py` (utilisateur)
- ✅ §5.2 tests `pytest`/`unittest` pour les deux chemins

## Critères NON couverts (volontairement)

- ❌ SQLite pour quota state (over-engineering, PRD pas explicite)
- ❌ Reprise automatique des chunks partiels (au-delà du compteur journalier) — laissé à un futur run
- ❌ Télémétrie Prometheus — hors scope
