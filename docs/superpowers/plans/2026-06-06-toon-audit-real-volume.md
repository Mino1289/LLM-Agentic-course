# Audit TOON — Option 1 (P1+P2+P3) sur volume réel (20 tickers)

**Date** : 2026-06-06
**Univers ciblé** : 20 tickers (15 SEC + 5 Euronext)
**Vectorstore réel** : 4,458 chunks indexés sur 6/15 tickers SEC (cause : rate limit GitHub Models)
**Outil de mesure** : `tiktoken` (cl100k_base, le tokenizer d'OpenAI GPT-4/3.5)

---

## 1. Contexte

Suite à l'audit initial du 2026-06-06 (`docs/superpowers/plans/2026-06-06-toon-integration-verification.md`),
ce second audit mesure les gains TOON sur le **volume réel** généré par l'indexation des 20 tickers
souhaités, et non plus sur des données synthétiques. Le script d'audit est
`scripts/audit_toon_real_volume.py` et produit un JSON `2026-06-06-toon-audit-real-volume-data.json`.

### Univers 20 tickers

| Catégorie | Tickers | SEC | Pacing |
|---|---|---|---|
| US (10) | NVDA, AMD, MSFT, AAPL, INTC, QCOM, BRK-B, JPM, CAT, NKE, XOM | ✅ | 0.15s |
| Foreign issuers (3) | ASML (NL), TSM (TW), ARM (UK) | ✅ (20-F/6-K) | 0.15s |
| Euronext Paris (5) | MC.PA, RMS.PA, KER.PA, AIR.PA, TTE.PA | ❌ (yfinance prix seuls) | 1.5s |

### État de l'indexation

| Métrique | Valeur |
|---|---|
| Documents SEC téléchargés | 404 (8-K, 10-K, 10-Q, 20-F, 6-K) |
| Sections preprocessées | 322 (Item 1A/7) |
| Chunks totaux disponibles | 12,657 |
| Chunks effectivement indexés | **4,458** sur **6 tickers** (NVDA, AMD, MSFT, ARM, AAPL, AVGO) |
| Cause indexation partielle | **GitHub Models rate limit 150/jour** sur `text-embedding-3-small` |

> ⚠️ **Limitation importante** : la free tier GitHub Models impose 150 requêtes/jour/modèle.
> L'embedding a été interrompu à 1,504 chunks (batch 48) avec un message
> "Rate limit of 150 per 86400s exceeded". Pour indexer les 9,703 chunks restants,
> il faut soit attendre 9h30, soit basculer sur OpenAI direct (OPENAI_API_KEY vide
> dans `.env`), soit utiliser un autre provider (Gemini embeddings).

---

## 2. Résultats P1-A : `format_rag_excerpts` (5 chunks/ticker)

Cette section mesure le gain réel sur 6 tickers (les seuls indexés).

| Ticker | Legacy (tok) | TOON (tok) | Économie | % |
|---|---:|---:|---:|---:|
| NVDA | 1,349 | 1,236 | +113 | **+8.4%** |
| AMD | 999 | 990 | +9 | **+0.9%** |
| MSFT | 1,469 | 1,299 | +170 | **+11.6%** |
| ARM | 1,584 | 1,338 | +246 | **+15.5%** |
| AAPL | 1,341 | 1,214 | +127 | **+9.5%** |
| AVGO | 1,154 | 1,098 | +56 | **+4.9%** |
| **Total P1-A** | **7,896** | **7,175** | **+721** | **+9.1%** |

### Interprétation

L'audit initial mesurait **+37.2%** sur P1-A mais comparait TOON à un **JSON canonique**.
Ici on compare TOON au **format legacy texte** (`"[1] TICKER/YEAR/FORM | section=X | source=Y\n    text..."`)
qui était déjà plus compact que JSON.

Le gain TOON vs legacy texte est **modeste (+9.1%)** mais le gain TOON vs JSON
pur serait de l'ordre de **+25-35%** (cf. audit initial). Le format legacy
avait déjà fait un travail d'optimisation manual, donc le gain marginal est
plus faible.

> 💡 **Action** : P1-A reste pertinent pour standardiser le format et faciliter
> l'évolution, mais le ROI marginal est faible tant que le format legacy est
> préservé en commentaire pour rétrocompat.

---

## 3. Résultats P1-B : `tool schema` JSON → TOON (**BLOQUÉ**)

| Format | Tokens | Économie |
|---|---:|---:|
| JSON tool schema | 191 | — |
| TOON tool schema | 119 | **+37.7%** |

### Statut : ❌ BLOQUÉ par l'API OpenAI

L'API OpenAI Chat Completions exige que le champ `tools` soit un **tableau JSON** :
```json
[{"type": "function", "function": {"name": "...", "parameters": {...}}}]
```

Le schéma TOON ne respecte pas ce contrat. Pour activer P1-B il faudrait :
1. Valider en amont avec un parseur custom que la conversion JSON↔TOON est bijective
2. Soumettre une RFC à OpenAI pour accepter des formats tabulaires
3. **Ou** réduire la verbosité des descriptions/schémas en JSON (par exemple en
   éliminant les champs `description` redondants avec le `name`)

**Recommandation** : pas de mise en œuvre. Le coût est disproportionné par
rapport au gain (5 outils × ~50 tok = 250 tok max par requête).

---

## 4. Résultats P2-C : `format_memory_context` + `format_chat_context`

| Contexte | Legacy (tok) | TOON (tok) | Économie | % |
|---|---:|---:|---:|---:|
| `memory_context` (10 tours) | 331 | 195 | +136 | **+41.1%** |
| `chat_context` (6 tours) | 164 | 85 | +79 | **+48.2%** |
| **Total P2-C** | **495** | **280** | **+215** | **+43.4%** |

### Interprétation

P2-C confirme l'audit initial (**+29.6%** mesuré alors) et le dépasse légèrement.
Le format TOON `turns[N]{role,content}:` est très efficace pour les tableaux
homogènes (rôle + contenu de même structure). Le gain est **composé** : sur 10
tours de conversation on sauve ~136 tokens, ce qui s'additionne à chaque tour.

> 💡 **Action** : P2-C est l'optimisation TOON la plus rentable. Maintenue et
> intégrée (`format_memory_context`/`format_chat_context` dans
> `rag/nodes/memory_store.py`).

---

## 5. Résultats P3 : autres serializers

### P3 : `_build_nli_prompt` (mixed format)

| Format | Tokens | Économie |
|---|---:|---:|
| JSON NLI prompt (claim + 2 chunks + metadata + instruction) | 119 | — |
| TOON NLI prompt | 96 | **+19.3%** |

### Interprétation

Le gain est partiel (+19%) car le prompt NLI est **mixte** : du texte libre
(claim, instruction) cohabite avec des données structurées (chunks, metadata).
TOON optimise la partie structurée mais le texte libre est neutre.

**Recommandation** : migration **différée**. Le gain marginal ne justifie pas
le risque de régression sur la compréhension LLM. À réévaluer si le LLM
montre des signes d'incompréhension du format TOON en production.

### P3bis (non mesuré) : `format_universe_hint`, `tool_events` UI

- `format_universe_hint` : gain marginal (~5%), format déjà compact
- `tool_events` UI : gain marginal, format custom déjà optimisé pour l'affichage

---

## 6. Synthèse globale

| Catégorie | Legacy (tok) | TOON (tok) | Économie | % |
|---|---:|---:|---:|---:|
| P1-A (RAG excerpts) | 7,896 | 7,175 | +721 | +9.1% |
| P1-B (tool schema) | — | — | — | BLOQUÉ |
| P2-C (memory/chat) | 495 | 280 | +215 | +43.4% |
| P3 (NLI prompt) | 119 | 96 | +23 | +19.3% |
| **Total mesuré** | **8,510** | **7,551** | **+959** | **+11.3%** |

### Projection sur 20 tickers

Si l'on extrapole linéairement le gain P1-A sur les 15 tickers SEC attendus
(au lieu des 6 actuellement indexés) :

| Catégorie | 6 tickers (mesuré) | 15 tickers (estimé) | 20 tickers (incl. .PA) |
|---|---:|---:|---:|
| P1-A RAG excerpts (×5 chunks × 1300 tok) | 7,896 | 19,740 | 19,740 (.PA = yfinance) |
| Économie P1-A (+9.1%) | +721 | **+1,802** | +1,802 |
| Économie P2-C (constante) | +215 | +215 | +215 |
| **Économie totale / requête type** | **+936** | **+2,017** | **+2,017** |

> 💡 Pour les **6 tickers indexés**, chaque requête type RAG multi-tours
> économise **~936 tokens**. Si l'indexation était complète (15 tickers SEC),
> l'économie passerait à **~2,017 tokens/requête**.

---

## 7. Recommandations

### Court terme (déjà fait)
- ✅ **P2-C intégré** : `format_memory_context` + `format_chat_context` retournent
  du TOON. Gain composé : ~215 tok économisés par conversation multi-tours.
- ✅ **P1-A intégré** : `format_rag_excerpts` retourne du TOON. Gain modeste vs
  legacy texte (+9%), mais standardise le format.

### Moyen terme
- ⏸ **P1-B** : rester bloqué, ROI disproportionné. Si critique, réduire la
  verbosité des descriptions dans `rag/tools.py` directement.
- ⏸ **P3 NLI** : différer. Réévaluer après quelques semaines de production P2-C.

### Long terme
- 🔄 **Réindexer les 14 tickers manquants** (ASML, TSM, AMD, INTC, QCOM, BRK-B,
  JPM, CAT, NKE, XOM) une fois le rate limit GitHub Models reset (~9h30)
  ou via un autre provider embeddings.
- 📊 **Mesure coût OpenAI réelle** sur 50 requêtes types avant/après TOON,
  pour valider l'économie USD.

---

## 8. Commandes utiles

```bash
# Relancer l'audit à tout moment
.venv/bin/python -m scripts.audit_toon_real_volume

# Voir le JSON brut
cat docs/superpowers/plans/2026-06-06-toon-audit-real-volume-data.json
```

## 9. Fichiers liés

- `scripts/audit_toon_real_volume.py` — script d'audit reproductible
- `docs/superpowers/plans/2026-06-06-toon-audit-real-volume-data.json` — résultats JSON
- `docs/superpowers/plans/2026-06-06-toon-integration-verification.md` — audit initial (synthétique)
- `docs/superpowers/plans/2026-06-06-llm-streaming-ui-verification.md` — vérification ÉTAPE 6
- `tests/test_toon_integration.py` — 9 tests d'intégration TOON
- `tests/test_toon_serialization.py` — 10 tests de sérialisation TOON
