# TOON Integration — Verification Report

**Date** : 2026-06-06
**Branche** : `etape2`
**Commit** : `65575ed feat(deps): add toon_format[pydantic] + tiktoken, 9 integration tests green`
**Statut** : ✅ Intégration minimale réussie, optimisations **non implémentées** (audit only)

---

## 1. Critères de succès

| # | Critère | Statut |
|---|---------|--------|
| 1 | `toon_format[pydantic]` ajouté à `requirements.txt` | ✅ |
| 2 | `tiktoken` ajouté à `requirements.txt` | ✅ |
| 3 | 9 tests d'intégration RED → GREEN | ✅ |
| 4 | Audit P1+P2+P3 livré | ✅ (ce document) |
| 5 | Aucun fichier `rag/` modifié (intégration minimale) | ✅ |
| 6 | Full test suite 114/114 verts | ✅ |
| 7 | Rapport de vérif écrit | ✅ |

---

## 2. Résultats tests

### Suite complète

| Catégorie | Tests | Résultat |
|-----------|-------|----------|
| `test_toon_integration.py` (nouveau) | 9 | ✅ verts |
| Tests précédents (étape 1-6) | 105 | ✅ verts |
| **TOTAL** | **114** | **✅ 114/114** |

### Détail des 9 tests TOON

| Test | Vérifie |
|------|---------|
| `test_import_toon_format_succeeds` | `from toon_format import encode, decode` |
| `test_import_token_helpers_succeeds` | `estimate_savings`, `count_tokens` |
| `test_import_pydantic_extra_succeeds` | `from toon_format.pydantic import ToonPydanticModel` |
| `test_encode_simple_object_produces_toon_string` | `encode(dict)` → str non-vide |
| `test_decode_roundtrip_simple_dict` | roundtrip plat |
| `test_decode_roundtrip_nested_dict` | roundtrip imbriqué |
| `test_encode_tabular_array_uses_compact_format` | forme `]{fields}:` détectée |
| `test_decode_tabular_array_returns_list` | décodage inverse |
| `test_estimate_savings_returns_positive_for_tabular_data` | sanity check (mesuré 56.7% sur 4 lignes RAG) |

---

## 3. Versions installées

| Package | Version | Source |
|---------|---------|--------|
| `toon-format` | 0.9.0b1 | `git+https://github.com/toon-format/toon-python.git@e475c82e` |
| `tiktoken` | 0.13.0 | PyPI |
| `pydantic` | 2.13.4 | déjà installé (via `[pydantic]` extra) |

**Notes** :
- Le repo PyPI package name = `toon_format` (underscore, conforme PEP 503). Le `name` du `pyproject.toml` du repo est `toon_format`.
- Le `name` du repo GitHub = `toon-python` (tiret). L'URL d'install est donc `toon_format[pydantic] @ git+https://github.com/toon-format/toon-python.git` — mélange valide.
- Status upstream : **v0.9.x beta** (API peut changer avant 1.0).

---

## 4. Audit des sites d'optimisation (P1+P2+P3)

### P1-A : `format_rag_excerpts` (`rag/tools.py:115`)

**Structure actuelle** : liste de chunks, chacun formaté en texte brut avec 5 champs metadata inline.

```python
[1] ticker=NVDA year=2024 file_type=10-K section=Item_1A source=nvda-10-k_2024-01-01.htm
We are subject to risks related to global supply chain...

---

[2] ticker=NVDA year=2024 ...
```

**Forme TOON candidate** (sortie de `encode(rag_excerpts)`) :

```
excerpts[5]{i,ticker,year,file_type,section,source,text}:
  1,NVDA,"2024",10-K,Item_1A,nvda-10-k_2024-01-01.htm,"We are subject to risks related to..."
  2,NVDA,"2024",10-K,Item_1A,nvda-10-k_2024-01-01.htm,"Our data center business depends..."
```

**Mesure (tiktoken + count_tokens)** :
- JSON tokens (équivalent) : **486**
- TOON tokens : **305**
- **Économie : 37.2%** (−181 tokens par appel RAG)

**Multiplicateur d'usage** : 1× par appel `sec_filings_rag_tool`. Sur une requête complexe "compare MSFT/NVDA risques + perf 6 mois + simulate + export" → 1 à 3 appels RAG → **−180 à −540 tokens par requête**.

**Compatibilité LLM** : ⚠️ **À valider**. Le format TOON est conçu pour les LLM (testé upstream sur GPT-5 et autres), mais les 5 champs metadata + chunk deviennent tabulaires. Le LLM doit comprendre la structure `[N]{i,ticker,year,...}:`. Risque : 1ère itération nécessaire pour calibrer le prompt system ("`excerpts` is a TOON-formatted array; columns are i, ticker, year, file_type, section, source, text").

**Recommandation** : ⭐⭐⭐ **P1** — gain mesuré, code d'intégration trivial (~10 lignes), risque modéré (validation prompt).

---

### P1-B : Schémas outils LLM (`rag/tools.py:487`)

**Structure actuelle** : `get_tool_definitions()` retourne 5 dicts JSON-shaped conformes OpenAI function-calling. Envoyés **à chaque appel LLM** dans le payload system/developer.

**Forme TOON candidate** (via `ToonPydanticModel.schema_to_toon()`) :

```python
from toon_format.pydantic import ToonPydanticModel
from rag.tool_schemas import SecFilingsRAGArgs, MarketPriceArgs, ...

class SecFilingsRAGToolDef(ToonPydanticModel):
    name: str = "sec_filings_rag_tool"
    description: str = SEC_FILINGS_RAG_DESCRIPTION
    parameters: SecFilingsRAGArgs

toon_schema = SecFilingsRAGToolDef.schema_to_toon()
# → "name:str,description:str,parameters:{query:str,...}"
```

**Mesure (tiktoken + count_tokens)** :
- JSON tokens (5 outils) : **1026**
- TOON tokens : **737**
- **Économie : 28.2%** (−289 tokens par appel LLM)

**Multiplicateur d'usage** : **CHAQUE appel LLM** (agent loop + NLI + summary + decompose). Sur une requête agent typique → 2 à 5 appels LLM → **−580 à −1450 tokens par requête**.

**Compatibilité LLM** : ⚠️⚠️ **Risque élevé**. OpenAI/GitHub Models exigent un format JSON spécifique pour `tools` (avec `type: "function"`, etc.). Convertir le payload `tools[]` en TOON casserait l'API. **Le gain n'est PAS applicable tel quel** — il faudrait investiguer si les providers acceptent des formats alternatifs.

**Recommandation** : ⭐ **P3 (réévaluation)** — gain théorique élevé, mais incompatibilité probable avec l'API OpenAI. À investiguer upstream avant implémentation. Alternative : utiliser TOON uniquement pour la **description textuelle** des outils injectée dans le system prompt (si pertinent).

---

### P2-C : `MemoryStore` window + summary (`rag/nodes/memory_store.py:54,64`)

**Structure actuelle** : 
- `format_memory_context(summary, window)` : texte libre concaténé
- `format_chat_context(messages)` : turn-by-turn texte

```python
"Resume memoire: User asked about NVDA risks then MSFT risks...
Derniers echanges:
user: Quels sont les risques de NVDA ?
assistant: Les principaux risques incluent..."
```

**Forme TOON candidate** : `encode(memory)` →

```
summary: "User asked about NVDA risks then MSFT risks..."
turns[8]{role,content}:
  user,"Quels sont les risques de NVDA ?"
  assistant,"Les principaux risques incluent..."
```

**Mesure** :
- JSON tokens (8 turns) : **416**
- TOON tokens : **293**
- **Économie : 29.6%** (−123 tokens par tour)

**Multiplicateur d'usage** : 1× par tour de conversation (system prompt memory). Sur une conversation 10 tours → **−1230 tokens cumulés** + impact sur la fenêtre de contexte.

**Compatibilité LLM** : ✅ **Aucun risque** — c'est du contenu textuel injecté dans le system prompt, pas un payload structuré. Le LLM lit le texte, peu importe qu'il vienne de TOON ou de format custom.

**Recommandation** : ⭐⭐ **P2** — gain mesuré, compatibilité totale, code isolé. Bonne cible d'implémentation future.

---

## 5. Recommandations priorisées

### Court terme (à implémenter si pertinent)

1. **P1-A `format_rag_excerpts` → TOON** ⭐⭐⭐
   - Effort : ~10 lignes (`import toon_format`, `encode(mapping)` au lieu de la boucle texte)
   - Gain : −180 tokens / appel RAG
   - Risque : faible (texte → texte, validation prompt à faire)
   - **À faire en TDD** : test qui vérifie que `format_rag_excerpts` retourne du TOON valide et que le LLM le comprend.

2. **P2-C `MemoryStore` window → TOON** ⭐⭐
   - Effort : ~15 lignes (refactor `format_memory_context`)
   - Gain : −123 tokens / tour
   - Risque : très faible
   - **À faire en TDD** : test sur 8 turns de conversation simulée.

### Moyen terme (investigation requise)

3. **P1-B schémas outils** ⭐
   - **Bloqueur** : l'API OpenAI exige `tools: [{type: "function", function: {...}}]` en JSON. Conversion TOON incompatible.
   - **Piste** : vérifier si les providers acceptent une description textuelle des outils dans le system prompt (à la place du champ `tools`) → dans ce cas, TOON pourrait économiser 50%+ sur cette description.
   - **Action** : prototype avec un appel minimal, comparer la qualité des réponses.

### Hors scope (P4-P5)

- P3-D : `tool_events` (déjà format custom, gain marginal)
- P3-E : `_build_nli_prompt` (mixed-format, gain attendu ~15-20%)
- P5-F : `format_universe_hint` (3 tickers max, gain <5 tokens)

---

## 6. Risques identifiés

| Risque | Sévérité | Mitigation |
|--------|----------|------------|
| **Beta status (v0.9.x)** : API peut changer avant 1.0 | Moyen | Pin sur commit Git (`e475c82e`) ; surveiller releases upstream ; tests d'intégration détecteront les régressions |
| **Quoting rules strictes** : valeurs avec virgules, sauts de ligne, caractères spéciaux | Faible | Tests roundtrip couvrent les cas ; `decode(..., strict=True)` valide la syntaxe |
| **Incompatibilité API OpenAI tools** | Élevé (P1-B) | Investigation préalable, pas d'implémentation tant que pas validé |
| **Format TOON dans le system prompt** : le LLM peut-il l'interpréter correctement ? | Moyen | Test live avant déploiement ; fallback au format custom possible |
| **Dépendance `tiktoken`** : +1MB, déjà installé | Aucun | Coût négligeable |

---

## 7. État final

- ✅ `toon_format[pydantic]` 0.9.0b1 disponible dans le venv
- ✅ `tiktoken` 0.13.0 disponible
- ✅ 9 tests d'intégration verts
- ✅ 114/114 tests verts
- ✅ Aucun code `rag/` modifié (intégration **non invasive**)
- ✅ Audit P1+P2+P3 livré avec **mesures tiktoken** réelles
- ⏸️ Implémentation des optimisations : **différée** (sera scope d'une étape ultérieure si tu le décides)

**Prochaine étape possible** (si tu valides l'audit) : implémenter P1-A + P2-C en TDD, avec un test live dans Streamlit pour valider que le LLM interprète correctement le format TOON.
