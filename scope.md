# Projet de session : du RAG à l'orchestration agentique

Ce projet se déroule en **trois itérations**, chacune correspondant à une phase précise du cycle de vie d'un système IA moderne.  
Chaque équipe travaillera sur un **même sujet choisi en début de session**, qui servira de fil conducteur.

---

## Phase 1 – Construction d'un système RAG

### Objectif
Construire un système de **Retrieval-Augmented Generation (RAG)** capable d'ingérer et d'exploiter plusieurs types de fichiers et sources d'information.

### Capacités attendues
- Traiter :
  - documents textuels
  - fichiers PDF
  - tableaux
  - fichiers structurés / semi-structurés
  - idéalement plusieurs formats hétérogènes liés au sujet
- Comparer les stratégies vues en cours :
  - découpage des documents
  - indexation
  - embeddings
  - recherche hybride
  - reranking
  - gestion des métadonnées
  - traitement des tableaux
  - qualité des réponses générées

### Livrables attendus (démonstration + analyse)
- Expliquer les choix techniques effectués
- Présenter les limites rencontrées
- Discuter les compromis entre :
  - précision / rappel
  - coût / latence / complexité

---

## Phase 2 – Construction d'un système avec 1 agent et MCP

### Objectif
Construire un système composé d'un **agent LLM unique**, connecté à votre RAG et à des outils via une approche de type **MCP (Model-Context-Protocol)**.

### Capacités attendues
L'agent doit pouvoir :
- utiliser le RAG comme source de connaissance
- appeler **un ou plusieurs outils** pour accomplir une tâche plus complète

#### Exemples d'actions possibles selon le sujet
- Consulter des documents
- Extraire des informations pertinentes
- Appeler une API
- Comparer des options
- Produire une recommandation
- Générer un rapport
- Valider une réponse à partir de sources
- Exécuter une action simulée dans un environnement contrôlé

### Livrables attendus
- Démonstration de la combinaison : raisonnement + récupération d'information + outils
- Explication :
  - ce qui a bien fonctionné
  - ce qui a été difficile
  - quels patrons d'architecture ont été les plus robustes

---

## Phase 3 – Orchestration agentique complète : Architecture "Hub-and-Spoke"

### Objectif
Transformer le système en une architecture plus complète d'**orchestration IA**, dépassant l'agent unique, en implémentant une topologie "Hub-and-Spoke" supervisée spécialisée pour la gestion de portefeuille.

### Architecture et Agents Spécialisés
Le système s'articule autour d'un superviseur central (le Hub) et d'agents experts (les Spokes) opérant dans des silos stricts :

| Agent | Rôle et Outils | Responsabilité |
|-------|----------------|----------------|
| **👔 Portfolio Manager (Hub)** | *Aucun outil direct* | Cerveau pur : décompose la requête, délègue, synthétise les rapports, prend de la hauteur et génère le plan d'investissement. |
| **📚 Analyste Fondamental (Spoke)** | `sec_filings_rag_tool`, `get_news_tool` | Expert qualitatif : analyse le RAG (fichiers SEC) et lit les dernières news pour capter le sentiment de marché. |
| **📈 Analyste Quantitatif (Spoke)** | `market_price_tool`, `portfolio_history_tool` | Expert quantitatif : analyse les prix du marché et la courbe de performance historique du portefeuille. |
| **🛡️ Compliance Validator (Guardrail)** | `validate_claims_tool`, `portfolio_info_tool`, `account_activity_tool` | Gestionnaire de risques : vérifie le "Buying Power", les positions ouvertes et l'historique récent pour valider ou rejeter un ordre. |
| **⚡ Executor Trader** | `place_trade_tool`, `close_position_tool` | Bras armé : interagit de manière exclusive avec le marché (ouvrir/liquider) **uniquement** après un "PASS" explicite du Compliance Validator. |

### Flux d'exécution (Communication)
1. **Délégation :** Le Portfolio Manager reçoit la mission et la découpe en tâches spécifiques pour les analystes.
2. **Exécution isolée (Parallèle) :** L'Analyste Fondamental et l'Analyste Quantitatif exécutent leurs recherches simultanément.
3. **Consolidation :** Les analystes renvoient leurs données. Le PM lit tout et prend la décision d'investissement.
4. **Validation :** Le PM soumet sa décision formelle au Compliance Validator.
5. **Action ou Boucle de correction :** - Si validé, le dossier passe à l'Exécuteur. 
   - Si refusé (ex: manque de fonds), le dossier retourne au PM avec les raisons du refus pour ajustement du plan.

### Capacités attendues (notions avancées)
- Plusieurs agents spécialisés avec une **séparation stricte des rôles**.
- Exécution de tâches en **parallèle** pour optimisation de la latence.
- **Supervision des actions** et boucle de rétroaction (PM ↔ Compliance).
- **Sécurité et gestion des risques** (implémentation de garde-fous stricts avant exécution).
- Coordination complexe entre outils locaux, appels API externes et historique conversationnel (état partagé).
- Traçabilité complète des décisions (logique d'évaluation et justification des trades).

### Livrable attendu
- Montrer **pourquoi** cette orchestration améliore la robustesse, la sécurité (absence d'hallucinations d'ordres de bourse) et l'efficacité du système par rapport aux phases précédentes.

---

# Présentations en classe (communes aux 3 phases)

Pour **chaque phase**, chaque équipe dispose d'environ **15 minutes** de présentation.

## Structure obligatoire de la présentation

| Section | Contenu attendu |
|---------|----------------|
| **Ce que vous avez construit** | Objectif, architecture, fonctionnalités principales |
| **Choix techniques** | Approches retenues, outils utilisés, justifications |
| **Difficultés rencontrées** | Problèmes, limites, blocages, échecs observés (ex: gestion des boucles PM/Compliance en Phase 3) |
| **Apprentissages intéressants** | Découvertes surprenantes, erreurs instructives, apprentissages inattendus |
| **Patrons qui ont le mieux fonctionné** | Stratégies les plus efficaces et pourquoi |
| **Lien avec la littérature scientifique** | **Au moins un article scientifique** pertinent par phase (voir détail ci-dessous) |

---

# Ancrage scientifique obligatoire

Pour **chacune des trois phases**, chaque équipe doit identifier et mobiliser **au moins un article scientifique pertinent** :

| Phase | Thème scientifique recherché |
|-------|------------------------------|
| Phase 1 | Article lié au **RAG** |
| Phase 2 | Article lié aux **agents LLM**, à l'utilisation d'outils ou à **MCP** |
| Phase 3 | Article lié aux **systèmes multi-agents**, à l'orchestration (ex: hiérarchies d'agents, Hub-and-Spoke) ou à l'évaluation des systèmes agentiques |

### Usage attendu des articles
L'article doit :
- appuyer votre travail
- expliquer une méthode que vous avez utilisée
- clarifier une difficulté rencontrée
- justifier un choix d'architecture (ex: justifier pourquoi le PM n'a pas d'outils via la littérature sur la charge cognitive des LLM ou la sécurité multi-agents)

> L'objectif n'est pas simplement de citer un article, mais de montrer que votre conception technique est **reliée à des travaux de recherche existants**.

---

# Résultat final attendu à la fin de la session

Chaque équipe aura produit une **trajectoire complète et démontrable** :