# Tuteur IA - Rapport de Recherche

## Résumé Exécutif

**Tuteur IA** est un assistant vocal éducatif basé sur une architecture **Multi-Agent RAG** (Retrieval-Augmented Generation). Le système permet aux étudiants de poser des questions orales en mathématiques, physique ou anglais, et reçoit des réponses vocales contextualisées grâce à une base de connaissances spécialisée.

**Résultats clés :**
- **Hit Rate RAG : 80%** (amélioration de +73% vs baseline)
- **Latence moyenne : 46ms** par requête
- **4 agents spécialisés** avec LLMs différenciés

---

## 1. Problématique

### Contexte
Les tuteurs humains sont coûteux et limités en disponibilité. Les chatbots génériques (ChatGPT, Bard) ne sont pas optimisés pour l'éducation française et manquent de sources vérifiables pour le programme scolaire.

### Objectifs
1. Créer un assistant vocal **full local** (privacy-first)
2. Réponses **contextualisées** par le programme scolaire français
3. Architecture **multi-agents** pour spécialisation par matière
4. **Citer les sources** pour vérifiabilité

---

## 2. Architecture Technique

### 2.1 Vue d'ensemble

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   AUDIO     │────▶│   WHISPER   │────▶│  ROUTEUR    │
│   (Voice)   │     │   (STT)     │     │  (Agent)    │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
            ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
            │ MATH AGENT  │            │PHYSICS AGENT│            │ENGLISH AGENT│
            │  RAG + LLM  │            │  RAG + LLM  │            │  RAG + LLM  │
            └─────────────┘            └─────────────┘            └─────────────┘
                    │                          │                          │
                    └──────────────────────────┼──────────────────────────┘
                                               ▼
                                       ┌─────────────┐
                                       │   PIPER     │
                                       │   (TTS)     │
                                       └─────────────┘
```

### 2.2 Composants

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **STT** | Whisper (base) | Transcription vocale |
| **Routing** | Classification par mots-clés | Sélection de l'agent |
| **RAG** | ChromaDB + Hybrid Search | Récupération de contexte |
| **LLM** | Ollama (qwen2.5, llama3.2, gemma) | Génération de réponse |
| **TTS** | Piper | Synthèse vocale |

---

## 3. Méthodologie RAG

### 3.1 Problème initial
Le RAG classique (vector search uniquement) atteignait seulement **46.7% de hit rate** sur nos requêtes de test.

### 3.2 Approche Hybrid Search

Nous avons implémenté une **recherche hybride** combinant :

1. **Vector Search** (embeddings sémantiques)
   - Modèle : `paraphrase-multilingual-MiniLM-L12-v2`
   - Capture la similarité sémantique

2. **BM25** (keyword matching)
   - Indexation par tokens
   - Capture les correspondances exactes (dates, formules, noms)

3. **Reciprocal Rank Fusion (RRF)**
   - Combinaison des deux rankings
   - Formule : `score = 1/(k + rank_vector) + weight * 1/(k + rank_bm25)`
   - k = 60, weight = 0.3

### 3.3 Résultats

| Métrique | Baseline | Multilingual | Hybrid |
|----------|----------|--------------|--------|
| **Hit Rate** | 40.0% | 46.7% | **80.0%** |
| **MRR** | 0.356 | 0.413 | **0.717** |
| **Latence** | 25ms | 25ms | **27ms** |

**Amélioration de +100%** avec seulement +2ms de latence.

---

## 4. Base de Connaissances

### 4.1 Corpus

| Matière | Documents | Chunks | Niveaux |
|---------|-----------|--------|---------|
| **Mathématiques** | 17 | 994 | 6ème, Terminale |
| **Physique** | 9 | 1640 | 3ème-Terminale |
| **Anglais** | 3 | 6 | Collège |

### 4.2 Sources
- Cahiers d'exercices officiels (6ème)
- Livres de physique-chimie (Nathan, Bordas)
- Fiches de cours niveau lycée

---

## 5. Évaluation

### 5.1 Protocole
- 20 questions de test couvrant les 3 matières
- Évaluation par keywords matching
- Métriques : Hit Rate, MRR, Latence

### 5.2 Itérations d'optimisation

| Itération | Action | Hit Rate |
|-----------|--------|----------|
| 1 | Baseline (all-MiniLM) | 40.0% |
| 2 | Multilingual embeddings | 46.7% |
| 3 | Nettoyage PDFs corrompus | 53.3% |
| 4 | Adaptation des requêtes | 60.0% |
| 5 | **Hybrid Search (BM25+Vector)** | **80.0%** |

---

## 6. Limites et Perspectives

### 6.1 Limites actuelles
- Pas de mémoire conversationnelle
- Pas d'exécution d'outils (calculatrice)
- Corpus anglais limité

### 6.2 Améliorations futures
1. **Mémoire de conversation** : Garder le contexte des échanges précédents
2. **Tool Calling** : Intégrer une calculatrice pour les calculs complexes
3. **Reranking** : Ajouter un modèle de reranking (cross-encoder)
4. **Évaluation humaine** : Tests avec de vrais étudiants

---

## 7. Conclusion

Le système **Tuteur IA** démontre qu'une architecture Multi-Agent RAG locale peut atteindre des performances compétitives (80% hit rate) avec une latence acceptable (46ms). L'approche hybride (Vector + BM25) s'est révélée cruciale pour améliorer la précision sans sacrifier la vitesse.

---

## Références

1. Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
2. Robertson et al. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond"
3. Cormack et al. (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"

---

**Auteur :** Romain Mallet  
**Date :** Décembre 2024  
**Version :** 1.6.0
