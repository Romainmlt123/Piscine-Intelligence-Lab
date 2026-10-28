# Tuteur IA - Présentation I/O

## 🎯 Objectif

**Créer un assistant vocal éducatif full-local** qui aide les étudiants français en mathématiques, physique et anglais, avec des réponses contextualisées et sourcées.

---

## 📥 INPUT (Entrées)

### Entrée Utilisateur
```
┌─────────────────────────────────────────┐
│         🎤 VOIX DE L'UTILISATEUR         │
│                                         │
│  "C'est quoi la loi d'Ohm ?"           │
│  "Comment calculer une intégrale ?"    │
│  "Explique le present perfect"         │
└─────────────────────────────────────────┘
```

### Données du Système

| Type | Description | Volume |
|------|-------------|--------|
| **Math** | Cahiers 6ème + Fiches Terminale | 994 chunks |
| **Physique** | Livres 3ème-Terminale | 1640 chunks |
| **Anglais** | Fiches collège | 6 chunks |
| **Total** | Tous corpus combinés | **2640 chunks** |

### Modèles Utilisés

| Modèle | Usage | Taille |
|--------|-------|--------|
| Whisper base | STT | ~74MB |
| paraphrase-multilingual-MiniLM-L12-v2 | Embeddings | ~118MB |
| qwen2.5:1.5b | LLM Math | 1.5B params |
| llama3.2:1b | LLM Physics | 1B params |
| gemma:2b | LLM English | 2B params |
| Piper fr_FR-upmc-medium | TTS | ~65MB |

---

## ⚙️ PROCESSING (Traitement)

### Pipeline Complet

```
    🎤 AUDIO                                                    🔊 AUDIO
       │                                                            ▲
       ▼                                                            │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   WHISPER   │───▶│   ROUTER    │───▶│   RAG +     │───▶│    PIPER    │
│    (STT)    │    │  (Classify) │    │    LLM      │    │    (TTS)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │   MATH    │   │  PHYSICS  │   │  ENGLISH  │
    │           │   │           │   │           │
    │  RAG:     │   │  RAG:     │   │  RAG:     │
    │  994 docs │   │  1640 docs│   │  6 docs   │
    │           │   │           │   │           │
    │  LLM:     │   │  LLM:     │   │  LLM:     │
    │  qwen2.5:1.5b  │   │  llama3.2:1b │   │  gemma:2b │
    └───────────┘   └───────────┘   └───────────┘
```

### Détail du RAG Hybrid Search

```
                    ┌────────────────────┐
                    │       QUERY        │
                    │ "loi d'Ohm"        │
                    └────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  VECTOR SEARCH  │           │      BM25       │
    │   (sémantique)  │           │  (mots-clés)    │
    │     70%         │           │     30%         │
    └────────┬────────┘           └────────┬────────┘
             │                             │
             └──────────────┬──────────────┘
                            ▼
                  ┌─────────────────┐
                  │  RRF FUSION     │
                  │  (k=60)         │
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │  TOP 5 CHUNKS   │
                  └─────────────────┘
```

---

## 📤 OUTPUT (Sorties)

### Réponse Vocale
```
┌─────────────────────────────────────────┐
│         🔊 RÉPONSE AUDIO STREAMÉE        │
│                                         │
│  "La loi d'Ohm établit que la tension  │
│   U aux bornes d'un conducteur         │
│   ohmique est égale au produit de      │
│   l'intensité I par la résistance R.   │
│   On écrit U = R × I."                 │
└─────────────────────────────────────────┘
```

### Interface Web (ChatGPT-style)

| Élément | Description |
|---------|-------------|
| **Avatar** | 🎓 pour IA, 👤 pour utilisateur |
| **Badge** | `PHYSICS • llama3.2:1b` |
| **Texte streaming** | Réponse token par token |
| **Sources RAG** | 5 chunks avec préview |
| **Métriques** | STT, LLM, TTS, Total |

---

## 📊 Résultats Mesurés

### Performance RAG (Benchmark 20 questions)

| Métrique | Baseline | Hybrid Search | Amélioration |
|----------|----------|---------------|--------------|
| **Hit Rate** | 46.7% | **80.0%** | **+71%** |
| **MRR** | 0.413 | **0.717** | **+74%** |
| **Latence RAG** | 25ms | **27ms** | +2ms seulement |

### Évolution par Itération

| Itération | Action | Hit Rate |
|-----------|--------|----------|
| 1 | Benchmark initial | 46.7% |
| 2 | Nettoyage PDFs corrompus | 53.3% |
| 3 | Questions optimisées | 60.0% |
| **4** | **Hybrid Search (BM25+Vector)** | **80.0%** |

### Niveau de Performance

| Métrique | Valeur | Seuil | Niveau |
|----------|--------|-------|--------|
| Hit Rate | 80% | ≥70% = BON | 🟢 **BON** |
| MRR | 0.717 | ≥0.60 = BON | 🟢 **BON** |

---

## 🏛️ Architecture Complète

```
┌──────────────────────────────────────────────────────────────────────┐
│                         TUTEUR IA v1.6.0                             │
│                    (Multi-Agent RAG + Voice)                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                        FRONTEND                               │   │
│  │              index.html (ChatGPT-style)                      │   │
│  │              WebSocket + Audio Streaming                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                         BACKEND                               │   │
│  │                    FastAPI + Uvicorn                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │   WHISPER  │  │  ROUTER    │  │  AGENTS    │  │   PIPER    │    │
│  │   (STT)    │  │ (Classify) │  │ (RAG+LLM)  │  │   (TTS)    │    │
│  │            │  │            │  │            │  │            │    │
│  │  base      │  │ mots-clés  │  │ 3 agents   │  │ fr_FR      │    │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
│                                          │                          │
│                    ┌─────────────────────┼─────────────────────┐    │
│                    ▼                     ▼                     ▼    │
│             ┌───────────┐         ┌───────────┐         ┌─────────┐│
│             │   MATH    │         │  PHYSICS  │         │ ENGLISH ││
│             │           │         │           │         │         ││
│             │ FAISS     │         │ FAISS     │         │FAISS    ││
│             │ 994 chunks│         │1640 chunks│         │ 6 chunks││
│             │           │         │           │         │         ││
│             │ + BM25    │         │ + BM25    │         │ + BM25  ││
│             │           │         │           │         │         ││
│             │ qwen2.5:  │         │ llama3.2: │         │ gemma:  ││
│             │  1.5b    │         │    1b     │         │   2b    ││
│             └───────────┘         └───────────┘         └─────────┘│
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ✅ Bénéfices

| Effet | Description |
|-------|-------------|
| 🎯 **Précision** | 80% hit rate sur les questions scolaires |
| ⚡ **Rapidité** | 27ms latence RAG |
| 🔒 **Privacy** | 100% local, aucun cloud |
| 📚 **Sourçable** | Citations des documents originaux |
| 🗣️ **Vocal** | Mains libres, naturel |

---

## 🔗 Liens

- **GitHub** : https://github.com/Romainmlt123/Piscine-Intelligence-Lab
- **Branch** : romain-pipeline-agent-ia
- **Version** : v1.6.0
