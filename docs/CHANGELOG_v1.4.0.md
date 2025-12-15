# Documentation v1.4.0 - Streaming TTS et RAG Multi-Agent

## Vue d'ensemble

Cette documentation retrace les améliorations apportées au système de Professeur IA Vocal, notamment l'optimisation de la latence, le streaming TTS, et l'architecture multi-modèle par matière.

---

## 1. Problèmes initiaux identifiés

### 1.1 Audio qui se superpose
**Symptôme** : Plusieurs chunks audio se jouaient simultanément, rendant le son incompréhensible.

**Causes racines** :
1. **Pas de backpressure** : Le serveur envoyait les chunks plus vite que le client ne pouvait les jouer
2. **Race condition** dans le frontend : Le flag `isPlayingAudio` était vérifié avant que `.play()` ne soit résolu
3. **Aucune coordination** : Pas de mécanisme d'indexation des chunks

### 1.2 TTFA (Time To First Audio) trop élevé
**Symptôme** : 80+ secondes avant d'entendre le premier son.

**Causes** :
1. Le `SentenceBuffer` attendait des phrases trop longues (150+ caractères)
2. Piper TTS prend ~2-3s par chunk sur CPU
3. Le LLM générait des réponses très longues

### 1.3 STT qui tronque les phrases
**Symptôme** : "équation du second degré" était transcrit comme "du second degré".

**Cause** : VAD trop agressif (mode 3) avec padding insuffisant (300ms).

---

## 2. Solutions implémentées

### 2.1 Architecture Audio Streaming

#### Nouveau module `audio_streamer.py`

```
┌─────────────────────────────────────────────────────────────────┐
│                          SERVER                                 │
│  ┌─────────┐    ┌─────────────┐    ┌─────────┐                 │
│  │   LLM   │───►│ Text Buffer │───►│   TTS   │                 │
│  │ Stream  │    │ (Sentences) │    │  Piper  │                 │
│  └─────────┘    └─────────────┘    └────┬────┘                 │
│                                         │                       │
│                     ┌───────────────────▼───────────────────┐  │
│                     │     AudioStreamManager                │  │
│                     │  (Thread-safe FIFO, indexed chunks)   │  │
│                     └───────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ WebSocket (indexed)
┌─────────────────────────────────────────────────────────────────┐
│                          CLIENT                                 │
│                     ┌───────────────────────────────────────┐  │
│                     │         AudioPlayer Class             │  │
│                     │  - Queue ordonnée par index           │  │
│                     │  - await sur audio.play()             │  │
│                     │  - onended avant chunk suivant        │  │
│                     └───────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Composants clés** :
- **AudioStreamManager** : Worker TTS dans un thread séparé, queue FIFO thread-safe
- **SentenceBuffer** : Découpe le texte sur les frontières sémantiques (`.`, `!`, `?`, `,`)
- **AudioPlayer (JS)** : Classe avec queue triée par index et playback bloquant

#### Fichiers modifiés
- `audio_streamer.py` (NOUVEAU)
- `server.py` : Intégration AudioStreamManager, envoi `audio_chunk_meta` + `audio_reset`
- `static/index.html` : Classe AudioPlayer avec gestion d'index

---

### 2.2 Optimisation TTFA

| Paramètre | Avant | Après | Impact |
|-----------|-------|-------|--------|
| SentenceBuffer `min_chars` | 150 | 5 | Chunks plus petits, plus rapides |
| SentenceBuffer `max_chars` | 200 | 50 | Force le découpage |
| Premier chunk threshold | 20 | 5 | Premier audio ultra-rapide |
| Prompts LLM | Longs | "Sois CONCIS. Phrases COURTES." | Réponses plus courtes |

**Résultat** : TTFA passé de **80s** à **~15-25s** ⬇️

---

### 2.3 Correction VAD

| Paramètre | Avant | Après |
|-----------|-------|-------|
| Agressivité | 3 (max) | 2 (medium-high) |
| Padding | 300ms | 600ms |

**Résultat** : Phrases capturées complètement.

---

### 2.4 Architecture Multi-Modèle par Matière

#### Configuration des agents

| Matière | Modèle LLM | RAG Collection | Couleur Badge |
|---------|------------|----------------|---------------|
| 🔵 MATH | `qwen2.5:1.5b` | `math_agent` | Bleu |
| 🔴 PHYSICS | `llama3.2:1b` | `physics_agent` | Rouge |
| 🟢 ENGLISH | `gemma:2b` | `english_agent` | Vert |
| 🟣 GENERAL | `qwen2.5:1.5b` | - | Violet |

#### Routing intelligent

1. **Keyword Matching** (rapide) : Détection par mots-clés
   - Math : équation, calcul, algèbre, x², dérivée...
   - Physics : gravité, force, newton, énergie...
   - English : grammar, tense, vocabulary, conjugation...

2. **LLM Fallback** (lent) : Si aucun keyword, le LLM classifie

#### Fichiers modifiés
- `orchestrator.py` : 4 instances LLM, 3 RAG modules, routing ENGLISH
- `llm_module.py` : Ajout `model_name` property
- `server.py` : Envoi du `model_name` dans les events
- `static/index.html` : Affichage modèle dans le badge

---

### 2.5 Math-to-Speech

#### Nouveau module `math_to_speech.py`

Convertit les symboles mathématiques en français parlé **avant** le TTS.

| Symbole | Conversion |
|---------|------------|
| `x²` | "x au carré" |
| `x³` | "x au cube" |
| `x^n` | "x puissance n" |
| `√` | "racine carrée de" |
| `=` | "égale" |
| `+` `-` `×` `÷` | "plus" "moins" "multiplié par" "divisé par" |
| `π` `θ` `α` | "pi" "thêta" "alpha" |
| `∫` `∑` | "intégrale de" "somme de" |
| `1/2` | "un demi" |
| `x₁` | "x indice 1" |

**Exemple** :
```
Entrée:  "x² + 2x - 4 = 0"
Sortie:  "x au carré plus 2x moins 4 égale, 0"
```

---

## 3. Knowledge Base

### Structure des dossiers
```
knowledge_base/
├── math/
│   └── equations.txt       # Équations du second degré
├── physics/
│   └── gravitation.txt     # Loi de la gravitation
└── english/
    ├── grammar_basics.txt  # Present simple, perfect, etc.
    └── vocabulary.txt      # Idioms, connectors
```

---

## 4. Métriques de performance

### Latence typique (CPU only)

| Étape | Durée |
|-------|-------|
| STT (Whisper base) | 3-6s |
| Routing (keywords) | ~0s |
| Routing (LLM fallback) | ~10s |
| RAG | 0.3-0.5s |
| **TTFA** | **15-25s** |
| LLM total | 20-80s (selon longueur) |
| TTS par chunk | 2-3s |

### Bottlenecks identifiés

1. **LLM sur CPU** : Le temps de génération du premier token est élevé
2. **Piper TTS sur CPU** : ~2-3s par phrase
3. **Whisper sur CPU** : ~4-6s par transcription

### Recommandations pour aller plus loin

- **GPU** : Accélérerait tout de 5-10x
- **Whisper tiny** : Plus rapide mais moins précis
- **TTS espeak** : Instantané mais voix robotique
- **Modèles quantifiés** : Q4_K_M pour réduire la mémoire

---

## 5. API WebSocket

### Messages serveur → client

| Type | Description |
|------|-------------|
| `user_text` | Transcription STT |
| `ai_text_chunk` | Token LLM (streaming) |
| `ai_text` | Texte complet final |
| `rag_sources` | Contexte RAG utilisé |
| `audio_reset` | Reset du player audio |
| `audio_chunk_meta` | Métadonnées chunk (index) |
| `latency_metrics` | Métriques de performance |
| (binary) | Données audio WAV |

---

## 6. Fichiers modifiés/créés

### Nouveaux fichiers
- `audio_streamer.py` - Gestion streaming audio
- `math_to_speech.py` - Conversion maths → parole
- `knowledge_base/english/` - Documents anglais
- `docs/model_choices.md` - Documentation modèles

### Fichiers modifiés
- `server.py` - Intégration AudioStreamManager
- `orchestrator.py` - Multi-modèle, ENGLISH
- `llm_module.py` - model_name property
- `vad_module.py` - Agressivité réduite
- `static/index.html` - AudioPlayer class, badges colorés

---

## 7. Commandes utiles

```bash
# Démarrer le serveur
./start.sh

# Lister les modèles Ollama
ollama list

# Installer un nouveau modèle
ollama pull <model_name>

# Tester math-to-speech
source venv/bin/activate
python -c "from math_to_speech import convert_math_to_speech; print(convert_math_to_speech('x² = 4'))"
```

---

## 8. Prochaines étapes possibles

1. **GPU Support** : Activer CUDA pour Whisper/LLM/TTS
2. **Voix personnalisée** : Entraîner une voix Piper custom
3. **Multi-langue** : Détecter la langue et adapter TTS
4. **Historique conversation** : Persister les échanges
5. **UI améliorée** : Visualisation formules LaTeX
