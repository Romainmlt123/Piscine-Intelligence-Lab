# 🎓 Voice Agent - Professeur IA Vocal

Un assistant vocal intelligent avec RAG multi-agent et TTS streaming.

## 🚀 Fonctionnalités

- **Multi-Agent RAG** : Agents spécialisés (Math, Physics, English) avec routing intelligent
- **Multi-Modèle** : LLM différent par matière (Qwen, Llama, Gemma)
- **Streaming TTS** : Réponse audio progressive avec Piper
- **Math-to-Speech** : Conversion des équations en texte parlé
- **Faible latence** : TTFA optimisé (~15-25s sur CPU)

## 📁 Structure du projet

```
├── src/                    # Code source
│   ├── main.py             # Point d'entrée FastAPI
│   ├── config.py           # Configuration centralisée
│   ├── agents/             # Orchestrateur et LLM
│   ├── rag/                # Module RAG (ChromaDB)
│   └── speech/             # STT, TTS, VAD, Math-to-Speech
├── static/                 # Frontend (index.html)
├── knowledge_base/         # Documents RAG
├── models/                 # Modèles (Piper TTS)
├── data/                   # Runtime (ChromaDB, logs)
├── docs/                   # Documentation
└── tests/                  # Tests
```

## ⚙️ Installation

```bash
# Cloner le repo
git clone <repo-url>
cd voice-agent

# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Configurer (optionnel)
cp .env.example .env
# Éditer .env selon vos besoins
```

## 🏃 Démarrage

```bash
./start.sh
```

Puis ouvrir http://localhost:8001

## 🧠 Modèles requis

### Ollama (LLM)
```bash
ollama pull qwen2.5:1.5b
ollama pull llama3.2:1b
ollama pull gemma:2b
```

### Piper (TTS)
Télécharger depuis https://github.com/rhasspy/piper/releases et placer dans `models/piper/`

## 📊 Configuration

Voir `.env.example` pour toutes les options :
- Modèles LLM par matière
- Agressivité VAD
- Paramètres de streaming

## 📖 Documentation

- [CHANGELOG v1.4.0](docs/CHANGELOG_v1.4.0.md) - Historique des modifications
- [Choix des modèles](docs/model_choices.md) - Justification des LLM

## 📝 License

MIT
