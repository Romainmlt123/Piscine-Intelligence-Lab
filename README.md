# Agent Vocal Local - Migration WSL2

Ce projet a été migré vers WSL2 pour assurer la compatibilité WebSocket et améliorer les performances.

## Prérequis
- WSL2 (Ubuntu)
- Python 3.10+
- Ollama (installé via `curl https://ollama.ai/install.sh | sh`)

## Installation
Si ce n'est pas déjà fait, installez les dépendances :
```bash
./install_deps.sh  # (Si créé) ou
pip install -r requirements.txt
```
*Note: L'installation de Torch peut prendre du temps.*

## Démarrage Rapide
Un script de démarrage est fourni pour lancer Ollama et le serveur en une seule commande :

```bash
chmod +x start.sh
./start.sh
```

## Vérification
Pour tester la connexion WebSocket indépendamment :
```bash
python test_websocket.py
```

## Utilisation
1. Lancez le serveur (`./start.sh`).
2. Ouvrez votre navigateur Windows à l'adresse : `http://localhost:8001` (L'interface s'affichera directement).
3. Cliquez sur "Démarrer" et parlez.

## Structure
- `server.py`: Serveur FastAPI + WebSocket.
- `llm_module.py`: Interface avec Ollama.
- `stt_module.py`: Transcription avec Whisper.
- `tts_module.py`: Synthèse vocale (Piper/Fallback).
