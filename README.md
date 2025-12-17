<div align="center">

# 🎓 Tuteur IA

### **AI-Powered Voice Tutor for French Students**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-LLM-purple)](https://ollama.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

*A multi-agent voice assistant that helps students learn Math, Physics, and English through natural conversation — 100% local, privacy-first.*

[Features](#-features) • [Demo](#-demo) • [Architecture](#-architecture) • [Installation](#-installation) • [Performance](#-performance)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎤 **Voice-First** | Speak naturally, get audio responses in real-time |
| 🧠 **Multi-Agent RAG** | Specialized agents for Math, Physics, English with dedicated knowledge bases |
| 🔍 **Hybrid Search** | Vector + BM25 retrieval achieving **80% hit rate** |
| ⚡ **Streaming Response** | Token-by-token text + progressive audio (TTFA ~3s) |
| 🔒 **100% Local** | All processing on your machine — no cloud, no API keys |
| 📚 **Source Citations** | Every answer shows the document chunks it used |

---

## 🎬 Demo

<div align="center">

```
🎤 "What is Ohm's Law?"
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  🟢 PHYSICS • llama3.2:1b                                   │
│                                                             │
│  Ohm's Law states that the voltage U across a conductor    │
│  equals the product of current I and resistance R.         │
│  Formula: U = R × I                                        │
│                                                             │
│  📚 Sources: Livre troisième 2017.pdf (chunks 42, 43, 45)  │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
🔊 Audio response plays automatically
```

</div>

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          TUTEUR IA v1.6.0                            │
│                     Multi-Agent RAG + Voice                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   🎤 INPUT                    PROCESSING                   OUTPUT 🔊│
│   ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│   Audio ──▶ [Whisper] ──▶ [Router] ──▶ [Agent] ──▶ [Piper] ──▶ Audio│
│              (STT)      (Classify)   (RAG+LLM)     (TTS)            │
│                              │                                       │
│              ┌───────────────┼───────────────┐                      │
│              ▼               ▼               ▼                      │
│        ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│        │   MATH   │    │ PHYSICS  │    │ ENGLISH  │                │
│        │          │    │          │    │          │                │
│        │ ChromaDB │    │ ChromaDB │    │ ChromaDB │                │
│        │ 994 docs │    │1640 docs │    │  6 docs  │                │
│        │          │    │          │    │          │                │
│        │ + BM25   │    │ + BM25   │    │ + BM25   │                │
│        │          │    │          │    │          │                │
│        │ qwen2.5  │    │ llama3.2 │    │  gemma   │                │
│        │  :1.5b   │    │   :1b    │    │   :2b    │                │
│        └──────────┘    └──────────┘    └──────────┘                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | FastAPI + Uvicorn | WebSocket server, async streaming |
| **STT** | OpenAI Whisper (base) | Speech-to-text transcription |
| **Embeddings** | sentence-transformers | `paraphrase-multilingual-MiniLM-L12-v2` |
| **Vector DB** | ChromaDB | Persistent vector storage |
| **Keyword Search** | rank-bm25 | BM25 for hybrid retrieval |
| **LLM** | Ollama | Local inference (Qwen, Llama, Gemma) |
| **TTS** | Piper | Neural text-to-speech (French) |
| **Frontend** | Vanilla HTML/CSS/JS | ChatGPT-style interface |

---

## 📊 Performance

### RAG Benchmark (20 test queries)

| Metric | Baseline | After Optimization | Improvement |
|--------|----------|-------------------|-------------|
| **Hit Rate** | 46.7% | **80.0%** | +71% 🚀 |
| **MRR** | 0.413 | **0.717** | +74% 🚀 |
| **Latency** | 25ms | 27ms | +2ms only |

### Optimization Journey

```
Iteration 1: Baseline vector search         → 46.7%
Iteration 2: Multilingual embeddings        → 53.3%
Iteration 3: Cleaned corrupted PDFs         → 60.0%
Iteration 4: Hybrid Search (BM25 + Vector)  → 80.0% ✅
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed
- ~8GB RAM recommended

### 1. Clone & Setup

```bash
git clone https://github.com/Romainmlt123/Piscine-Intelligence-Lab.git
cd Piscine-Intelligence-Lab

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download LLM Models

```bash
ollama pull qwen2.5:1.5b      # Math agent
ollama pull llama3.2:1b       # Physics agent
ollama pull gemma:2b          # English agent
```

### 3. Download TTS Model

```bash
mkdir -p models/piper
# Download from https://github.com/rhasspy/piper/releases
# Place fr_FR-upmc-medium.onnx in models/piper/
```

### 4. Run

```bash
./start.sh
```

Open **http://localhost:8001** and click the microphone! 🎤

---

## 📁 Project Structure

```
├── src/
│   ├── main.py              # FastAPI WebSocket server
│   ├── config.py            # Centralized configuration
│   ├── agents/
│   │   ├── orchestrator.py  # Multi-agent routing
│   │   └── llm_module.py    # Ollama wrapper
│   ├── rag/
│   │   ├── rag_module.py    # Hybrid RAG (Vector + BM25)
│   │   └── rag_benchmark.py # Performance testing
│   └── speech/
│       ├── stt_module.py    # Whisper STT
│       ├── tts_module.py    # Piper TTS
│       └── vad_module.py    # Voice Activity Detection
├── static/
│   └── index.html           # ChatGPT-style frontend
├── knowledge_base/
│   ├── math/                # 994 chunks (6ème PDFs, Terminale)
│   ├── physics/             # 1640 chunks (3ème-Terminale books)
│   └── english/             # Grammar basics
├── docs/
│   ├── RAG_EVALUATION.md    # Benchmark methodology & results
│   ├── PRESENTATION_IO.md   # I/O architecture diagram
│   └── RAPPORT_RECHERCHE.md # Technical research report
└── models/
    └── piper/               # TTS voice models
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [RAG Evaluation](docs/RAG_EVALUATION.md) | Benchmark methodology, 4 optimization iterations |
| [I/O Presentation](docs/PRESENTATION_IO.md) | System architecture with diagrams |
| [Research Report](docs/RAPPORT_RECHERCHE.md) | Technical deep-dive (French) |
| [Model Choices](docs/model_choices.md) | LLM selection rationale |

---

## 🔮 Future Improvements

- [ ] **Conversation Memory** — Multi-turn context retention
- [ ] **Tool Calling** — Calculator for math computations
- [ ] **Cross-encoder Reranking** — Further improve retrieval
- [ ] **More Languages** — Spanish, German agents

---

## 👨‍💻 Author

**Romain Mallet**

---

## 📝 License

MIT License — feel free to use and modify!

---

<div align="center">

**⭐ Star this repo if you find it useful!**

</div>
