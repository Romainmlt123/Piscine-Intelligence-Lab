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
│        │  FAISS   │    │  FAISS   │    │  FAISS   │                │
│        │ E5 + BM25│    │ E5 + BM25│    │ E5 + BM25│                │
│        │          │    │          │    │          │                │
│        │+ rerank  │    │+ rerank  │    │+ rerank  │                │
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
| **Embeddings** | sentence-transformers | `intfloat/multilingual-e5-small` (default), `multilingual-e5-base` (option) |
| **Vector index** | FAISS (`IndexFlatIP`) | Exact cosine search, persisted on disk |
| **Re-ranking** | `CrossEncoder` | Rescoring of the retrieved candidates (`mmarco-mMiniLMv2`) |
| **Keyword Search** | rank-bm25 | Optional BM25 fusion (RRF) |
| **LLM** | Ollama | Local inference (Qwen, Llama, Gemma) |
| **TTS** | Piper | Neural text-to-speech (French) |
| **Frontend** | Vanilla HTML/CSS/JS | ChatGPT-style interface |

---

## 📊 Performance

### RAG Benchmark

The retrieval stack was migrated from ChromaDB + MiniLM to FAISS + E5 + cross-encoder re-ranking.
Results of the previous stack (hit rate 46.7% → 80.0%) are kept for history in
[docs/RAG_EVALUATION.md](docs/RAG_EVALUATION.md) and **do not apply to the current pipeline**.

Reproduce the measurements of the current pipeline (20 queries, ablation dense / hybrid / re-ranking):

```bash
python -m src.rag.download_models --e5-base   # once, needs network access
python -m src.rag.rag_benchmark --run
python -m src.rag.rag_benchmark --report
```

The benchmark reports keyword hit / MRR, source hit / MRR, mean and p95 latency, and the environment.

### Fully local runtime

After `python -m src.rag.download_models`, set `RAG_OFFLINE=1`: no request is made to the
Hugging Face Hub, and the RAG makes no network call at runtime.
Settings: `RAG_EMBEDDING_MODEL`, `RAG_RERANKER_MODEL`, `RAG_USE_RERANKER`, `RAG_RERANK_CANDIDATES`.

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
│   │   ├── rag_module.py    # RAG: FAISS + E5 + cross-encoder re-ranking
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
