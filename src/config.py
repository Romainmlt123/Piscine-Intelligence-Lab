"""
Configuration Module

Centralized configuration for the Voice Agent application.
Uses environment variables with sensible defaults.
"""

import os
from pathlib import Path


# Base Paths
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
STATIC_DIR = BASE_DIR / "static"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Piper TTS
PIPER_DIR = MODELS_DIR / "piper"
PIPER_MODEL = os.getenv("PIPER_MODEL", str(PIPER_DIR / "fr_FR-upmc-medium.onnx"))

# ChromaDB
CHROMA_DB_DIR = str(DATA_DIR / "chroma_db")

# Whisper STT
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# LLM Models (Ollama)
LLM_MODEL_MATH = os.getenv("LLM_MODEL_MATH", "qwen2.5:1.5b")
LLM_MODEL_PHYSICS = os.getenv("LLM_MODEL_PHYSICS", "llama3.2:1b")
LLM_MODEL_ENGLISH = os.getenv("LLM_MODEL_ENGLISH", "gemma:2b")
LLM_MODEL_GENERAL = os.getenv("LLM_MODEL_GENERAL", "qwen2.5:1.5b")

# Server
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8001"))

# VAD Settings
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
VAD_PADDING_MS = int(os.getenv("VAD_PADDING_MS", "600"))

# Sentence Buffer (for streaming)
SENTENCE_BUFFER_MIN_CHARS = int(os.getenv("SENTENCE_BUFFER_MIN_CHARS", "5"))
SENTENCE_BUFFER_MAX_CHARS = int(os.getenv("SENTENCE_BUFFER_MAX_CHARS", "50"))


def print_config():
    """Print current configuration for debugging"""
    print("=" * 50)
    print("Voice Agent Configuration")
    print("=" * 50)
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"PIPER_MODEL: {PIPER_MODEL}")
    print(f"WHISPER_MODEL: {WHISPER_MODEL}")
    print(f"LLM_MODEL_MATH: {LLM_MODEL_MATH}")
    print(f"LLM_MODEL_PHYSICS: {LLM_MODEL_PHYSICS}")
    print(f"LLM_MODEL_ENGLISH: {LLM_MODEL_ENGLISH}")
    print(f"SERVER: {SERVER_HOST}:{SERVER_PORT}")
    print("=" * 50)
