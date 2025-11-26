import whisper
import numpy as np
import torch

class STTModule:
    def __init__(self, model_size="base"):
        print(f"Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size)
        print("Whisper model loaded.")

    def transcribe(self, audio_data):
        # audio_data expected to be numpy array or path
        # If raw bytes, need conversion. For now assuming path or prepared array.
        try:
            result = self.model.transcribe(audio_data)
            return result["text"]
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
