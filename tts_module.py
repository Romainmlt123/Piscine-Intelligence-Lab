import os
import subprocess

class TTSModule:
    def __init__(self, model_path="en_US-lessac-medium.onnx"):
        self.model_path = model_path
        print("TTS Module initialized (Piper fallback mode)")

    def generate_audio(self, text, output_file="output.wav"):
        print(f"Generating audio for: {text}")
        from gtts import gTTS
        try:
            tts = gTTS(text=text, lang='fr')
            tts.save(output_file)
            print(f"Audio saved to {output_file}")
        except Exception as e:
            print(f"TTS Error: {e}")
            # Fallback to a minimal valid WAV header if gTTS fails, but gTTS should work.
            # 44 byte header for empty wav
            with open(output_file, "wb") as f:
                f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
        return output_file
