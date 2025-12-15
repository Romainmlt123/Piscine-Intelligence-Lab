import os
import sys
import subprocess
from pathlib import Path

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PIPER_DIR

class TTSModule:
    def __init__(self, model_path=None, piper_binary=None):
        self.model_path = model_path or str(PIPER_DIR / "fr_FR-upmc-medium.onnx")
        self.piper_binary = piper_binary or str(PIPER_DIR / "piper" / "piper")
        print(f"TTS Module initialized (Piper: {self.model_path})")

    def generate_audio(self, text, output_file="output.wav"):
        # print(f"Generating audio for: {text}") # Too verbose for streaming
        try:
            # Piper expects text via stdin
            command = f"{self.piper_binary} --model {self.model_path} --output_file {output_file}"
            process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(input=text.encode('utf-8'))
            
            if process.returncode != 0:
                print(f"Piper Error: {stderr.decode()}")
            else:
                # print(f"Audio saved to {output_file}")
                pass
                
        except Exception as e:
            print(f"TTS Error: {e}")
            # Fallback to empty wav
            with open(output_file, "wb") as f:
                f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
        return output_file
