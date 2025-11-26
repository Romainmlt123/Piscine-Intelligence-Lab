import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

from fastapi.responses import FileResponse

@app.get("/")
async def get():
    return FileResponse("static/index.html")

from vad_module import VADManager
from stt_module import STTModule
from llm_module import LLMModule
from tts_module import TTSModule
import tempfile
import os
import wave

# Initialize Modules
stt = STTModule(model_size="base")
llm = LLMModule(model="gemma:2b")
tts = TTSModule()

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    vad = VADManager()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            # logger.info(f"Received chunk: {len(data)} bytes") # Too verbose for streaming
            
            # Process chunk via VAD
            speech_segment = vad.process_chunk(data)
            
            if speech_segment:
                logger.info(f"Speech segment detected: {len(speech_segment)} bytes")
                
                # 1. Save PCM to WAV (Whisper needs WAV file or specific numpy format)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                    with wave.open(tmp_audio.name, 'wb') as wav_file:
                        wav_file.setnchannels(1) # Mono
                        wav_file.setsampwidth(2) # 16-bit
                        wav_file.setframerate(16000) # 16kHz
                        wav_file.writeframes(speech_segment)
                    tmp_audio_path = tmp_audio.name
                
                try:
                    # 2. STT
                    text = stt.transcribe(tmp_audio_path)
                    logger.info(f"Transcribed: {text}")
                    
                    if text.strip():
                        # Send User Text to Client
                        await websocket.send_json({"type": "user_text", "content": text})
                        
                        # 3. LLM
                        response_text = llm.generate_response(text)
                        logger.info(f"LLM Response: {response_text}")
                        
                        # Send AI Text to Client
                        await websocket.send_json({"type": "ai_text", "content": response_text})
                        
                        # 4. TTS
                        output_audio_path = tts.generate_audio(response_text)
                        
                        # 5. Send Audio back
                        with open(output_audio_path, "rb") as f:
                            audio_response = f.read()
                        await websocket.send_bytes(audio_response)
                        logger.info("Sent audio response")
                        
                        if os.path.exists(output_audio_path):
                            os.remove(output_audio_path)
                finally:
                    if os.path.exists(tmp_audio_path):
                        os.remove(tmp_audio_path)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error: {e}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
