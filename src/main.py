import os
import sys
import asyncio
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import time
import tempfile
import wave

# Import from new package structure
from speech.vad_module import VADManager
from speech.stt_module import STTModule
from speech.tts_module import TTSModule
from speech.audio_streamer import AudioStreamManager, SentenceBuffer
from agents.orchestrator import AgentOrchestrator
from config import STATIC_DIR, SERVER_HOST, SERVER_PORT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Agent", version="1.4.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

from fastapi.responses import FileResponse

@app.get("/")
async def get():
    return FileResponse(str(STATIC_DIR / "index.html"))

# Initialize Modules
print("Starting Voice Agent Server...")
stt = STTModule(model_size="base")
tts = TTSModule()
orchestrator = AgentOrchestrator()


@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    vad = VADManager()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            speech_segment = vad.process_chunk(data)
            
            if speech_segment:
                logger.info(f"Speech segment detected: {len(speech_segment)} bytes")
                start_total = time.time()
                
                # 1. Save PCM to WAV
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                    with wave.open(tmp_audio.name, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(16000)
                        wav_file.writeframes(speech_segment)
                    tmp_audio_path = tmp_audio.name
                
                try:
                    # 2. STT
                    start_stt = time.time()
                    text = stt.transcribe(tmp_audio_path)
                    stt_duration = time.time() - start_stt
                    logger.info(f"Transcribed: {text} ({stt_duration:.2f}s)")
                    
                    if text.strip():
                        # Send user text to client
                        await websocket.send_json({"type": "user_text", "content": text})
                        
                        # Reset audio sequence for new response
                        await websocket.send_json({"type": "audio_reset"})
                        
                        # Initialize streaming components
                        audio_manager = AudioStreamManager(tts)
                        sentence_buffer = SentenceBuffer(min_chars=5, max_chars=50)
                        audio_manager.start()
                        
                        full_response_text = ""
                        first_audio_sent = False
                        ttfa = 0
                        agent_name = "Assistant"
                        model_name = ""
                        metrics = {}  # Initialize metrics
                        
                        try:
                            # Process LLM stream
                            iterator = orchestrator.process_stream(text)
                            
                            for event_type, event_data in iterator:
                                
                                if event_type == 'routing':
                                    # New format: {'agent': ..., 'model': ...}
                                    agent_name = event_data['agent']
                                    model_name = event_data['model']
                                    logger.info(f"Routing: {agent_name} -> Model: {model_name}")
                                    
                                elif event_type == 'rag':
                                    context = event_data['context']
                                    source_name = event_data['source']
                                    if context:
                                        await websocket.send_json({
                                            "type": "rag_sources", 
                                            "content": context, 
                                            "source": source_name,
                                            "agent": agent_name,
                                            "model": model_name
                                        })
                                        
                                elif event_type == 'llm_chunk':
                                    token = event_data
                                    full_response_text += token
                                    
                                    # Send text chunk to UI immediately
                                    await websocket.send_json({
                                        "type": "ai_text_chunk",
                                        "content": token,
                                        "agent": agent_name,
                                        "model": model_name
                                    })
                                    
                                    # Buffer text until sentence boundary
                                    sentence = sentence_buffer.add(token)
                                    if sentence:
                                        audio_manager.add_text(sentence)
                                    
                                    # Check for available audio and send it
                                    while True:
                                        chunk = audio_manager.get_audio(timeout=0.01)
                                        if chunk is None:
                                            break
                                        
                                        # Send audio with index for ordered playback
                                        await websocket.send_json({
                                            "type": "audio_chunk_meta",
                                            "index": chunk.index,
                                            "text": chunk.text[:50] + "..." if len(chunk.text) > 50 else chunk.text
                                        })
                                        await websocket.send_bytes(chunk.audio_bytes)
                                        
                                        if not first_audio_sent:
                                            ttfa = time.time() - start_total
                                            first_audio_sent = True
                                            logger.info(f"TTFA: {ttfa:.2f}s")
                                        
                                elif event_type == 'metrics':
                                    # Store metrics, will send at end
                                    metrics = event_data
                            
                            # Flush remaining text buffer
                            remaining = sentence_buffer.flush()
                            if remaining:
                                audio_manager.add_text(remaining)
                            
                            # Signal no more text coming
                            audio_manager.finish_generation()
                            
                            # Wait for and send remaining audio chunks
                            for chunk in audio_manager.iter_audio():
                                await websocket.send_json({
                                    "type": "audio_chunk_meta",
                                    "index": chunk.index,
                                    "text": chunk.text[:50] + "..." if len(chunk.text) > 50 else chunk.text
                                })
                                await websocket.send_bytes(chunk.audio_bytes)
                                
                                if not first_audio_sent:
                                    ttfa = time.time() - start_total
                                    first_audio_sent = True
                            
                            # Send final text
                            await websocket.send_json({
                                "type": "ai_text", 
                                "content": full_response_text, 
                                "agent": agent_name
                            })
                            
                            # Send metrics
                            metrics['stt'] = stt_duration
                            metrics['ttfa'] = ttfa
                            metrics['total'] = time.time() - start_total
                            await websocket.send_json({"type": "latency_metrics", "data": metrics})
                            
                            logger.info(f"Stream finished. TTFA: {ttfa:.2f}s, Total: {metrics['total']:.2f}s")
                            
                        finally:
                            audio_manager.stop()
                        
                finally:
                    if os.path.exists(tmp_audio_path):
                        os.remove(tmp_audio_path)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

