"""
Audio Streaming Manager for Sequential TTS Playback

This module ensures:
1. Text chunks are converted to audio sequentially (no parallelism)
2. Each audio chunk is indexed for ordered playback
3. Thread-safe queues for producer-consumer pattern
4. Mathematical expressions are converted to spoken French
"""

import threading
import queue
import time
import os
from dataclasses import dataclass
from typing import Optional, Generator, Tuple

# Import math-to-speech converter from same package
from speech.math_to_speech import convert_math_to_speech


@dataclass
class AudioChunk:
    """Represents a generated audio chunk with metadata"""
    index: int
    audio_bytes: bytes
    text: str
    duration_ms: int = 0


class AudioStreamManager:
    """
    Manages TTS generation queue and sequential audio delivery.
    
    Architecture:
    - Text chunks are added via add_text()
    - A worker thread converts them to audio sequentially
    - Math expressions are converted to spoken French before TTS
    - Audio chunks can be retrieved via get_audio() or iter_audio()
    - Each chunk has an index for ordered playback on the client
    """
    
    def __init__(self, tts_module):
        self.tts = tts_module
        self.text_queue: queue.Queue = queue.Queue()
        self.audio_queue: queue.Queue = queue.Queue()
        self.chunk_index = 0
        self.running = False
        self.tts_thread: Optional[threading.Thread] = None
        self.generation_complete = threading.Event()
        self._lock = threading.Lock()
    
    def start(self):
        """Start the TTS worker thread"""
        with self._lock:
            if self.running:
                return
            
            self.running = True
            self.chunk_index = 0
            self.generation_complete.clear()
            
            # Clear queues
            while not self.text_queue.empty():
                try:
                    self.text_queue.get_nowait()
                except queue.Empty:
                    break
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
    
    def stop(self):
        """Stop the TTS worker thread"""
        with self._lock:
            if not self.running:
                return
            self.running = False
        
        # Send poison pill to unblock the worker
        self.text_queue.put(None)
        
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2.0)
    
    def add_text(self, text: str):
        """Add a text chunk to be converted to audio"""
        if text and text.strip():
            self.text_queue.put(text.strip())
    
    def finish_generation(self):
        """Signal that no more text will be added"""
        self.text_queue.put(None)  # Poison pill
    
    def get_audio(self, timeout: float = 0.1) -> Optional[AudioChunk]:
        """Get next audio chunk if available (non-blocking with timeout)"""
        try:
            chunk = self.audio_queue.get(timeout=timeout)
            return chunk
        except queue.Empty:
            return None
    
    def iter_audio(self) -> Generator[AudioChunk, None, None]:
        """
        Iterate over audio chunks as they become available.
        Blocks until all chunks are generated.
        """
        while True:
            # Check if we're done
            if self.generation_complete.is_set() and self.audio_queue.empty():
                break
            
            chunk = self.get_audio(timeout=0.2)
            if chunk is not None:
                yield chunk
    
    def _tts_worker(self):
        """Worker thread that processes text -> audio sequentially"""
        while self.running:
            try:
                text = self.text_queue.get(timeout=0.5)
                
                if text is None:  # Poison pill
                    self.generation_complete.set()
                    break
                
                # ===== MATH-TO-SPEECH CONVERSION =====
                # Convert mathematical notation to spoken French
                spoken_text = convert_math_to_speech(text)
                
                # Generate audio (blocking, sequential - this is the key!)
                audio_path = f"chunk_{self.chunk_index}_{int(time.time()*1000)}.wav"
                
                try:
                    # Use the converted spoken text for TTS
                    self.tts.generate_audio(spoken_text, output_file=audio_path)
                    
                    # Read the audio file
                    if os.path.exists(audio_path):
                        with open(audio_path, "rb") as f:
                            audio_bytes = f.read()
                        
                        # Create chunk with metadata
                        chunk = AudioChunk(
                            index=self.chunk_index,
                            audio_bytes=audio_bytes,
                            text=text
                        )
                        self.audio_queue.put(chunk)
                        self.chunk_index += 1
                        
                        # Cleanup temp file
                        os.remove(audio_path)
                    
                except Exception as e:
                    print(f"TTS Worker Error: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS Worker Exception: {e}")
                continue
        
        self.generation_complete.set()


class SentenceBuffer:
    """
    Buffers streaming text until sentence boundaries are detected.
    
    This ensures TTS gets complete sentences for natural speech.
    Optimized for LOW LATENCY - sends first chunk quickly.
    """
    
    # Characters that mark end of a sentence or meaningful pause
    SENTENCE_DELIMITERS = {'.', '!', '?', '\n'}
    CLAUSE_DELIMITERS = {',', ';', ':', '—', '–'}
    
    def __init__(self, min_chars: int = 5, max_chars: int = 50):
        self.buffer = ""
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.first_chunk_sent = False
    
    def add(self, text: str) -> Optional[str]:
        """
        Add text to buffer, return complete sentence if available.
        
        Returns:
            Complete sentence if boundary detected, None otherwise
        """
        self.buffer += text
        
        # Check for sentence end
        for i, char in enumerate(self.buffer):
            if char in self.SENTENCE_DELIMITERS:
                # Found sentence end
                sentence = self.buffer[:i+1].strip()
                self.buffer = self.buffer[i+1:]
                
                # For first chunk, use lower threshold
                min_len = 5 if not self.first_chunk_sent else self.min_chars
                
                if len(sentence) >= min_len:
                    self.first_chunk_sent = True
                    return sentence
                elif sentence:
                    # Too short, keep buffering but prepend this
                    self.buffer = sentence + " " + self.buffer
                    return None
        
        # Check for clause delimiter if buffer is getting long
        if len(self.buffer) >= self.max_chars:
            for i, char in enumerate(self.buffer):
                if char in self.CLAUSE_DELIMITERS and i >= self.min_chars:
                    sentence = self.buffer[:i+1].strip()
                    self.buffer = self.buffer[i+1:]
                    self.first_chunk_sent = True
                    return sentence
            
            # No delimiter found, force split at word boundary
            words = self.buffer.split()
            if len(words) > 3:
                # Take first half of words
                split_point = len(words) // 2
                sentence = " ".join(words[:split_point])
                self.buffer = " ".join(words[split_point:])
                self.first_chunk_sent = True
                return sentence
        
        return None
    
    def flush(self) -> Optional[str]:
        """Flush any remaining text in the buffer"""
        if self.buffer.strip():
            result = self.buffer.strip()
            self.buffer = ""
            return result
        return None
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = ""
        self.first_chunk_sent = False
