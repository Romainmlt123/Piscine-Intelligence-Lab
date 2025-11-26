import webrtcvad
import collections
import sys

class VADManager:
    def __init__(self, sample_rate=16000, frame_duration_ms=30, padding_duration_ms=300):
        self.vad = webrtcvad.Vad(3) # Aggressiveness mode 3 (High)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000 * 2) # 2 bytes per sample (16-bit)
        
        self.buffer = b""
        self.ring_buffer = collections.deque(maxlen=int(padding_duration_ms / frame_duration_ms))
        self.triggered = False
        self.voiced_frames = []
        
    def process_chunk(self, chunk):
        """
        Process a raw PCM chunk.
        Returns:
            - None if speech is continuing or silence (no action needed)
            - bytes (Audio Data) if a speech segment is completed
        """
        self.buffer += chunk
        
        while len(self.buffer) >= self.frame_size:
            frame = self.buffer[:self.frame_size]
            self.buffer = self.buffer[self.frame_size:]
            
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            
            if not self.triggered:
                self.ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in self.ring_buffer if speech])
                if num_voiced > 0.9 * self.ring_buffer.maxlen:
                    self.triggered = True
                    # Start of speech detected
                    # Include ring buffer in voiced frames
                    self.voiced_frames.extend([f for f, s in self.ring_buffer])
                    self.ring_buffer.clear()
            else:
                self.voiced_frames.append(frame)
                self.ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
                if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                    self.triggered = False
                    # End of speech detected (Silence)
                    audio_data = b"".join(self.voiced_frames)
                    self.voiced_frames = []
                    self.ring_buffer.clear()
                    return audio_data
                    
        return None
