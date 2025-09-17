# meeting_companion/recorder.py

import tempfile
import sounddevice as sd
import soundfile as sf
import time

def recorder(audio_queue, device_id, samplerate, channels, chunk_duration, stop_event):
    frames = int(chunk_duration * samplerate)
    chunk_index = 0
    while True:
        chunk_index += 1
        audio = sd.rec(frames, samplerate=samplerate,
                       channels=channels, dtype='float32', device=device_id)
        sd.wait()

        tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmpf.name, audio, samplerate)
        if stop_event.is_set():
            break  # exit thread
        audio_queue.put((tmpf.name, time.time(), audio))
        print(f"[recorder] queued chunk {chunk_index}")

