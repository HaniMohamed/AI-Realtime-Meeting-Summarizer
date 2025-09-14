# meeting_companion/transcriber.py

import os
import numpy as np
from utils import pretty_time

def transcribe_file_whisper(model, wav_path):
    segments, _ = model.transcribe(wav_path, beam_size=5, language="en")
    return [{"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()} for seg in segments]

def transcriber(audio_queue, transcript_queue, model, mic_channel, meeting_start):
    while True:
        wav_path, chunk_start, audio = audio_queue.get()
        try:
            segments = transcribe_file_whisper(model, wav_path)
        finally:
            os.unlink(wav_path)

        # Speaker attribution
        if audio.ndim == 1:
            channel_rms = [np.sqrt(np.mean(audio**2) + 1e-12)]
        else:
            channel_rms = list(np.sqrt(np.mean(np.square(audio), axis=0) + 1e-12))
        other_rms = max([val for idx,val in enumerate(channel_rms) if idx != mic_channel] + [0.0])
        mic_rms = channel_rms[mic_channel] if mic_channel < len(channel_rms) else 0.0

        if mic_rms > other_rms * 1.5 and mic_rms > 1e-4:
            speaker = "You"
        elif other_rms > mic_rms * 1.2 and other_rms > 1e-4:
            speaker = "Other"
        else:
            speaker = "Unknown"

        chunk_elapsed = chunk_start - meeting_start
        for seg in segments:
            abs_start = seg['start'] + chunk_elapsed
            entry = {
                "start_s": abs_start,
                "end_s": seg['end'] + chunk_elapsed,
                "time": pretty_time(abs_start),
                "duration_s": seg['end'] - seg['start'],
                "text": seg['text'],
                "speaker": speaker
            }
            transcript_queue.put(entry)
            print(f"[{entry['time']} - {speaker}] {entry['text']}")

        audio_queue.task_done()
