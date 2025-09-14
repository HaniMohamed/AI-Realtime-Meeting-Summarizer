#!/usr/bin/env python3
"""
meeting_companion_macos_multithread.py
- Records continuously from macOS aggregate device (BlackHole + mic).
- Uses faster-whisper for transcription.
- Summarizes periodically with Ollama in a background thread.
- Uses threads + queues so recording never stops while processing.
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
import threading
import queue
from datetime import timedelta
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

# -------------------------
# Utility helpers
# -------------------------
def pretty_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

def list_input_devices():
    print("Available input devices (from sounddevice):")
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            print(f"[{idx}] {d['name']}  channels={d['max_input_channels']}  default_sr={d['default_samplerate']}")
    return devices

def create_results_folder(folder_path: str, folder_name: str):
    new_folder = Path(folder_path) / folder_name
    new_folder.mkdir(exist_ok=True)
    print(f"Folder created at: {new_folder}")
    return new_folder

# -------------------------
# Calibration
# -------------------------
def calibrate_mic(device_id: int, samplerate: int, channels: int, duration: float = 3.0) -> int:
    print("\n=== Calibration step ===")
    print(f"Please be the ONLY one speaking for {duration:.1f}s")
    input("Press Enter to start calibration...")

    frames = int(duration * samplerate)
    audio = sd.rec(frames, samplerate=samplerate, channels=channels,
                   dtype='float32', device=device_id)
    sd.wait()

    if audio.ndim == 1:
        print("Mono device, no differentiation possible.")
        return 0

    rms = np.sqrt(np.mean(np.square(audio), axis=0) + 1e-12)
    mic_channel = int(np.argmax(rms))
    for ch, val in enumerate(rms):
        print(f"  channel {ch}: RMS = {val:.6f}")
    print(f"Detected mic channel index = {mic_channel}")
    return mic_channel

# -------------------------
# Whisper transcription
# -------------------------
def transcribe_file_whisper(model, wav_path):
    segments, _ = model.transcribe(wav_path, beam_size=5, language="en")
    results = []
    for seg in segments:
        results.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip()
        })
    return results

# -------------------------
# Ollama summarization
# -------------------------
def ollama_summarize(ollama_model: str, prompt_text: str) -> str:
    if not prompt_text.strip():
        return ""
    cmd = ["ollama", "run", ollama_model, prompt_text]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, text=True, timeout=120)
        return proc.stdout.strip()
    except Exception as e:
        print("Ollama call failed:", e)
        return ""

# -------------------------
# Threads
# -------------------------
def recorder(audio_queue, device_id, samplerate, channels, chunk_duration):
    frames = int(chunk_duration * samplerate)
    chunk_index = 0
    while True:
        chunk_index += 1
        audio = sd.rec(frames, samplerate=samplerate,
                       channels=channels, dtype='float32', device=device_id)
        sd.wait()

        tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmpf.name, audio, samplerate)
        audio_queue.put((tmpf.name, time.time(), audio))
        print(f"[recorder] queued chunk {chunk_index}")

def transcriber(audio_queue, transcript_queue, model, mic_channel, meeting_start):
    while True:
        wav_path, chunk_start, audio = audio_queue.get()
        try:
            segments = transcribe_file_whisper(model, wav_path)
        finally:
            os.unlink(wav_path)

        # speaker attribution
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

def summarizer(transcript_queue, summary_path, ollama_model, summary_interval):
    buffer = []
    last_time = time.time()
    while True:
        try:
            entry = transcript_queue.get(timeout=1)
            buffer.append(entry["text"])
            transcript_queue.task_done()
        except queue.Empty:
            pass

        now = time.time()
        if now - last_time >= summary_interval and buffer:
            joined = "\n".join(buffer[-50:])
            prompt = (
                "You are an assistant. Please produce a succinct meeting summary with:\n"
                "- Key points\n- Decisions\n- Action items\n\n"
                f"Transcript excerpt:\n{joined}\n"
            )
            summary = ollama_summarize(ollama_model, prompt)
            if summary:
                print("\n--- Interim Summary ---\n", summary)
                with open(summary_path, "a", encoding="utf-8") as fh:
                    fh.write(f"\n--- Interim Summary at {time.ctime()} ---\n")
                    fh.write(summary + "\n")
            last_time = now

# -------------------------
# Main
# -------------------------
def run_meeting(device_id, chunk_duration, summary_interval, whisper_model_name,
                ollama_model, use_gpu, home_folder_path):

    info = sd.query_devices(device_id)
    samplerate = int(info['default_samplerate'])
    channels = int(info['max_input_channels'])
    print(f"Using device [{device_id}] {info['name']} samplerate={samplerate} channels={channels}")

    device_str = "cuda" if use_gpu else "cpu"
    print(f"Loading Whisper model '{whisper_model_name}' on {device_str} ...")
    model = WhisperModel(whisper_model_name, device=device_str)

    mic_channel = calibrate_mic(device_id, samplerate, channels, duration=3.0)

    meeting_start = time.time()
    readable_time = time.strftime("%Y_%m_%d_%H:%M", time.localtime(meeting_start))
    folder_path = create_results_folder(home_folder_path, readable_time)

    summary_path = f"{folder_path}/meeting_summary_{int(meeting_start)}.txt"
    out_json_path = f"{folder_path}/meeting_transcript_{int(meeting_start)}.json"

    audio_queue = queue.Queue()
    transcript_queue = queue.Queue()
    transcript_list = []

    print("\n=== Starting threads ===\n")
    threading.Thread(target=recorder, args=(audio_queue, device_id, samplerate, channels, chunk_duration), daemon=True).start()
    threading.Thread(target=transcriber, args=(audio_queue, transcript_queue, model, mic_channel, meeting_start), daemon=True).start()
    threading.Thread(target=summarizer, args=(transcript_queue, summary_path, ollama_model, summary_interval), daemon=True).start()

    try:
        while True:
            # Collect transcripts for saving later
            try:
                entry = transcript_queue.get(timeout=1)
                transcript_list.append(entry)
                transcript_queue.task_done()
            except queue.Empty:
                pass
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping... Saving final transcript.")
        with open(out_json_path, "w", encoding="utf-8") as fh:
            json.dump({"meeting_start": meeting_start, "segments": transcript_list},
                      fh, ensure_ascii=False, indent=2)
        print(f"Transcript saved to {out_json_path}")
        print(f"Summaries in {summary_path}")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--chunk-duration", type=float, default=5.0)
    parser.add_argument("--summary-interval", type=float, default=60.0)
    parser.add_argument("--whisper-model", type=str, default="small")
    parser.add_argument("--ollama-model", type=str, default="llama3.1:latest")
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    if args.device_id is None:
        list_input_devices()
        args.device_id = int(input("Enter device index: ").strip())

    home = Path.home()
    home_folder_path = create_results_folder(home/"Desktop","Teams_AI_Summarize")

    run_meeting(args.device_id, args.chunk_duration, args.summary_interval,
                args.whisper_model, args.ollama_model, args.use_gpu, home_folder_path)

if __name__ == "__main__":
    main()
