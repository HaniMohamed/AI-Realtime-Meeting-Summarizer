#!/usr/bin/env python3
"""
meeting_companion_macos.py
MVP: record from an aggregate macOS device (BlackHole + mic), transcribe chunks using faster-whisper,
do simple speaker attribution (calibrated), and summarize via Ollama CLI periodically.

Usage:
    python meeting_companion_macos.py --device-id <n>   # or run without and pick device
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from datetime import timedelta
from pathlib import Path


import numpy as np
import sounddevice as sd
import soundfile as sf

# faster-whisper
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

# -------------------------
# Calibration: detect user's mic channel
# -------------------------
def calibrate_mic(device_id: int, samplerate: int, channels: int, duration: float = 3.0) -> int:
    """
    Ask user to speak for `duration` seconds (others quiet). Return the channel index that corresponds to the mic
    (simple RMS-based choice).
    """
    print("\n=== Calibration step ===")
    print(f"Please be the ONLY one speaking for the next {duration:.1f}s (say a short sentence).")
    input("Press Enter to start calibration...")

    frames = int(duration * samplerate)
    print("Recording calibration...")
    audio = sd.rec(frames, samplerate=samplerate, channels=channels, dtype='float32', device=device_id)
    sd.wait()
    print("Calibration done. Computing channel RMS...")

    # audio shape: (frames, channels) or (frames,) if mono
    if audio.ndim == 1:
        print("Device appears mono; no speaker channel differentiation possible.")
        return 0

    rms = np.sqrt(np.mean(np.square(audio), axis=0) + 1e-12)
    mic_channel = int(np.argmax(rms))
    for ch, val in enumerate(rms):
        print(f"  channel {ch}: RMS = {val:.6f}")
    print(f"Detected mic channel index = {mic_channel}")
    return mic_channel

# -------------------------
# Transcribe a chunk file with faster-whisper
# -------------------------
def transcribe_file_whisper(model, wav_path):
    """
    Return a list of segments with fields: start, end, text
    """
    segments, _ = model.transcribe(wav_path, beam_size=5, language="en")
    results = []
    for seg in segments:
        # seg.start, seg.end, seg.text
        results.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
    return results

# -------------------------
# Call Ollama CLI to summarize
# -------------------------
def ollama_summarize(ollama_model: str, prompt_text: str, max_tokens: int = 512) -> str:
    """
    Uses the installed 'ollama' binary. Make sure it's on PATH and the model exists locally.
    Returns the model output (string).
    """
    if not prompt_text.strip():
        return ""

    print("Calling Ollama to summarize (this may take a few seconds)...")
    cmd = ["ollama", "run", ollama_model, prompt_text]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, text=True, timeout=120)
        return proc.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("Ollama CLI failed:", e, e.stdout, e.stderr)
        return ""
    except Exception as e:
        print("Ollama call error:", e)
        return ""
    
# -------------------------
# Main recording + transcribe loop
# -------------------------
def run_meeting(device_id: int,
                chunk_duration: float,
                summary_interval: float,
                whisper_model_name: str,
                ollama_model: str,
                use_gpu: bool,
                home_folder_path: str):

    # Device info
    info = sd.query_devices(device_id)
    samplerate = int(info['default_samplerate'])
    channels = int(info['max_input_channels'])
    print(f"Using device [{device_id}] {info['name']}  samplerate={samplerate}  channels={channels}")

    if channels <= 0:
        raise RuntimeError("Selected device has no input channels.")

    # Load Whisper model once
    device_str = "cuda" if use_gpu else "cpu"
    print(f"Loading Whisper model '{whisper_model_name}' on {device_str} ...")
    model = WhisperModel(whisper_model_name, device=device_str)

    # Calibrate mic channel
    mic_channel = calibrate_mic(device_id, samplerate, channels, duration=3.0)

    # Prepare outputs
    meeting_start = time.time()
    transcript_list = []  # list of segments with absolute timestamps
    batch_texts = []  # for summarization
    last_summary_time = time.time()
    readable_time = time.strftime("%Y_%m_%d_%H:%M", time.localtime(meeting_start))

    folder_path = create_results_folder(home_folder_path, readable_time)

    out_json_path = f"{folder_path}/meeting_transcript_{int(meeting_start)}.json"
    summary_path = f"{folder_path}/meeting_summary_{int(meeting_start)}.txt"

    print("\n=== Starting recording loop ===")
    print("Press Ctrl+C to stop and create final summary.\n")

    chunk_index = 0
    try:
        while True:
            chunk_index += 1
            chunk_start = time.time()
            frames = int(chunk_duration * samplerate)
            # record chunk synchronously
            audio = sd.rec(frames, samplerate=samplerate, channels=channels, dtype='float32', device=device_id)
            sd.wait()

            # save to temp wav
            tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmpf.name
            tmpf.close()
            sf.write(tmp_path, audio, samplerate)
            print(f"[chunk {chunk_index}] Recorded {chunk_duration:.1f}s -> {tmp_path}")

            # transcribe
            segments = transcribe_file_whisper(model, tmp_path)
            os.unlink(tmp_path)

            # Determine simple speaker attribution using RMS per-channel
            if audio.ndim == 1:
                channel_rms = [np.sqrt(np.mean(audio**2) + 1e-12)]
            else:
                channel_rms = list(np.sqrt(np.mean(np.square(audio), axis=0) + 1e-12))

            other_rms = max([val for idx,val in enumerate(channel_rms) if idx != mic_channel] + [0.0])
            mic_rms = channel_rms[mic_channel] if mic_channel < len(channel_rms) else 0.0
            # heuristic threshold
            speaker_for_chunk = "Unknown"
            if mic_rms > other_rms * 1.5 and mic_rms > 1e-4:
                speaker_for_chunk = "You"
            elif other_rms > mic_rms * 1.2 and other_rms > 1e-4:
                speaker_for_chunk = "Other"
            else:
                speaker_for_chunk = "Unknown"

            # append segments with absolute timestamps
            chunk_elapsed = chunk_start - meeting_start
            for seg in segments:
                abs_start = seg['start'] + chunk_elapsed
                abs_end = seg['end'] + chunk_elapsed
                entry = {
                    "start_s": abs_start,
                    "end_s": abs_end,
                    "time": pretty_time(abs_start),
                    "duration_s": seg['end'] - seg['start'],
                    "text": seg['text'],
                    "speaker": speaker_for_chunk
                }
                transcript_list.append(entry)
                batch_texts.append(seg['text'])
                # print live caption
                print(f"[{pretty_time(abs_start)} - {speaker_for_chunk}] {seg['text']}")

            # Periodic summarization
            now = time.time()
            if (now - last_summary_time) >= summary_interval and batch_texts:
                joined = "\n".join(batch_texts[-50:])  # last N pieces (avoid too long prompt)
                prompt = (
                    "You are an assistant. Please produce a succinct meeting summary with:\n"
                    "- Key points\n- Decisions\n- Action items (bullet list)\n\n"
                    f"Transcript excerpt:\n{joined}\n\n"
                    "Output: Key Points, Decisions, Action Items."
                )
                summary = ollama_summarize(ollama_model, prompt)
                if summary:
                    print("\n--- Interim Summary ---")
                    print(summary)
                    with open(summary_path, "a", encoding="utf-8") as fh:
                        fh.write(f"\n--- Interim Summary at {time.ctime()} ---\n")
                        fh.write(summary + "\n")
                last_summary_time = now

    except KeyboardInterrupt:
        print("\nRecording stopped by user. Creating final summary & saving files...")
    finally:
        # final summarization using entire meeting transcript
        all_text = "\n".join([t['text'] for t in transcript_list])
        final_prompt = (
            "You are an assistant. Produce a final meeting summary with:\n"
            "- Short summary (3-20 lines)\n- Key points\n- Decisions\n- Action items with owners if mentioned\n\n"
            f"Full transcript:\n{all_text}\n\nOutput structured sections."
        )
        final_summary = ollama_summarize(ollama_model, final_prompt)
        if final_summary:
            print("\n=== Final Summary ===\n")
            print(final_summary)
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write(f"\n=== Final Summary at {time.ctime()} ===\n")
                fh.write(final_summary + "\n")
        # save transcript JSON
        with open(out_json_path, "w", encoding="utf-8") as fh:
            json.dump({"meeting_start": meeting_start, "segments": transcript_list}, fh, ensure_ascii=False, indent=2)
        print(f"Saved transcript -> {out_json_path}")
        print(f"Saved summary -> {summary_path}")
        print("Done.")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Meeting Companion MVP (macOS) - record + whisper + ollama")
    parser.add_argument("--device-id", type=int, default=None, help="sounddevice device id (input). If omitted you will be prompted.")
    parser.add_argument("--chunk-duration", type=float, default=5.0, help="seconds per audio chunk to transcribe")
    parser.add_argument("--summary-interval", type=float, default=60.0, help="seconds between interim summaries to Ollama")
    parser.add_argument("--whisper-model", type=str, default="small", help="faster-whisper model name (small/medium/large...)")
    parser.add_argument("--ollama-model", type=str, default="llama3.1:latest", help="Ollama model name installed locally")
    parser.add_argument("--use-gpu", action="store_true", help="run whisper on GPU if available")
    args = parser.parse_args()

    if args.device_id is None:
        list_input_devices()
        selection = input("\nEnter the device index to use (e.g. 3): ").strip()
        args.device_id = int(selection)

    home = Path.home()
    home_folder_path = create_results_folder(home/"Desktop","Teams_AI_Summarize")

    run_meeting(device_id=args.device_id,
                chunk_duration=args.chunk_duration,
                summary_interval=args.summary_interval,
                whisper_model_name=args.whisper_model,
                ollama_model=args.ollama_model,
                use_gpu=args.use_gpu,
                home_folder_path=home_folder_path)



def create_results_folder(folder_path: str,
                folder_name: str,):
    # Get desktop path
    new_folder = Path(folder_path) / folder_name
    new_folder.mkdir(exist_ok=True)

    print(f"Folder created at: {new_folder}")
    return new_folder


if __name__ == "__main__":
    main()
