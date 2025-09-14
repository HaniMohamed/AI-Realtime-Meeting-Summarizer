# meeting_companion/main.py

import argparse
import time
import threading
import queue
import json
from pathlib import Path
import sounddevice as sd
from faster_whisper import WhisperModel

import config
from utils import list_input_devices, create_results_folder, calibrate_mic
from recorder import recorder
from transcriber import transcriber
from summarizer import summarizer, finalSummarizer

def run_meeting(device_id, chunk_duration, summary_interval, whisper_model_name,
                ollama_model, use_gpu, output_folder):

    info = sd.query_devices(device_id)
    samplerate = int(info['default_samplerate'])
    channels = int(info['max_input_channels'])
    print(f"Using device [{device_id}] {info['name']} samplerate={samplerate} channels={channels}")

    device_str = "cuda" if use_gpu else "cpu"
    print(f"Loading Whisper model '{whisper_model_name}' on {device_str} ...")
    model = WhisperModel(whisper_model_name, device=device_str)

    mic_channel = calibrate_mic(device_id, samplerate, channels)

    meeting_start = time.time()
    readable_time = time.strftime("%Y_%m_%d_%H:%M", time.localtime(meeting_start))
    folder_path = create_results_folder(output_folder, readable_time)

    summary_path = f"{folder_path}/meeting_summary_{int(meeting_start)}.txt"
    out_json_path = f"{folder_path}/meeting_transcript_{int(meeting_start)}.json"

    audio_queue = queue.Queue()
    transcript_queue = queue.Queue()
    transcript_list = []

    # Threads
    threading.Thread(target=recorder, args=(audio_queue, device_id, samplerate, channels, chunk_duration), daemon=True).start()
    threading.Thread(target=transcriber, args=(audio_queue, transcript_queue, model, mic_channel, meeting_start), daemon=True).start()
    threading.Thread(target=summarizer, args=(transcript_queue, summary_path, ollama_model, summary_interval), daemon=True).start()

    try:
        while True:
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
            json.dump({"meeting_start": meeting_start, "segments": transcript_list}, fh, ensure_ascii=False, indent=2)
        print(f"Transcript saved to {out_json_path}")
        all_text = "\n".join([t['text'] for t in transcript_list])
        finalSummarizer(all_text, summary_path, ollama_model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--chunk-duration", type=float, default=config.DEFAULT_CHUNK_DURATION)
    parser.add_argument("--summary-interval", type=float, default=config.DEFAULT_SUMMARY_INTERVAL)
    parser.add_argument("--whisper-model", type=str, default=config.DEFAULT_WHISPER_MODEL)
    parser.add_argument("--ollama-model", type=str, default=config.DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    if args.device_id is None:
        list_input_devices()
        args.device_id = int(input("Enter device index: ").strip())

    output_folder = config.OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)

    run_meeting(args.device_id, args.chunk_duration, args.summary_interval,
                args.whisper_model, args.ollama_model, args.use_gpu, output_folder)

if __name__ == "__main__":
    main()
