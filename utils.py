# meeting_companion/utils.py

import time
from datetime import timedelta
from pathlib import Path
import sounddevice as sd
import numpy as np

def pretty_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

def list_input_devices():
    print("Available input devices (from sounddevice):")
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            print(f"[{idx}] {d['name']}  channels={d['max_input_channels']}  default_sr={d['default_samplerate']}")
    return devices

def create_results_folder(base_folder: Path, folder_name: str):
    new_folder = Path(base_folder) / folder_name
    new_folder.mkdir(parents=True, exist_ok=True)
    print(f"Folder created at: {new_folder}")
    return new_folder

def calibrate_mic(device_id: int, samplerate: int, channels: int, duration: float = 3.0) -> int:
    print("\n=== Calibration step ===")
    print(f"Speak for {duration:.1f}s")
    input("Press Enter to start...")

    frames = int(duration * samplerate)
    audio = sd.rec(frames, samplerate=samplerate, channels=channels,
                   dtype='float32', device=device_id)
    sd.wait()

    if audio.ndim == 1:
        print("Mono device. Using channel 0.")
        return 0

    rms = np.sqrt(np.mean(np.square(audio), axis=0) + 1e-12)
    mic_channel = int(np.argmax(rms))
    for ch, val in enumerate(rms):
        print(f"  channel {ch}: RMS={val:.6f}")
    print(f"Detected mic channel index = {mic_channel}")
    return mic_channel
