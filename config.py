# meeting_companion/config.py

from pathlib import Path

# Defaults
DEFAULT_CHUNK_DURATION = 5.0
DEFAULT_SUMMARY_INTERVAL = 60.0
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_OLLAMA_MODEL = "llama3.1:latest"

# Save location (Desktop/Teams_AI_Summarize)
HOME_FOLDER = Path.home()
OUTPUT_FOLDER = HOME_FOLDER / "Desktop" / "Teams_AI_Summarize"
