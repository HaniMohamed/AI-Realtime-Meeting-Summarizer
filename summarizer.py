# meeting_companion/summarizer.py

import subprocess
import time
import queue

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

def summarizer(transcript_queue, summary_path, ollama_model, summary_interval, stop_event):
    buffer = []
    last_time = time.time()
    while True:
        try:
            entry = transcript_queue.get(timeout=1)
            buffer.append(entry["text"])
            transcript_queue.task_done()
            if stop_event.is_set():
                break  # exit thread
        except queue.Empty:
            pass

        now = time.time()
        if now - last_time >= summary_interval and buffer:
            joined = "\n".join(buffer[-50:])
            prompt = (
                "You are an assistant. Please produce a succinct meeting (mainly about software features and development) summary with:\n"
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

def finalSummarizer(all_text, summary_path, ollama_model):
    final_prompt = (
        "You are an assistant. Produce a final meeting (mainly about software features and development) summary with:\n"
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
    print(f"Saved summary -> {summary_path}")
    print("Done.")