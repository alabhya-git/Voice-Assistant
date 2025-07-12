import sounddevice as sd
import queue
import sys
import os
from vosk import Model, KaldiRecognizer
import json
from config import VOSK_MODEL_PATH, SAMPLE_RATE

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print("⚠️", status, file=sys.stderr)
    q.put(bytes(indata))

def transcribe_audio():
    if not os.path.exists(VOSK_MODEL_PATH):
        raise FileNotFoundError(f"Vosk model not found at {VOSK_MODEL_PATH}")

    print("🎙️ Listening... Speak into your mic.")

    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                return result.get("text", "")
