# prep.py
import sys, os, json, subprocess, time
from pathlib import Path
import numpy as np, librosa
from faster_whisper import WhisperModel
from google import genai
from google.genai import types

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
DST.mkdir(parents=True, exist_ok=True)
TARGET, SNAP, MIN_TAIL = 42.0, 3.0, 20.0     # seconds
GEMINI_MODEL = "gemini-2.5-flash"             # change if deprecated
EXTS = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a"}

whisper = WhisperModel("large-v3", device="cuda", compute_type="float16")  # CPU: device="cpu", compute_type="int8"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
NOTES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def find_cuts(y, sr):
    dur = len(y) / sr
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    cuts, pos = [0.0], 0.0
    while dur - pos > TARGET + MIN_TAIL:
        c = pos + TARGET
        m = (t > c - SNAP) & (t < c + SNAP)
        c = t[m][np.argmin(rms[m])]
        cuts.append(float(c)); pos = c
    cuts.append(dur)
    return cuts

def bpm_key(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(float(np.atleast_1d(tempo)[0])))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    best, name = -2, "C major"
    for i in range(12):
        for prof, mode in ((MAJ, "major"), (MIN, "minor")):
            s = np.corrcoef(chroma, np.roll(prof, i))[0, 1]
            if s > best: best, name = s, f"{NOTES[i]} {mode}"
    return bpm, name

def caption(path, genre=""):
    prompt = (f"This clip is from a Hindi {genre} song. "
              "Write a 1-2 sentence English caption describing this music clip: genre/style, "
              "instruments, mood, tempo feel, vocal type (gender, style). "
              "Do not transcribe lyrics. Output only the caption.")
    part = types.Part.from_bytes(data=Path(path).read_bytes(), mime_type="audio/mpeg")
    for attempt in range(4):
        try:
            return client.models.generate_content(model=GEMINI_MODEL, contents=[part, prompt]).text.strip()
        except Exception as e:
            print("gemini error:", e)
            time.sleep(5 * (attempt + 1))
    return ""

for song in sorted(p for p in SRC.rglob("*") if p.is_file() and p.suffix.lower() in EXTS):
    stem = song.stem
    out = DST / song.parent.relative_to(SRC)
    out.mkdir(parents=True, exist_ok=True)
    genre = song.parent.name.replace("_", " ")
    if (out / f"{stem}_000.json").exists():
        print("skip", stem); continue
    print("processing", genre, "/", stem)
    y, sr = librosa.load(song, sr=22050, mono=True)
    cuts = find_cuts(y, sr)

    segs, _ = whisper.transcribe(str(song), language="hi", word_timestamps=True,
                                 condition_on_previous_text=False, vad_filter=False)
    words = [w for s in segs for w in s.words]

    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        base = out / f"{stem}_{i:03d}"
        mp3 = Path(f"{base}.mp3")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
                        "-i", str(song), "-ar", "44100", "-b:a", "192k", str(mp3)], check=True)

        # lyrics: words whose start falls in [a, b); new line on gaps > 0.8s
        lines, cur, last = [], [], None
        for w in words:
            if a <= w.start < b:
                if last is not None and w.start - last > 0.8 and cur:
                    lines.append(" ".join(cur)); cur = []
                cur.append(w.word.strip()); last = w.end
        if cur: lines.append(" ".join(cur))
        lyrics = "\n".join(lines) if lines else "[instrumental]"
        Path(f"{base}.lyrics.txt").write_text(lyrics, encoding="utf-8")

        seg = y[int(a * sr):int(b * sr)]
        bpm, key = bpm_key(seg, sr)
        cap = caption(mp3, genre)
        Path(f"{base}.caption.txt").write_text(cap, encoding="utf-8")
        with open(f"{base}.json", "w", encoding="utf-8") as f:
            json.dump({"caption": cap, "bpm": bpm, "keyscale": key,
                       "timesignature": "4", "language": "hi"},
                      f, ensure_ascii=False, indent=2)