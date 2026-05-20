import wave
from pathlib import Path
import csv

audio_dir = Path("/mnt/d/workdir/data/fijian/audio")   # 改成你的音频目录

rows = []
for wav_file in sorted(audio_dir.glob("*.wav")):
    try:
        with wave.open(str(wav_file), "rb") as wf:
            rows.append({
                "file": wav_file.name,
                "channels": wf.getnchannels(),
                "sample_rate": wf.getframerate(),
                "sample_width": wf.getsampwidth(),
                "n_frames": wf.getnframes(),
            })
    except Exception as e:
        rows.append({
            "file": wav_file.name,
            "channels": "ERROR",
            "sample_rate": str(e),
            "sample_width": "",
            "n_frames": "",
        })

with open("wav_info.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "channels", "sample_rate", "sample_width", "n_frames"])
    writer.writeheader()
    writer.writerows(rows)

print("已导出 wav_info.csv")