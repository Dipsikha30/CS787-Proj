# fetch_jamendo.py
import requests, sys
from pathlib import Path

CID, TAG, OUT = sys.argv[1], sys.argv[2], Path("songs")
OUT.mkdir(exist_ok=True)
total, offset = 0, 0
while total < 5 * 3600:
    r = requests.get("https://api.jamendo.com/v3.0/tracks/", params={
        "client_id": CID, "format": "json", "limit": 50, "offset": offset,
        "tags": TAG, "durationbetween": "120_420",
        "audiodownload_allowed": "true"}).json()["results"]
    if not r: break
    for t in r:
        if total >= 5 * 3600: break
        f = OUT / f"{t['id']}.mp3"
        if not f.exists():
            f.write_bytes(requests.get(t["audiodownload"]).content)
        (OUT / f"{t['id']}.license.txt").write_text(t.get("license_ccurl", ""))
        total += t["duration"]
    offset += 50
print(f"{total/3600:.1f} hours downloaded")