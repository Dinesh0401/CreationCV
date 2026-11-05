import os
import sys
import time
import requests
from pathlib import Path

# Find snapshot path
cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--h94--IP-Adapter" / "snapshots"
snapshots = list(cache_dir.glob("*"))
if not snapshots:
    print("Snapshot dir not found!")
    sys.exit(1)

target_dir = snapshots[0] / "models" / "image_encoder"
target_dir.mkdir(parents=True, exist_ok=True)
target_file = target_dir / "model.safetensors"
part_file = target_dir / "model.safetensors.part"

url = "https://hf-mirror.com/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"
print(f"Direct downloading {url} to {target_file}...", flush=True)

headers = {}
downloaded_bytes = 0
if part_file.exists():
    downloaded_bytes = part_file.stat().st_size
    headers["Range"] = f"bytes={downloaded_bytes}-"
    print(f"Resuming from byte offset {downloaded_bytes / 1024**2:.1f} MB...", flush=True)

resp = requests.get(url, headers=headers, stream=True, timeout=30)
if resp.status_code not in (200, 206):
    print(f"Error: HTTP {resp.status_code}")
    sys.exit(1)

total_length = int(resp.headers.get("content-length", 0)) + downloaded_bytes
print(f"Total size: {total_length / 1024**2:.1f} MB", flush=True)

mode = "ab" if downloaded_bytes > 0 else "wb"
t0 = time.time()
last_report = t0

with open(part_file, mode) as f:
    for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):  # 4 MB chunks
        if chunk:
            f.write(chunk)
            downloaded_bytes += len(chunk)
            t_now = time.time()
            if t_now - last_report >= 5.0:
                speed_mb = (downloaded_bytes / 1024**2) / max(0.1, t_now - t0)
                pct = (downloaded_bytes / total_length) * 100 if total_length else 0
                print(f"Downloaded: {downloaded_bytes / 1024**2:.1f} MB / {total_length / 1024**2:.1f} MB ({pct:.1f}%) [{speed_mb:.1f} MB/s]", flush=True)
                last_report = t_now

print("Download complete! Finalizing file...", flush=True)
if target_file.exists():
    target_file.unlink()
part_file.rename(target_file)
print(f"Saved to {target_file} ({target_file.stat().st_size / 1024**2:.1f} MB)!", flush=True)

