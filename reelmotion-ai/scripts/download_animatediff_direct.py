import os
import sys
import time
import requests
from pathlib import Path

cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--guoyww--animatediff-motion-adapter-v1-5-2"
snapshot_dir = cache_dir / "snapshots" / "main"
snapshot_dir.mkdir(parents=True, exist_ok=True)

# 1. Download config.json
config_url = "https://hf-mirror.com/guoyww/animatediff-motion-adapter-v1-5-2/raw/main/config.json"
print("Downloading config.json...", flush=True)
r = requests.get(config_url, timeout=15)
(snapshot_dir / "config.json").write_bytes(r.content)
print("Saved config.json", flush=True)

# 2. Download diffusion_pytorch_model.fp16.safetensors
model_url = "https://hf-mirror.com/guoyww/animatediff-motion-adapter-v1-5-2/resolve/main/diffusion_pytorch_model.fp16.safetensors"
target_file = snapshot_dir / "diffusion_pytorch_model.fp16.safetensors"
part_file = snapshot_dir / "diffusion_pytorch_model.fp16.safetensors.part"

print(f"Downloading model from {model_url}...", flush=True)
downloaded = 0
headers = {}
if part_file.exists():
    downloaded = part_file.stat().st_size
    headers["Range"] = f"bytes={downloaded}-"
    print(f"Resuming from {downloaded / 1024**2:.1f} MB...", flush=True)

resp = requests.get(model_url, headers=headers, stream=True, timeout=30)
total_size = int(resp.headers.get("content-length", 0)) + downloaded
print(f"Total size: {total_size / 1024**2:.1f} MB", flush=True)

t0 = time.time()
last_t = t0
mode = "ab" if downloaded > 0 else "wb"
with open(part_file, mode) as f:
    for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            t_now = time.time()
            if t_now - last_t >= 4.0:
                pct = (downloaded / total_size) * 100
                speed = (downloaded / 1024**2) / max(0.1, t_now - t0)
                print(f"Downloaded: {downloaded / 1024**2:.1f} MB / {total_size / 1024**2:.1f} MB ({pct:.1f}%) [{speed:.1f} MB/s]", flush=True)
                last_t = t_now

if target_file.exists():
    target_file.unlink()
part_file.rename(target_file)
print(f"AnimateDiff download complete: {target_file} ({target_file.stat().st_size / 1024**2:.1f} MB) in {time.time()-t0:.1f}s!", flush=True)
