import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

url = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"

# Get snapshot dir
cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--h94--IP-Adapter" / "snapshots"
snapshots = list(cache_dir.glob("*"))
target_dir = snapshots[0] / "models" / "image_encoder"
target_dir.mkdir(parents=True, exist_ok=True)
target_file = target_dir / "model.safetensors"

print(f"Checking remote file size for {url}...", flush=True)
head = requests.head(url, allow_redirects=True, timeout=15)
total_size = int(head.headers.get("content-length", 0))
redirect_url = head.url
print(f"Direct CDN URL: {redirect_url[:80]}...", flush=True)
print(f"Total size: {total_size / 1024**2:.1f} MB", flush=True)

NUM_WORKERS = 8
chunk_size = (total_size + NUM_WORKERS - 1) // NUM_WORKERS

# Pre-allocate output file
with open(target_file, "wb") as f:
    f.seek(total_size - 1)
    f.write(b"\0")

print(f"Pre-allocated target file. Spawning {NUM_WORKERS} parallel download threads...", flush=True)

downloaded = [0] * NUM_WORKERS
t0 = time.time()

def download_chunk(worker_id: int):
    start = worker_id * chunk_size
    end = min(total_size - 1, (worker_id + 1) * chunk_size - 1)
    if start >= total_size:
        return
    headers = {"Range": f"bytes={start}-{end}"}
    resp = requests.get(redirect_url, headers=headers, stream=True, timeout=60)
    with open(target_file, "r+b") as f:
        f.seek(start)
        for data in resp.iter_content(chunk_size=1024 * 1024):
            if data:
                f.write(data)
                downloaded[worker_id] += len(data)

def monitor():
    while True:
        time.sleep(4.0)
        total_dl = sum(downloaded)
        pct = (total_dl / total_size) * 100
        speed = (total_dl / 1024**2) / max(0.1, time.time() - t0)
        print(f"Progress: {total_dl / 1024**2:.1f} MB / {total_size / 1024**2:.1f} MB ({pct:.1f}%) [{speed:.1f} MB/s]", flush=True)
        if total_dl >= total_size:
            break

import threading
mon_thread = threading.Thread(target=monitor, daemon=True)
mon_thread.start()

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(download_chunk, i) for i in range(NUM_WORKERS)]
    for f in as_completed(futures):
        f.result()

print(f"Download completed successfully in {time.time()-t0:.1f}s!", flush=True)
print(f"Final file size: {target_file.stat().st_size / 1024**2:.1f} MB", flush=True)
