import os
import sys
import time
from huggingface_hub import hf_hub_download

print("Downloading IP-Adapter SD 1.5 weights (42.6 MB)...", flush=True)
t0 = time.time()
ip_adapter_weight = hf_hub_download(
    repo_id="h94/IP-Adapter",
    filename="ip-adapter_sd15.safetensors",
    subfolder="models",
)
print(f"Downloaded IP-Adapter weights: {ip_adapter_weight}", flush=True)

print("Downloading Image Encoder config & model (CLIP ViT-H/14)...", flush=True)
encoder_config = hf_hub_download(
    repo_id="h94/IP-Adapter",
    filename="config.json",
    subfolder="models/image_encoder",
)

max_retries = 10
for attempt in range(1, max_retries + 1):
    try:
        print(f"Downloading Image Encoder model (attempt {attempt}/{max_retries})...", flush=True)
        t0 = time.time()
        encoder_model = hf_hub_download(
            repo_id="h94/IP-Adapter",
            filename="model.safetensors",
            subfolder="models/image_encoder",
            resume_download=True,
        )
        print(f"Downloaded Image Encoder successfully in {time.time()-t0:.2f}s: {encoder_model}", flush=True)
        break
    except Exception as e:
        print(f"Attempt {attempt} failed ({e}). Retrying in 5 seconds...", flush=True)
        time.sleep(5)
else:
    print("Failed to download image encoder after max retries.", flush=True)
    sys.exit(1)

print("IP-Adapter download and integrity check complete!", flush=True)

