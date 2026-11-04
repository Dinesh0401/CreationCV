import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
from huggingface_hub import hf_hub_download

print("Downloading AnimateDiff v1-5-2 motion adapter (FP16)...", flush=True)
t0 = time.time()
motion_path = hf_hub_download(
    repo_id="guoyww/animatediff-motion-adapter-v1-5-2",
    filename="diffusion_pytorch_model.fp16.safetensors",
)
config_path = hf_hub_download(
    repo_id="guoyww/animatediff-motion-adapter-v1-5-2",
    filename="config.json",
)
print(f"Downloaded AnimateDiff motion adapter in {time.time()-t0:.2f}s: {motion_path}", flush=True)

