import os
import sys
import time
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

os.makedirs("recordings/stage2_benchmarks", exist_ok=True)

print("Torch version:", torch.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)
print("Device:", torch.cuda.get_device_name(0), flush=True)
print("Mem get info:", torch.cuda.mem_get_info(), flush=True)

print("Loading pipeline...", flush=True)
t0 = time.time()
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
print(f"Loaded in {time.time()-t0:.2f}s", flush=True)

print("Moving to cuda...", flush=True)
pipe = pipe.to("cuda")
print("Moved to cuda successfully!", flush=True)

print("Testing 20-step generation with DPMSolver...", flush=True)
t_gen = time.time()
with torch.inference_mode():
    img = pipe(
        prompt="1girl, anime, masterpiece, high quality, colorful portrait",
        negative_prompt="low quality, bad anatomy, blurry",
        num_inference_steps=20,
        width=512,
        height=512,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
gen_time = time.time() - t_gen
print(f"Generated 20-step image in {gen_time:.2f}s ({gen_time/20:.3f}s/step)!", flush=True)
img.save("recordings/stage2_benchmarks/test_sd15_20steps.png")
print("Saved to recordings/stage2_benchmarks/test_sd15_20steps.png", flush=True)
print("Max allocated:", torch.cuda.max_memory_allocated() / 1024**2, "MB", flush=True)
print("Max reserved:", torch.cuda.max_memory_reserved() / 1024**2, "MB", flush=True)

