import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image
import numpy as np

os.makedirs("recordings/stage2_benchmarks", exist_ok=True)

print("=" * 70, flush=True)
print("STAGE 2.2: CONTROLNET OPENPOSE DOWNLOAD & GPU BENCHMARK", flush=True)
print("=" * 70, flush=True)

print("Free VRAM before loading ControlNet:", torch.cuda.mem_get_info()[0] / 1024**2, "MB", flush=True)

controlnet_id = "lllyasviel/control_v11p_sd15_openpose"
print(f"Loading {controlnet_id}...", flush=True)
t0 = time.time()
controlnet = ControlNetModel.from_pretrained(
    controlnet_id,
    torch_dtype=torch.float16,
    variant="fp16",
)
print(f"ControlNet loaded in {time.time()-t0:.2f}s", flush=True)

print("Creating StableDiffusionControlNetPipeline...", flush=True)
t0 = time.time()
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
print(f"Full ControlNet pipeline loaded and moved to CUDA in {time.time()-t0:.2f}s", flush=True)
print("VRAM after loading to CUDA:", torch.cuda.memory_allocated() / 1024**2, "MB allocated", flush=True)

# Create a synthetic OpenPose skeleton image (black background with colored bones)
pose_img = np.zeros((512, 512, 3), dtype=np.uint8)
import cv2
cv2.circle(pose_img, (256, 100), 20, (0, 0, 255), -1)  # head
cv2.line(pose_img, (256, 120), (256, 280), (255, 0, 0), 4)  # spine
cv2.line(pose_img, (256, 150), (180, 220), (0, 255, 0), 4)
cv2.line(pose_img, (180, 220), (140, 300), (0, 255, 0), 4)
cv2.line(pose_img, (256, 150), (332, 220), (255, 255, 0), 4)
cv2.line(pose_img, (332, 220), (372, 300), (255, 255, 0), 4)
cv2.line(pose_img, (256, 280), (210, 400), (255, 0, 255), 4)
cv2.line(pose_img, (210, 400), (200, 500), (255, 0, 255), 4)
cv2.line(pose_img, (256, 280), (302, 400), (0, 255, 255), 4)
cv2.line(pose_img, (302, 400), (312, 500), (0, 255, 255), 4)

pose_pil = Image.fromarray(pose_img)
pose_pil.save("recordings/stage2_benchmarks/test_openpose_conditioning.png")
print("Saved pose conditioning map to recordings/stage2_benchmarks/test_openpose_conditioning.png", flush=True)

print("Running 20-step ControlNet generation on GPU...", flush=True)
torch.cuda.reset_peak_memory_stats()
t_gen = time.time()
with torch.inference_mode():
    # 1. Generate latents with ControlNet + UNet on GPU
    latents = pipe(
        prompt="full body anime girl standing in dynamic pose, high quality, masterpiece, detailed artwork",
        negative_prompt="blurry, low quality, bad anatomy, deformed limbs",
        image=pose_pil,
        num_inference_steps=20,
        controlnet_conditioning_scale=1.0,
        output_type="latent",
        generator=torch.Generator("cuda").manual_seed(42),
    ).images

    # 2. Free up UNet and ControlNet memory by deleting them before VAE decode
    print("Denoising complete. Deleting UNet & ControlNet to free VRAM for VAE...", flush=True)
    del pipe.unet
    del pipe.controlnet
    del controlnet
    torch.cuda.empty_cache()

    # 3. Decode latents with VAE on GPU
    print("Decoding latents with VAE on GPU...", flush=True)
    latents = latents.to("cuda")
    decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    out_img = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

gen_time = time.time() - t_gen
print(f"Generated ControlNet image in {gen_time:.2f}s ({gen_time/20:.3f}s/step)!", flush=True)
out_img.save("recordings/stage2_benchmarks/test_controlnet_openpose_result.png")
print("Saved result to recordings/stage2_benchmarks/test_controlnet_openpose_result.png", flush=True)

peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
peak_res = torch.cuda.max_memory_reserved() / 1024**2
free_vram, total_vram = torch.cuda.mem_get_info()
print(f"Peak VRAM Allocated: {peak_alloc:.1f} MB ({peak_alloc/1024:.2f} GB)", flush=True)
print(f"Peak VRAM Reserved:  {peak_res:.1f} MB ({peak_res/1024:.2f} GB)", flush=True)
print(f"Remaining Free VRAM: {free_vram/1024**2:.1f} MB ({free_vram/1024**3:.2f} GB)", flush=True)
print("STAGE 2.2: SUCCESSFUL", flush=True)

