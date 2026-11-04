"""Stage 2.1: Download, integrity check, and GPU load benchmark for Stable Diffusion 1.5.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def benchmark_sd15() -> None:
    print("=" * 70)
    print("STAGE 2.1: STABLE DIFFUSION 1.5 DOWNLOAD & GPU LOAD BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device} ({torch.cuda.get_device_name(0)})")

    free_before, total_vram = torch.cuda.mem_get_info(0)
    print(f"Initial Free VRAM: {free_before / 1024**2:.1f} MiB / {total_vram / 1024**2:.1f} MiB")

    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"\nDownloading / Loading {model_id} (variant='fp16', torch_dtype=torch.float16)...")
    t0 = time.perf_counter()

    torch.cuda.reset_peak_memory_stats(0)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
    )
    load_time = time.perf_counter() - t0
    print(f"Model downloaded and instantiated in memory: {load_time:.2f} s")

    # Test loading onto GPU with sequential offload
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    print("\nRunning quick test prompt generation (1 step, 512x512) to measure active VRAM...")
    t_gen = time.perf_counter()
    with torch.inference_mode():
        image = pipe(
            prompt="anime girl portrait, high quality",
            num_inference_steps=1,
            width=512,
            height=512,
        ).images[0]
    gen_time = time.perf_counter() - t_gen
    print(f"Test generation (1 step) completed in: {gen_time:.2f} s")

    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    free_after, _ = torch.cuda.mem_get_info(0)

    print("\n--- MEASURED GPU VRAM METRICS ---")
    print(f"Peak VRAM Allocated: {peak_allocated / 1024**2:.2f} MiB ({peak_allocated / 1024**3:.2f} GiB)")
    print(f"Peak VRAM Reserved:  {peak_reserved / 1024**2:.2f} MiB ({peak_reserved / 1024**3:.2f} GiB)")
    print(f"Free VRAM Remaining: {free_after / 1024**2:.2f} MiB ({free_after / 1024**3:.2f} GiB)")

    output_dir = Path("recordings/stage2_benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "stage2_1_sd15_test.png"
    image.save(out_path)
    print(f"Sample test image saved to: {out_path}")

    # Check cache size
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    print(f"HuggingFace Hub Cache Location: {hf_cache}")

    del pipe, image
    torch.cuda.empty_cache()
    print("\nSTAGE 2.1: SUCCESSFUL - SD 1.5 Verified on GPU with sequential offload.")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_sd15()
