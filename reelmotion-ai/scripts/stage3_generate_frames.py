import os
import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

def generate_phase5_frames():
    print("=" * 75, flush=True)
    print("STAGE 3: CONTROLNET AI GENERATION ACROSS 8 POSE FRAMES", flush=True)
    print("=" * 75, flush=True)

    out_base = PROJECT_ROOT / "recordings" / "stage3_experiment"
    pose_dir = out_base / "pose_maps"
    gen_dir = out_base / "generated_frames"
    comp_dir = out_base / "composite_frames"
    src_dir = out_base / "source_frames"
    gen_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    bg_path = out_base / "isolated_background.png"
    isolated_bg = cv2.imread(str(bg_path))
    if isolated_bg is None:
        isolated_bg = np.zeros((512, 512, 3), dtype=np.uint8)

    print("Loading SD 1.5 + ControlNet OpenPose in FP16...", flush=True)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    print(f"Loaded pipeline on CUDA in {time.time()-t0:.2f}s", flush=True)
    print("VRAM after loading:", torch.cuda.memory_allocated() / 1024**2, "MB", flush=True)

    char_prompt = "full body anime girl warrior, pink hair, athletic stance, masterpiece, sharp details"
    neg_prompt = "blurry, low quality, bad anatomy, deformed limbs, photographic"

    pose_files = sorted(list(pose_dir.glob("frame_*_openpose.png")))
    num_frames = len(pose_files)
    print(f"Found {num_frames} extracted pose maps to generate.", flush=True)

    gen_images = []
    comp_images = []
    frame_times = []
    SEED = 42

    for i, p_file in enumerate(pose_files):
        t_f0 = time.time()
        pose_img = Image.open(p_file).convert("RGB")

        with torch.inference_mode():
            # Step 1: Denoise latents on GPU
            latents = pipe(
                prompt=char_prompt,
                negative_prompt=neg_prompt,
                image=pose_img,
                num_inference_steps=20,
                controlnet_conditioning_scale=1.0,
                output_type="latent",
                generator=torch.Generator("cuda").manual_seed(SEED),
            ).images

            # Step 2: Offload UNet & ControlNet momentarily to CPU or decode with low VRAM
            # Since latents is on CUDA, decode with VAE slicing
            # To avoid VAE OOM, we can decode latents using pipe.vae directly with slicing
            decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            out_img = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

        dur = time.time() - t_f0
        frame_times.append(dur)
        gen_images.append(out_img)

        # Save generated frame
        gen_path = gen_dir / f"frame_{i:02d}_anime.png"
        out_img.save(gen_path)

        # Composite onto beach background
        gen_bgr = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)
        # Background threshold (anime background is neutral/flat)
        char_gray = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(char_gray, 235, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (7, 7), 0)[..., None]

        comp_bgr = (gen_bgr.astype(np.float32) * alpha + isolated_bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        comp_path = comp_dir / f"frame_{i:02d}_composite.png"
        cv2.imwrite(str(comp_path), comp_bgr)
        comp_images.append(comp_bgr)

        print(f"[{i+1}/{num_frames}] Frame {i} generated in {dur:.2f}s ({20/dur:.1f} it/s)", flush=True)

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
    peak_res = torch.cuda.max_memory_reserved() / 1024**2
    free_mem, total_mem = torch.cuda.mem_get_info()

    print("\n--- PERFORMANCE METRICS ---", flush=True)
    print(f"Peak VRAM Allocated: {peak_alloc:.1f} MB ({peak_alloc/1024:.2f} GB)", flush=True)
    print(f"Peak VRAM Reserved:  {peak_res:.1f} MB ({peak_res/1024:.2f} GB)", flush=True)
    print(f"Free VRAM Remaining: {free_mem/1024**2:.1f} MB", flush=True)
    print(f"Average Frame Latency: {np.mean(frame_times):.2f}s", flush=True)
    print(f"Total Generation Time: {np.sum(frame_times):.2f}s", flush=True)

    # Build Side-by-Side Comparison Board
    print("\nAssembling side-by-side comparison board...", flush=True)
    board_rows = []
    header = np.zeros((36, 512 * 4, 3), dtype=np.uint8)
    cv2.putText(header, "1. HUMAN INPUT       |  2. MEDIAPIPE OPENPOSE  |  3. AI ANIME CHARACTER  |  4. COMPOSITE REPLACEMENT",
                (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    board_rows.append(header)

    sample_indices = [0, num_frames // 2, num_frames - 1]
    for idx in sample_indices:
        h_file = src_dir / f"frame_{idx:02d}_human.png"
        h_img = cv2.imread(str(h_file)) if h_file.exists() else np.zeros((512, 512, 3), dtype=np.uint8)
        p_img = cv2.imread(str(pose_files[idx]))
        a_img = cv2.cvtColor(np.array(gen_images[idx]), cv2.COLOR_RGB2BGR)
        c_img = comp_images[idx]
        row = np.hstack([h_img, p_img, a_img, c_img])
        board_rows.append(row)

    board = np.vstack(board_rows)
    board_path = out_base / "phase5_character_replacement_board.png"
    cv2.imwrite(str(board_path), board)
    print(f"Saved comparison board to: {board_path}", flush=True)

    # Export MP4 Video
    video_path = PROJECT_ROOT / "exports" / "phase5_ai_character_replacement.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (512, 512))
    for c_img in comp_images:
        writer.write(c_img)
    writer.release()
    print(f"Saved MP4 video to: {video_path}", flush=True)

    # Write JSON report
    report_data = {
        "status": "SUCCESS",
        "device": torch.cuda.get_device_name(0),
        "resolution": "512x512",
        "frames_generated": num_frames,
        "peak_vram_allocated_mb": round(peak_alloc, 1),
        "peak_vram_reserved_mb": round(peak_res, 1),
        "avg_frame_time_sec": round(float(np.mean(frame_times)), 2),
        "fps_during_generation": round(float(1.0 / np.mean(frame_times)), 2),
        "total_time_sec": round(float(np.sum(frame_times)), 2),
        "models": {
            "diffusion": "runwayml/stable-diffusion-v1-5 (FP16)",
            "controlnet": "lllyasviel/control_v11p_sd15_openpose (FP16)",
            "pose": "MediaPipe PoseLandmarker",
            "segmentation": "MediaPipe ImageSegmenter",
        },
        "outputs": {
            "comparison_board": str(board_path),
            "video": str(video_path),
        }
    }
    report_file = PROJECT_ROOT / "reports" / "phase5_generation_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report_data, indent=2))
    print(f"Saved metrics report to: {report_file}", flush=True)

if __name__ == "__main__":
    generate_phase5_frames()
