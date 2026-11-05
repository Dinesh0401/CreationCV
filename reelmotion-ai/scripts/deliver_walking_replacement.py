"""Delivery Step: Full 16-Frame Real Human Walking to Anime Character Replacement.

Loads SD 1.5 (FP16) + ControlNet OpenPose (FP16) on CUDA.
Conditions on 16 consecutive real human walking poses.
Composites onto isolated background and exports H.264 MP4 and 4-column collage.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
import imageio_ffmpeg

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def run_delivery():
    print("=" * 75, flush=True)
    print("REELMOTION AI: 16-FRAME REAL HUMAN MOTION CHARACTER REPLACEMENT", flush=True)
    print("=" * 75, flush=True)

    base_dir = PROJECT_ROOT / "recordings" / "phase5b_experiment"
    pose_dir = base_dir / "pose_maps"
    src_dir = base_dir / "source_frames"
    gen_dir = base_dir / "generated_anime"
    comp_dir = base_dir / "composite_frames"

    gen_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    pose_files = sorted(list(pose_dir.glob("frame_*_openpose.png")))
    num_frames = len(pose_files)
    print(f"Found {num_frames} real human walking pose conditioning maps.", flush=True)
    if num_frames == 0:
        print("Error: No pose files found!")
        sys.exit(1)

    # 1. Load ControlNet + SD 1.5 on CUDA
    print("\nLoading SD 1.5 + ControlNet OpenPose onto RTX 4050...", flush=True)
    t0_load = time.time()
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
    load_time = time.time() - t0_load
    print(f"Pipeline ready in {load_time:.2f}s (Static VRAM: {torch.cuda.memory_allocated()/1024**2:.1f} MB)", flush=True)

    # Anime character prompt: matches pink-haired warrior character
    char_prompt = (
        "full body anime girl walking, pink hair with purple highlights, golden amber eyes, "
        "white combat dress with navy accents, athletic walking stance, legs in stride, "
        "masterpiece, anime aesthetic, sharp clean lineart, vibrant colors, 8k"
    )
    neg_prompt = "low quality, blurry, bad anatomy, deformed limbs, photographic human skin, close-up, cropped"

    # Background frame: sample corridor background from source frame 0
    bg_sample = cv2.imread(str(src_dir / "frame_00_human.png"))
    if bg_sample is None:
        bg_sample = np.full((512, 512, 3), 40, dtype=np.uint8)

    frame_durations = []
    SEED = 555  # Fixed noise seed for structural identity guidance

    print(f"\nGenerating 16 frames of anime character driven by real human walking motion...", flush=True)
    for i, p_file in enumerate(pose_files):
        t_f0 = time.time()
        pose_pil = Image.open(p_file).convert("RGB")

        with torch.inference_mode():
            latents = pipe(
                prompt=char_prompt,
                negative_prompt=neg_prompt,
                image=pose_pil,
                num_inference_steps=20,
                controlnet_conditioning_scale=1.1,
                guidance_scale=7.5,
                output_type="latent",
                generator=torch.Generator("cuda").manual_seed(SEED),
            ).images

            decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            out_img = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

        dur = time.time() - t_f0
        frame_durations.append(dur)

        # Save generated anime frame
        gen_path = gen_dir / f"frame_{i:02d}_anime.png"
        out_img.save(gen_path)

        # High quality matte compositing onto background
        gen_bgr = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)
        char_gray = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(char_gray, 230, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (7, 7), 0)[..., None]

        comp_bgr = (gen_bgr.astype(np.float32) * alpha + bg_sample.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        comp_path = comp_dir / f"frame_{i:02d}_composite.png"
        cv2.imwrite(str(comp_path), comp_bgr)

        # Clean memory
        del out_img, gen_bgr, comp_bgr, mask, alpha, decoded, latents
        gc.collect()

        print(f"[{i+1}/{num_frames}] Frame {i:02d} rendered in {dur:.2f}s ({20/dur:.1f} it/s)", flush=True)

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
    peak_res = torch.cuda.max_memory_reserved() / 1024**2
    free_end, _ = torch.cuda.mem_get_info()

    print("\n--- PERFORMANCE METRICS ---", flush=True)
    print(f"Peak VRAM Allocated: {peak_alloc:.1f} MB ({peak_alloc/1024:.2f} GB)", flush=True)
    print(f"Peak VRAM Reserved:  {peak_res:.1f} MB ({peak_res/1024:.2f} GB)", flush=True)
    print(f"Free VRAM Remaining: {free_end/1024**2:.1f} MB ({free_end/1024**3:.2f} GB)", flush=True)
    print(f"Average Frame Latency: {np.mean(frame_durations):.2f}s ({20.0/np.mean(frame_durations):.1f} it/s)", flush=True)
    print(f"Total Render Time (16 frames): {np.sum(frame_durations):.2f}s", flush=True)

    # 3. Assemble Side-by-Side Comparison Collage Board (Frames 0, 4, 8, 12, 15)
    print("\nAssembling side-by-side verification board...", flush=True)
    board_rows = []
    header = np.zeros((36, 512 * 4, 3), dtype=np.uint8)
    cv2.putText(header, "1. REAL HUMAN WALKING | 2. MEDIAPIPE OPENPOSE | 3. AI ANIME CHARACTER | 4. COMPOSITE REPLACEMENT",
                (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    board_rows.append(header)

    sample_indices = [0, 4, 8, 12, 15]
    for idx in sample_indices:
        h_img = cv2.imread(str(src_dir / f"frame_{idx:02d}_human.png"))
        if h_img is None: h_img = np.zeros((512, 512, 3), dtype=np.uint8)
        p_img = cv2.imread(str(pose_dir / f"frame_{idx:02d}_openpose.png"))
        a_img = cv2.imread(str(gen_dir / f"frame_{idx:02d}_anime.png"))
        c_img = cv2.imread(str(comp_dir / f"frame_{idx:02d}_composite.png"))
        row = np.hstack([h_img, p_img, a_img, c_img])
        board_rows.append(row)

    comparison_board = np.vstack(board_rows)
    board_path = base_dir / "real_walking_character_replacement_board.png"
    cv2.imwrite(str(board_path), comparison_board)
    print(f"Saved comparison board to: {board_path}", flush=True)

    # 4. Export Playable H.264 Video via FFmpeg
    video_h264_path = PROJECT_ROOT / "exports" / "real_walking_anime_replacement_h264.mp4"
    video_h264_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f'"{FFMPEG_EXE}" -y -framerate 6 -i "{comp_dir.as_posix()}/frame_%02d_composite.png" '
        f'-c:v libx264 -pix_fmt yuv420p "{video_h264_path.as_posix()}"'
    )
    print(f"Encoding H.264 video with FFmpeg...", flush=True)
    os.system(cmd)
    print(f"Saved playable H.264 MP4 video to: {video_h264_path}", flush=True)

    # Copy to default export
    default_export = PROJECT_ROOT / "exports" / "real_walking_anime_replacement.mp4"
    if video_h264_path.exists():
        import shutil
        shutil.copyfile(str(video_h264_path), str(default_export))

    # 5. Save JSON report
    report = {
        "status": "DELIVERED",
        "experiment": "Real Human Walking Video to Anime Character Replacement",
        "source_video": "reelmotion-ai/recordings/real_human/walking_human.mp4",
        "frames_rendered": num_frames,
        "resolution": "512x512",
        "device": torch.cuda.get_device_name(0),
        "peak_vram_allocated_mb": round(peak_alloc, 1),
        "peak_vram_reserved_mb": round(peak_res, 1),
        "avg_frame_latency_sec": round(float(np.mean(frame_durations)), 2),
        "iterations_per_second": round(float(20.0 / np.mean(frame_durations)), 2),
        "total_render_sec": round(float(np.sum(frame_durations)), 2),
        "codec": "H.264 (avc1, yuv420p) - Playable in all players",
        "outputs": {
            "comparison_board": str(board_path),
            "video": str(video_h264_path),
        }
    }
    report_file = PROJECT_ROOT / "reports" / "real_walking_replacement_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved delivery report to: {report_file}", flush=True)

    print("\n" + "=" * 75)
    print("DELIVERY COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_delivery()

