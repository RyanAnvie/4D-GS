"""Spyder F5: show the saved real-data cook_spinach reconstruction results."""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from calibration_eval import load_saved_model
from data_io import CameraInfo, load_openmvg_sequence
from demo_config import new_output
from renderer import render_exposure_integrated


# Point this at the original project's 4D_GS folder when using another machine.
PUBLIC_PROJECT = ROOT.parent / "paper/AI_image/4D_GS"
if not PUBLIC_PROJECT.is_dir():
    PUBLIC_PROJECT = ROOT.parent / "4D_GS"
DATA = PUBLIC_PROJECT / "benchmark_data/cook_spinach_30f_50k"
RUNS = PUBLIC_PROJECT / "benchmark_results/cook_spinach_30f_surface_v1"
MODELS = ("dense_adaptive", "surface_adaptive")
LABELS = ("Adaptive densification", "Surface-constrained")
FPS = 30
HOLD_FRAMES = 4
COMPARISON_REPEATS = 3
ORBIT_DEGREES = 10.0
DETAIL_BOX = (120, 170, 428, 350)  # Same fixed image crop in all three columns.
PREVIEW_ONLY = False


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def encode_video(paths, output, fps, hold=1, repeats=1):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("FFmpeg must be installed and on PATH.")
    with Image.open(paths[0]) as first:
        width, height = first.size
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        for _ in range(repeats):
            for path in paths:
                with Image.open(path) as source:
                    frame = source.convert("RGB")
                    assert frame.size == (width, height)
                    pixels = frame.tobytes()
                for _ in range(hold):
                    process.stdin.write(pixels)
    finally:
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        code = process.wait()
    if code:
        raise RuntimeError(error)


def orbit_camera(base, distance, angle, timestamp, width=936, height=702):
    rotation = base.w2c[:3, :3].cpu().numpy().astype(np.float64)
    center = base.camera_center.cpu().numpy().astype(np.float64)
    target = center + rotation.T @ np.array([0., 0., distance])
    local_eye = distance * np.array([math.sin(angle), 0., 1.0 - math.cos(angle)])
    eye = center + rotation.T @ local_eye
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(rotation[1], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    w2c = np.eye(4)
    w2c[:3, :3] = np.stack((right, down, forward))
    w2c[:3, 3] = -w2c[:3, :3] @ eye
    sx, sy = width / base.width, height / base.height
    fx, fy = base.fx * sx, base.fy * sy
    # Preserve the original rasterizer's pixel-center convention under resizing.
    offset = base.raster_pixel_offset
    cx = (base.cx + offset) * sx - offset
    cy = (base.cy + offset) * sy - offset
    return CameraInfo(torch.zeros((3, height, width), device="cuda"), "virtual_orbit",
        torch.tensor(w2c, dtype=torch.float32, device="cuda"),
        torch.tensor(eye, dtype=torch.float32, device="cuda"), fx, fy, cx, cy, width, height,
        2 * math.atan(width / (2 * fx)), 2 * math.atan(height / (2 * fy)),
        base.frame_idx, timestamp, -1, 0., raster_pixel_offset=offset)


def main(preview_only=PREVIEW_ONLY):
    for kind in MODELS:
        if not (RUNS / kind / "dynamic_gaussians_final.npz").is_file():
            raise FileNotFoundError("Set PUBLIC_PROJECT to the original project with the saved cook_spinach runs.")
    output = new_output("cook_spinach_video")
    frames_root = output / "frames"
    frames_root.mkdir()
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
    fonts = []
    for size in (36, 28, 22):
        if font_path.is_file():
            fonts.append(ImageFont.truetype(str(font_path), size))
        else:
            fonts.append(ImageFont.load_default(size=size))
    title_font, label_font, small_font = fonts
    ink = "#15252b"
    credit = "Data: Neural 3D Video / Facebook Research | CC BY-NC 4.0 | Derived 4DGS results"
    comparison_paths = []
    source_images = {}
    for frame in range(30):
        stem = f"cam00_frame_{frame:06d}"
        reference = DATA / "images" / (stem + ".png")
        paths = [reference] + [RUNS / kind / "evaluation" / (stem + "_render.png") for kind in MODELS]
        canvas = Image.new("RGB", (1920, 1080), "#f0f2f7")
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 16), "cook_spinach | Real multiview video reconstruction", font=title_font, fill=ink)
        draw.text((24, 65), "Held-out camera 00 | 30 saved timestamps | 100,000 Gaussians per timestamp | SH degree 1", font=small_font, fill=ink)
        for column, (label, path) in enumerate(zip(("Original video: camera 00",) + LABELS, paths)):
            with Image.open(path) as source:
                picture = source.convert("RGB")
            assert picture.size == (480, 360)
            if column == 0:
                for kind in MODELS:
                    with Image.open(RUNS / kind / "evaluation" / (stem + "_reference.png")) as saved:
                        np.testing.assert_array_equal(np.asarray(picture), np.asarray(saved.convert("RGB")))
            source_images[path.relative_to(PUBLIC_PROJECT).as_posix()] = file_hash(path)
            left = 24 + column * 628
            draw.text((left, 108), label, font=label_font, fill=ink)
            canvas.paste(picture.resize((616, 462), Image.Resampling.BILINEAR), (left, 150))
            canvas.paste(picture.crop(DETAIL_BOX).resize((616, 360), Image.Resampling.BILINEAR), (left, 666))
        draw.text((24, 625), f"t = {frame / 30:.3f} s | frame {frame:02d}/29 | 0.25x playback | Below: same hand/pan detail crop", font=small_font, fill=ink)
        draw.text((24, 1040), credit, font=small_font, fill=ink)
        path = frames_root / f"comparison_{frame:06d}.png"
        canvas.save(path)
        comparison_paths.append(path)
        if frame == 15:
            canvas.save(output / "poster.png")
    print("Prepared all 30 original / adaptive / surface comparisons.", flush=True)
    if not preview_only:
        encode_video(comparison_paths, output / "cook_spinach_comparison.mp4", FPS,
                     hold=HOLD_FRAMES, repeats=COMPARISON_REPEATS)

    # Use the established OpenMVG reader, selecting the actual held-out camera.
    with open(DATA / "sfm.json", encoding="utf-8") as file:
        sfm = json.load(file)
    with open(DATA / "views_manifest.json", encoding="utf-8") as file:
        views = json.load(file)
    selected_views = []
    for view in sfm["views"]:
        filename = view["value"]["ptr_wrapper"]["data"]["filename"]
        if views[filename]["camera_id"] == 0:
            selected_views.append(view)
    sfm["views"] = selected_views
    sfm["root_path"] = str(DATA / "images")
    with open(output / "camera00_sfm.json", "w", encoding="utf-8") as file:
        json.dump(sfm, file)
    cameras = load_openmvg_sequence(str(output / "camera00_sfm.json"),
        views_manifest=str(DATA / "views_manifest.json"), target_width=480, device="cuda")
    assert [cam.frame_idx for cam in cameras] == list(range(30))
    np.testing.assert_allclose([cam.timestamp for cam in cameras], np.arange(30) / 30)
    cycle = 30 * HOLD_FRAMES
    schedule = []
    for index in range(3 * cycle):
        phase = index // cycle
        step = index % cycle
        frame = 15 if phase == 1 else step // HOLD_FRAMES
        angle = 0. if phase == 0 else ORBIT_DEGREES * math.sin(2 * math.pi * step / (cycle - 1))
        schedule.append({"frame": frame, "timestamp_s": frame / 30, "angle_degrees": angle,
                         "phase": phase})
    selected_indices = [0, cycle + cycle // 4, cycle + 3 * cycle // 4] if preview_only else list(range(len(schedule)))
    models_info = {}
    distance = None
    for kind in MODELS:
        run = RUNS / kind
        model_path = run / "dynamic_gaussians_final.npz"
        print("Loading", kind, flush=True)
        model = load_saved_model(model_path)
        with open(run / "render_settings.json", encoding="utf-8") as file:
            settings = json.load(file)
        with open(run / "train_config.json", encoding="utf-8") as file:
            config = json.load(file)
        native_max_difference = 0
        counts = []
        with torch.no_grad():
            for cam in cameras:
                native = render_exposure_integrated(model, cam, use_exposure=False, **settings)
                rgb = native["image"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                pixels = np.rint(rgb * 255).astype(np.uint8)
                with Image.open(run / "evaluation" / f"cam00_frame_{cam.frame_idx:06d}_render.png") as saved:
                    difference = abs(pixels.astype(np.int16) - np.asarray(saved).astype(np.int16))
                native_max_difference = max(native_max_difference, int(difference.max()))
                counts.append(len(native["active_indices"]))
            assert native_max_difference <= 1, (kind, native_max_difference)
            assert counts == [100000] * 30
            if distance is None:
                active = model.active_indices_at(cameras[15].timestamp, settings["temporal_window"])
                points = model.position_at(cameras[15].timestamp)[active]
                local = points @ cameras[0].w2c[:3, :3].T + cameras[0].w2c[:3, 3]
                # This statistic sets only the camera pivot; all Gaussians are rendered.
                distance = float(torch.median(local[:, 2]))
                assert distance > settings["znear"]
            for index in selected_indices:
                row = schedule[index]
                cam = orbit_camera(cameras[row["frame"]], distance,
                                   math.radians(row["angle_degrees"]), row["timestamp_s"])
                raw = render_exposure_integrated(model, cam, use_exposure=False, **settings)["image"]
                assert torch.isfinite(raw).all()
                pixels = np.rint(raw.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(pixels).save(frames_root / f"{kind}_{index:06d}.png")
                if index % 60 == 0:
                    print(kind, "orbit frame", index, "/", len(schedule), flush=True)
        models_info[kind] = {"file": model_path.relative_to(PUBLIC_PROJECT).as_posix(),
            "sha256": file_hash(model_path), "iterations": config["iterations"],
            "sh_degree": model.active_sh_degree, "active_gaussians_per_timestamp": counts,
            "native_render_max_uint8_difference": native_max_difference, "render_settings": settings}
        del model
        torch.cuda.empty_cache()
    orbit_paths = []
    for index in selected_indices:
        row = schedule[index]
        phases = ("Time playback | camera fixed", "Camera orbit | scene time fixed", "Time playback + camera orbit")
        canvas = Image.new("RGB", (1920, 900), "#f0f2f7")
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 12), "cook_spinach | Saved 3D Gaussian models", font=title_font, fill=ink)
        draw.text((24, 62), phases[row["phase"]] + f" | t = {row['timestamp_s']:.3f} s | angle = {row['angle_degrees']:+.1f} deg", font=small_font, fill=ink)
        for column, (kind, label) in enumerate(zip(MODELS, LABELS)):
            left = 16 + column * 952
            draw.text((left, 102), label, font=label_font, fill=ink)
            with Image.open(frames_root / f"{kind}_{index:06d}.png") as picture:
                canvas.paste(picture, (left, 148))
        draw.text((24, 862), credit, font=small_font, fill=ink)
        path = frames_root / f"orbit_{index:06d}.png"
        canvas.save(path)
        orbit_paths.append(path)
        if index == selected_indices[len(selected_indices) // 2]:
            canvas.save(output / "orbit_poster.png")
    if not preview_only:
        encode_video(orbit_paths, output / "cook_spinach_3d.mp4", FPS)
        ffmpeg = shutil.which("ffmpeg")
        palette = "fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse"
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-i",
            str(output / "cook_spinach_comparison.mp4"), "-t", "4", "-filter_complex", palette,
            "-loop", "0", str(output / "preview.gif")], check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    with open(output / "video_manifest.json", "w", encoding="utf-8") as file:
        json.dump({"dataset": "Neural 3D Video: cook_spinach", "source":
            "https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0",
            "license": "CC-BY-NC-4.0", "models": models_info, "source_images_sha256": source_images,
            "source_fps": 30, "source_frame_ids": list(range(30)), "source_image_size": [480, 360],
            "video_fps": FPS, "comparison_frames": 30 * HOLD_FRAMES * COMPARISON_REPEATS,
            "comparison_size": [1920, 1080], "orbit_size": [1920, 900],
            "comparison_playback_speed": 1.0 / HOLD_FRAMES, "comparison_repeats": COMPARISON_REPEATS,
            "detail_crop_xyxy": DETAIL_BOX, "comparison_resize": "Bilinear display scaling, full view retained above crop",
            "orbit_pivot_depth_sfm_units": distance, "orbit_schedule": schedule,
            "preview_only": preview_only, "opacity_filter": None, "scale_modifier": 1.0,
            "coordinates": "Original released SfM coordinates; unchanged",
            "rendering": "CUDA alpha-composited Gaussians, saved positions/scales/opacity/SH, black background"},
            file, indent=2, allow_nan=False)
    print("Public-data videos:", output, flush=True)
    return output


if __name__ == "__main__":
    OUTPUT = main()
