"""Spyder F5: render the saved Gaussian models into an annotated orbit video."""

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
from data_io import CameraInfo
from demo_config import RESULTS, new_output
from ply_utils import read_vertex_ply
from renderer import build_rasterizer, rasterize_gaussians
from view_pointclouds import _normalize_scalar, _scalar_colormap


VIDEO_WIDTH = 1440
VIDEO_HEIGHT = 960
TILE_WIDTH = 448
TILE_HEIGHT = 320
FPS = 24
HOLD_FRAMES = 6
REPEATS = 6
ORBIT_DEGREES = 70.0
BACKGROUND = (0.94, 0.95, 0.97)
EXPERIMENTS = ("translation", "deformation")


def orbit_camera(center, distance, angle, timestamp):
    eye = center + distance * np.array([math.sin(angle), -0.22, -math.cos(angle)])
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0., -1., 0.]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    w2c = np.eye(4)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -rotation @ eye
    focal = 0.5 * TILE_HEIGHT / math.tan(math.radians(25))
    image = torch.zeros((3, TILE_HEIGHT, TILE_WIDTH), device="cuda")
    return CameraInfo(image, "orbit", torch.tensor(w2c, dtype=torch.float32, device="cuda"),
        torch.tensor(eye, dtype=torch.float32, device="cuda"), focal, focal,
        (TILE_WIDTH - 1) / 2, (TILE_HEIGHT - 1) / 2, TILE_WIDTH, TILE_HEIGHT,
        2 * math.atan(TILE_WIDTH / (2 * focal)), 2 * math.atan(TILE_HEIGHT / (2 * focal)),
        0, timestamp, -1, 0., raster_pixel_offset=0.5)


def main():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("Install FFmpeg and put its bin directory on PATH.")
    output = new_output("video")
    models, settings, manifests = {}, {}, {}
    scalar_values = {"confidence": [], "uncertainty": []}
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    sources = {}
    for experiment in EXPERIMENTS:
        run = RESULTS / experiment / "sensitivity/fixed_on"
        path = run / "dynamic_gaussians_final.npz"
        models[experiment] = load_saved_model(path)
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        with open(run / "render_settings.json", encoding="utf-8") as file:
            settings[experiment] = json.load(file)
        with open(run / "independent_evaluation/sequence_manifest.json", encoding="utf-8") as file:
            manifests[experiment] = json.load(file)
        for row in manifests[experiment]["frames"]:
            vertex = read_vertex_ply(run / "independent_evaluation" / row["file"])
            xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")])
            scales = np.column_stack([vertex["scale_" + str(i)] for i in range(3)])
            radius = scales.max(axis=1, keepdims=True)
            low = np.minimum(low, (xyz - radius).min(axis=0))
            high = np.maximum(high, (xyz + radius).max(axis=0))
        scalar_values["confidence"].append(torch.log(models[experiment].get_opacity_concentration()).detach().cpu().numpy().ravel())
        scalar_values["uncertainty"].append(models[experiment].get_opacity_variance().detach().cpu().numpy().ravel())
    center = (low + high) / 2
    # Fit the entire bounds to every orbit view, using one common camera distance.
    corners = np.array([[x, y, z] for x in (low[0], high[0])
                        for y in (low[1], high[1]) for z in (low[2], high[2])])
    distance = 0.0
    focal = 0.5 * TILE_HEIGHT / math.tan(math.radians(25))
    for angle in np.linspace(-ORBIT_DEGREES / 2, ORBIT_DEGREES / 2, 253):
        angle = math.radians(angle)
        direction = np.array([math.sin(angle), -0.22, -math.cos(angle)])
        forward = -direction / np.linalg.norm(direction)
        right = np.cross(forward, np.array([0., -1., 0.]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        local = (corners - center) @ np.stack((right, down, forward)).T
        required_z = np.maximum(abs(local[:, 0]) * focal / (0.45 * TILE_WIDTH),
                                abs(local[:, 1]) * focal / (0.40 * TILE_HEIGHT))
        required = (required_z - local[:, 2]) / np.linalg.norm(direction)
        distance = max(distance, float(required.max()))
    distance *= 1.03
    ranges, colors = {}, {}
    for mode, arrays in scalar_values.items():
        combined = np.concatenate(arrays)
        assert np.isfinite(combined).all()
        ranges[mode] = (float(combined.min()), float(combined.max()))
        for experiment, values in zip(EXPERIMENTS, arrays):
            normalized, _, _ = _normalize_scalar(values, ranges[mode])
            rgb = _scalar_colormap(normalized, mode)
            colors[experiment, mode] = torch.tensor(rgb, dtype=torch.float32, device="cuda")
    frames = manifests[EXPERIMENTS[0]]["frames"]
    assert [row["timestamp"] for row in frames] == [row["timestamp"] for row in manifests[EXPERIMENTS[1]]["frames"]]
    frame_count = len(frames) * HOLD_FRAMES * REPEATS
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
    if font_path.exists():
        title_font = ImageFont.truetype(str(font_path), 32)
        label_font = ImageFont.truetype(str(font_path), 23)
        small_font = ImageFont.truetype(str(font_path), 18)
    else:
        title_font = ImageFont.load_default(size=32)
        label_font = ImageFont.load_default(size=23)
        small_font = ImageFont.load_default(size=18)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}", "-r", str(FPS),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / "4dgs_sequence.mp4")]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    counts = {}
    try:
        with torch.no_grad():
            for index in range(frame_count):
                frame = frames[(index // HOLD_FRAMES) % len(frames)]
                timestamp = float(frame["timestamp"])
                angle = math.radians(ORBIT_DEGREES) * (index / max(1, frame_count - 1) - 0.5)
                cam = orbit_camera(center, distance, angle, timestamp)
                canvas = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), "#f0f2f7")
                draw = ImageDraw.Draw(canvas)
                draw.text((24, 14), "Photogrammetry-Guided 4D Gaussian Splatting", font=title_font, fill="#162126")
                draw.text((24, 57), "Saved 3D Gaussian sequences | common orbit view | 12,000 Gaussians per frame", font=small_font, fill="#405057")
                for column, title in enumerate(("RGB", "Confidence: log(kappa)", "Uncertainty: opacity variance")):
                    draw.text((24 + 472 * column, 90), title, font=label_font, fill="#162126")
                for row, experiment in enumerate(EXPERIMENTS):
                    model = models[experiment]
                    params = model.get_render_params(timestamp,
                        temporal_window=settings[experiment]["temporal_window"],
                        temporal_sigma=settings[experiment]["temporal_sigma"])
                    active = params["active_indices"]
                    counts[experiment, frame["frame_idx"]] = len(active)
                    top = 125 + row * 410
                    bg = torch.tensor(BACKGROUND, dtype=torch.float32, device="cuda")
                    for column, mode in enumerate(("true", "confidence", "uncertainty")):
                        current = dict(params)
                        degree = model.active_sh_degree
                        if mode != "true":
                            current["colors_precomp"] = colors[experiment, mode][active]
                            current["shs"] = None
                            degree = 0
                        rasterizer = build_rasterizer(cam, degree, bg=bg, znear=0.01, zfar=100.)
                        rendered = rasterize_gaussians(rasterizer, current)[0]
                        assert torch.isfinite(rendered).all()
                        rgb = rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                        tile = Image.fromarray(np.rint(rgb * 255).astype(np.uint8))
                        left = 24 + 472 * column
                        canvas.paste(tile, (left, top))
                        if mode != "true":
                            gradient = _scalar_colormap(np.linspace(0, 1, TILE_WIDTH), mode)
                            bar = np.tile(np.rint(gradient[None] * 255).astype(np.uint8), (14, 1, 1))
                            canvas.paste(Image.fromarray(bar), (left, top + TILE_HEIGHT + 10))
                            lo, hi = ranges[mode]
                            draw.text((left, top + TILE_HEIGHT + 28), f"{lo:.5g}", font=small_font, fill="#26383d")
                            draw.text((left + TILE_WIDTH, top + TILE_HEIGHT + 28), f"{hi:.5g}", font=small_font, fill="#26383d", anchor="ra")
                    draw.text((34, top + 8), experiment.capitalize(), font=label_font, fill="#15252b")
                    draw.text((24, top + TILE_HEIGHT + 14), f"t = {timestamp:.2f} s   |   frame {frame['frame_idx']:02d}", font=label_font, fill="#162126")
                draw.text((24, 917), "Coordinates: meters   |   Saved timestamps: 0.00-0.30 s   |   Fixed color scales across both sequences", font=small_font, fill="#405057")
                if index == frame_count // 2 + (len(frames) // 2) * HOLD_FRAMES:
                    canvas.save(output / "poster.png")
                process.stdin.write(np.asarray(canvas).tobytes())
                if index % 42 == 0:
                    print(f"Video frame {index}/{frame_count}", flush=True)
    finally:
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        returncode = process.wait()
    if returncode:
        raise RuntimeError(error)
    palette_filter = "fps=8,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-i", str(output / "4dgs_sequence.mp4"),
        "-filter_complex", palette_filter, "-loop", "0", str(output / "preview.gif")], check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    records = [{"experiment": kind, "frame": frame, "gaussians": count} for (kind, frame), count in sorted(counts.items())]
    with open(output / "video_manifest.json", "w", encoding="utf-8") as file:
        json.dump({"renderer": "CUDA alpha-composited 3D Gaussian splatting", "source_sha256": sources,
            "width": VIDEO_WIDTH, "height": VIDEO_HEIGHT, "fps": FPS, "frames": frame_count,
            "physical_timestamps_s": [row["timestamp"] for row in frames], "hold_video_frames": HOLD_FRAMES,
            "repeats": REPEATS, "orbit_degrees": ORBIT_DEGREES, "opacity_filter": None,
            "scale_multiplier": 1.0, "color_ranges": ranges, "counts": records,
            "uncertainty": "mu*(1-mu)/(kappa+1), Beta opacity variance"}, file, indent=2)
    print("Video saved:", output)
    return output


if __name__ == "__main__":
    OUTPUT = main()
