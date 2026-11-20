"""Independent textured triangle-mesh sequences; editable Spyder experiment."""

import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import open3d as o3d

from data_io import estimate_normals
from prepare_eth3d_geometry import reconstruct_rectified_pair


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "benchmark_data/controlled_dynamics_dev_v1"
SCENES = ("static", "translation", "deformation")
SEED = 1201  # Development only; reserved calibration/test seeds 2201/3201.
WIDTH = 448
HEIGHT = 288
FOCAL_PX = 400.0
BASELINE_M = 0.12
FPS = 20.0
FRAME_IDS = tuple(range(7))
HELD_OUT_FRAME_IDS = (1, 3, 5)
HELD_OUT_CAMERA_IDS = (2,)
EXPOSURE_SECONDS = 0.04
SHUTTER_SAMPLES = 32
MAX_POINTS = 6000
MESH_COLUMNS = 121
MESH_ROWS = 89
CAMERA_CENTERS = np.array([[0., 0., 0.], [BASELINE_M, 0., 0.], [0.06, 0.05, 0.]], dtype=np.float32)


def make_surface(kind, timestamp, seed=SEED):
    """Designed meters/seconds, not a simulation of real granular material."""
    x, y = np.meshgrid(np.linspace(-0.85, 0.85, MESH_COLUMNS),
                       np.linspace(-0.58, 0.58, MESH_ROWS))
    phase = np.random.default_rng(seed).uniform(0., 2. * np.pi)
    z = 1.3 + 0.035 * np.sin(7. * x + phase) * np.cos(8. * y)
    if kind == "deformation":
        z += 0.03 * np.sin(2. * np.pi * (1.2 * timestamp + 0.8 * x))
    elif kind not in ("static", "translation"):
        raise ValueError(f"Unknown controlled surface: {kind}")
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)
    if kind != "static":
        vertices += timestamp * np.array([0.35, 0.03, 0.20], dtype=np.float32)
    uv = np.column_stack(((x.ravel() + 0.85) / 1.7, (y.ravel() + 0.58) / 1.16)).astype(np.float32)
    cells = np.arange(MESH_ROWS * MESH_COLUMNS).reshape(MESH_ROWS, MESH_COLUMNS)[:-1, :-1].ravel()
    first = np.column_stack((cells, cells + 1, cells + MESH_COLUMNS))
    second = np.column_stack((cells + 1, cells + MESH_COLUMNS + 1, cells + MESH_COLUMNS))
    triangles = np.concatenate((first, second)).astype(np.uint32)
    return vertices, triangles, uv


def make_texture(seed=SEED):
    rng = np.random.default_rng(seed)
    raw = rng.random((768, 1024, 3), dtype=np.float32)
    texture = cv2.GaussianBlur(raw, (0, 0), 0.6)
    return texture


def render_mesh_views(vertices, triangles, uv, texture, centers=CAMERA_CENTERS,
                      width=WIDTH, height=HEIGHT, focal=FOCAL_PX):
    """Pinhole rays [right, down, forward], integer pixel centers, linear RGB."""
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(triangles))
    rows, columns = np.indices((height, width), dtype=np.float32)
    direction = np.stack(((columns - (width - 1) / 2.) / focal,
                          (rows - (height - 1) / 2.) / focal, np.ones_like(rows)), axis=-1)
    rays = np.empty((len(centers), height, width, 6), dtype=np.float32)
    rays[..., 3:] = direction
    for index, center in enumerate(centers):
        rays[index, ..., :3] = center
    hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)
    depth = hit["t_hit"].numpy()
    valid = np.isfinite(depth)
    triangle_ids = hit["primitive_ids"].numpy()[valid]
    barycentric = hit["primitive_uvs"].numpy()[valid]
    weights = np.column_stack((1. - barycentric.sum(axis=1), barycentric))
    hit_uv = (uv[triangles[triangle_ids]] * weights[:, :, None]).sum(axis=1)
    map_x = (hit_uv[:, 0] * (texture.shape[1] - 1)).astype(np.float32)
    map_y = (hit_uv[:, 1] * (texture.shape[0] - 1)).astype(np.float32)
    # remap dimensions must be <32767 in OpenCV. A square lookup grid avoids
    # treating all image samples as one oversized row.
    size = int(np.ceil(np.sqrt(len(hit_uv))))
    lookup_x = np.zeros((size, size), dtype=np.float32)
    lookup_y = np.zeros((size, size), dtype=np.float32)
    lookup_x.ravel()[:len(hit_uv)] = map_x
    lookup_y.ravel()[:len(hit_uv)] = map_y
    sampled = cv2.remap(texture, lookup_x, lookup_y, cv2.INTER_LINEAR)
    rgb = np.zeros((len(centers), height, width, 3), dtype=np.float32)
    rgb[valid] = sampled.reshape(-1, 3)[:len(hit_uv)]
    return rgb, depth, valid


def prepare():
    if OUTPUT.exists():
        raise FileExistsError(f"Preserve existing data; choose a new OUTPUT: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    shutil.copy2(__file__, OUTPUT / "generator_source.py")
    texture = make_texture()
    np.save(OUTPUT / "texture_linear_rgb.npy", texture)
    metadata = {"phase": "development", "seed": SEED, "units": "meters and seconds",
                "scope": "Designed textured surfaces; not physical granular validation",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "image_size": [WIDTH, HEIGHT], "focal_px": FOCAL_PX,
                "camera_centers_m": CAMERA_CENTERS.tolist(), "fps": FPS,
                "frame_ids": FRAME_IDS, "held_out_frame_ids": HELD_OUT_FRAME_IDS,
                "held_out_camera_ids": HELD_OUT_CAMERA_IDS, "exposure_seconds": EXPOSURE_SECONDS,
                "shutter_samples": SHUTTER_SAMPLES, "quadrature": "uniform midpoint",
                "translation_m_per_s": [0.35, 0.03, 0.20],
                "deformation": "0.03 m * sin(2*pi*(1.2 Hz*t + 0.8 cycles/m*x))",
                "illumination": "Unlit material texture, linear RGB; black outside mesh",
                "input_provenance": "StereoSGBM from cameras 0/1 at training times only; no exact mesh/velocity input",
                "stereo_num_disparities": 64, "max_points_per_frame": MAX_POINTS, "scenes": []}
    with open(OUTPUT / "protocol.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    for kind in SCENES:
        root = OUTPUT / kind
        truth = root / "evaluation_only"
        truth.mkdir(parents=True)
        configurations = {}
        for label, exposure in (("sharp", 0.), ("blur", EXPOSURE_SECONDS)):
            folder = root / label
            (folder / "images").mkdir(parents=True)
            (folder / "clouds").mkdir()
            configurations[label] = {"sfm": {"root_path": str((folder / "images").resolve()),
                "pixel_coordinates": "opencv_integer_centers", "views": [], "intrinsics": [{"key": 0,
                "value": {"ptr_wrapper": {"data": {"focal_length": FOCAL_PX,
                "principal_point": [(WIDTH - 1) / 2., (HEIGHT - 1) / 2.]}}}}], "extrinsics": []}, "views": {}}
            for camera_id, center in enumerate(CAMERA_CENTERS):
                configurations[label]["sfm"]["extrinsics"].append({"key": camera_id, "value": {
                    "rotation": np.eye(3).tolist(), "center": center.tolist()}})
        records = []
        for frame_idx in FRAME_IDS:
            timestamp = frame_idx / FPS
            vertices, triangles, uv = make_surface(kind, timestamp)
            sharp, depth, visible = render_mesh_views(vertices, triangles, uv, texture)
            np.savez_compressed(truth / f"frame_{frame_idx:06d}.npz", vertices=vertices, triangles=triangles,
                                uv=uv, depth_m=depth, visible=visible, timestamp=timestamp)
            blur = np.zeros_like(sharp)
            for sample in range(SHUTTER_SAMPLES):
                offset = EXPOSURE_SECONDS * ((sample + 0.5) / SHUTTER_SAMPLES - 0.5)
                geometry = make_surface(kind, timestamp + offset)
                rgb, _, _ = render_mesh_views(*geometry, texture)
                blur += rgb / SHUTTER_SAMPLES
            for label, image_stack in (("sharp", sharp), ("blur", blur)):
                exposure = 0. if label == "sharp" else EXPOSURE_SECONDS
                folder = root / label
                for camera_id, image in enumerate(image_stack):
                    filename = f"frame_{frame_idx:06d}_cam{camera_id}.png"
                    cv2.imwrite(str(folder / "images" / filename), np.rint(image[:, :, ::-1] * 255).astype(np.uint8))
                    view_id = frame_idx * len(CAMERA_CENTERS) + camera_id
                    configurations[label]["sfm"]["views"].append({"key": view_id, "value": {"ptr_wrapper": {"data": {
                        "filename": filename, "id_pose": camera_id, "id_intrinsic": 0}}}})
                    configurations[label]["views"][filename] = {"frame_idx": frame_idx, "camera_id": camera_id,
                        "timestamp": timestamp, "exposure": exposure}
                if frame_idx in HELD_OUT_FRAME_IDS:
                    continue
                # Read training PNGs back so stereo sees the same quantized pixels as training.
                left = cv2.imread(str(folder / "images" / f"frame_{frame_idx:06d}_cam0.png"))
                right = cv2.imread(str(folder / "images" / f"frame_{frame_idx:06d}_cam1.png"))
                calibration = {"cam0": np.array([[FOCAL_PX, 0., WIDTH / 2.], [0., FOCAL_PX, HEIGHT / 2.], [0., 0., 1.]]),
                               "baseline_m": BASELINE_M, "doffs": 0.}
                points, colors, disparity, valid, lr_error = reconstruct_rectified_pair(left, right, calibration, num_disparities=64)
                candidates = len(points)
                if candidates < 100:
                    raise ValueError(f"Insufficient stereo at {kind} {label} frame {frame_idx}: {candidates}")
                rng = np.random.default_rng(SEED + frame_idx)
                if candidates > MAX_POINTS:
                    keep = np.sort(rng.choice(candidates, MAX_POINTS, replace=False))
                    points, colors = points[keep], colors[keep]
                normals = estimate_normals(points)
                normals[np.sum(normals * points, axis=1) > 0] *= -1
                np.savez_compressed(folder / "clouds" / f"frame_{frame_idx:06d}.npz",
                                    points=points, colors=colors, normals=normals, time=timestamp)
                np.savez_compressed(folder / f"stereo_{frame_idx:06d}.npz", raw_disparity=disparity,
                                    lr_error=lr_error, valid=valid)
                records.append({"label": label, "frame": frame_idx, "stereo_candidates": candidates, "points": len(points)})
            print(f"{kind} frame {frame_idx}: sharp + {SHUTTER_SAMPLES}-sample blur", flush=True)
        for label, data in configurations.items():
            for filename, value in (("sfm.json", data["sfm"]), ("views_manifest.json", data["views"])):
                with open(root / label / filename, "w", encoding="utf-8") as f:
                    json.dump(value, f, indent=2)
        metadata["scenes"].append({"kind": kind, "input_records": records})
    with open(OUTPUT / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


if __name__ == "__main__":
    prepare()
