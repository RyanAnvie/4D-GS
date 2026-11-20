"""Prepare image-only ETH3D inputs. Never reads the reference disparity/masks."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from data_io import estimate_normals


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "benchmark_data/eth3d_raw/inputs"
OUTPUT = ROOT / "benchmark_data/eth3d_geometry_v1"
SEED = 42
MAX_POINTS = 12000
NUM_DISPARITIES = 128
BLOCK_SIZE = 5
UNIQUENESS_RATIO = 10
LR_TOLERANCE_PX = 1.0
DEVELOPMENT = ("delivery_area", "electro")
CALIBRATION = ("facade", "terrace")
TEST = ("forest", "playground", "terrains")


def read_stereo_calibration(path):
    values = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            if key in ("cam0", "cam1"):
                values[key] = np.fromstring(value.strip("[]").replace(";", " "), sep=" ").reshape(3, 3)
            else:
                values[key] = float(value)
    # Middlebury calib.txt baseline is mm. ETH3D uses this file format.
    values["baseline_m"] = values["baseline"] / 1000.0
    if values["baseline_m"] <= 0:
        raise ValueError("Stereo baseline must be positive.")
    return values


def reconstruct_rectified_pair(left, right, calibration, num_disparities=NUM_DISPARITIES,
                               horizontal_padding=0):
    """Left-image stereo points in meters; image-only LR filtering is explicit."""
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    count = min(num_disparities, ((left.shape[1] - 1) // 16) * 16)
    if horizontal_padding < 0 or int(horizontal_padding) != horizontal_padding:
        raise ValueError("Horizontal padding must be a nonnegative integer.")
    if horizontal_padding:
        gray_left = cv2.copyMakeBorder(gray_left, 0, 0, horizontal_padding, horizontal_padding, cv2.BORDER_REPLICATE)
        gray_right = cv2.copyMakeBorder(gray_right, 0, 0, horizontal_padding, horizontal_padding, cv2.BORDER_REPLICATE)
    parameters = dict(numDisparities=count, blockSize=BLOCK_SIZE,
                      P1=8 * BLOCK_SIZE ** 2, P2=32 * BLOCK_SIZE ** 2,
                      uniquenessRatio=UNIQUENESS_RATIO, speckleWindowSize=100,
                      speckleRange=2, disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    forward = cv2.StereoSGBM_create(minDisparity=0, **parameters)
    backward = cv2.StereoSGBM_create(minDisparity=-count, **parameters)
    disparity = forward.compute(gray_left, gray_right).astype(np.float32) / 16.0
    reverse = backward.compute(gray_right, gray_left).astype(np.float32) / 16.0
    if horizontal_padding:
        # The optimizer sees artificial border context, but accepted correspondences
        # and reconstructed coordinates are restricted to the original images.
        disparity = disparity[:, horizontal_padding:-horizontal_padding]
        reverse = reverse[:, horizontal_padding:-horizontal_padding]
    rows, columns = np.indices(disparity.shape)
    right_columns = np.rint(columns - disparity).astype(np.int32)
    inside = (right_columns >= 0) & (right_columns < left.shape[1])
    reverse_match = reverse[rows, np.clip(right_columns, 0, left.shape[1] - 1)]
    lr_error = np.abs(disparity + reverse_match)
    valid = inside & (disparity > 0) & (reverse_match > -count - 1)
    valid &= lr_error <= LR_TOLERANCE_PX
    denominator = disparity + calibration["doffs"]
    valid &= denominator > 0
    K = calibration["cam0"].copy()
    # ETH3D/COLMAP origin is the pixel corner; array centers have integer indices.
    K[:2, 2] -= 0.5
    depth = K[0, 0] * calibration["baseline_m"] / denominator[valid]
    x = (columns[valid] - K[0, 2]) * depth / K[0, 0]
    y = (rows[valid] - K[1, 2]) * depth / K[1, 1]
    points = np.column_stack((x, y, depth)).astype(np.float32)
    colors = left[valid][:, ::-1].astype(np.float32) / 255.0
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite reconstructed points; no silent filtering.")
    return points, colors, disparity, valid, lr_error


def prepare():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Choose a fresh OUTPUT; existing prepared data are preserved: {OUTPUT}")
    scenes = sorted(path for path in RAW.iterdir() if path.is_dir() and path.name.endswith("s"))
    if not scenes:
        raise FileNotFoundError(f"Extract the official two_view_training.7z to {RAW}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    records = []
    for scene in scenes:
        group = scene.name.rsplit("_", 1)[0]
        if group in DEVELOPMENT:
            split = "development"
        elif group in CALIBRATION:
            split = "calibration"
        elif group in TEST:
            split = "test"
        else:
            raise ValueError(f"Unregistered scene group: {group}")
        calibration = read_stereo_calibration(scene / "calib.txt")
        left = cv2.imread(str(scene / "im0.png"))
        right = cv2.imread(str(scene / "im1.png"))
        if left is None or right is None or left.shape != right.shape:
            raise ValueError(f"Invalid stereo images: {scene}")
        height, width = left.shape[:2]
        points, colors, disparity, valid, lr_error = reconstruct_rectified_pair(left, right, calibration)
        initial_count = len(points)
        if initial_count < 100:
            raise ValueError(f"Insufficient image-only stereo support: {scene}")
        if initial_count > MAX_POINTS:
            keep = np.sort(rng.choice(initial_count, MAX_POINTS, replace=False))
            points = points[keep]
            colors = colors[keep]
        normals = estimate_normals(points)
        normals[np.sum(normals * points, axis=1) > 0] *= -1
        folder = OUTPUT / scene.name
        cloud_dir = folder / "clouds"
        cloud_dir.mkdir(parents=True)
        np.savez_compressed(cloud_dir / "frame_000000.npz", points=points, colors=colors,
                            normals=normals, time=0.0)
        np.savez_compressed(folder / "stereo_diagnostics.npz", disparity=disparity,
                            input_valid=valid, lr_error=lr_error)
        sfm = {"root_path": str(scene.resolve()), "pixel_coordinates": "opencv_integer_centers",
               "views": [], "intrinsics": [], "extrinsics": []}
        manifest = {}
        for camera in (0, 1):
            K = calibration[f"cam{camera}"].copy()
            K[:2, 2] -= 0.5
            filename = f"im{camera}.png"
            sfm["views"].append({"key": camera, "value": {"ptr_wrapper": {"data": {
                "filename": filename, "id_pose": camera, "id_intrinsic": camera}}}})
            sfm["intrinsics"].append({"key": camera, "value": {"ptr_wrapper": {"data": {
                "focal_length": float(K[0, 0]), "focal_length_y": float(K[1, 1]),
                "principal_point": K[:2, 2].tolist()}}}})
            sfm["extrinsics"].append({"key": camera, "value": {
                "rotation": np.eye(3).tolist(), "center": [camera * calibration["baseline_m"], 0., 0.]}})
            manifest[filename] = {"camera_id": camera, "frame_idx": 0, "timestamp": 0., "exposure": 0.}
        for name, value in (("sfm.json", sfm), ("views_manifest.json", manifest)):
            with open(folder / name, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2)
        input_hashes = {}
        for name in ("im0.png", "im1.png", "calib.txt", "cameras.txt", "images.txt"):
            input_hashes[name] = hashlib.sha256((scene / name).read_bytes()).hexdigest()
        record = {"scene": scene.name, "group": group, "split": split, "image_size": [width, height],
                  "baseline_m": calibration["baseline_m"], "stereo_points": initial_count,
                  "initial_gaussians": len(points), "input_valid_fraction": float(valid.mean()),
                  "input_sha256": input_hashes}
        records.append(record)
        print(f"{scene.name} [{split}]: {initial_count} stereo points -> {len(points)} seeded samples", flush=True)
    archives = ROOT / "benchmark_data/eth3d_raw/downloads"
    archive_hashes = {}
    for name in ("two_view_training.7z", "two_view_training_gt.7z"):
        archive_hashes[name] = hashlib.sha256((archives / name).read_bytes()).hexdigest()
    metadata = {"source": "https://eth3d.ethz.ch/datasets", "coordinate_units": "meters",
                "coordinate_frame": "rectified left camera: right, down, forward",
                "pixel_coordinates": "OpenCV integer centers; released principal points minus 0.5",
                "reference_use": "NONE in preparation. Reference archive hash only; no disparity/mask reads.",
                "seed": SEED, "max_points": MAX_POINTS, "num_disparities": NUM_DISPARITIES,
                "block_size": BLOCK_SIZE, "uniqueness_ratio": UNIQUENESS_RATIO,
                "lr_tolerance_px": LR_TOLERANCE_PX, "depth_filter": "positive disparity+doffs only; no GT thresholds",
                "archive_sha256": archive_hashes, "scenes": records}
    with open(OUTPUT / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)
    return metadata


if __name__ == "__main__":
    prepare()
