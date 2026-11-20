"""Train-only feature tracks and fixed-camera multi-view triangulation."""

from pathlib import Path
import struct

import numpy as np
from scipy.spatial.transform import Rotation


def merge_feature_tracks(edges):
    """Merge consistent observations; reject edges that duplicate an image in a track."""
    assignments, tracks = {}, []
    conflicts = 0
    for first, second in edges:
        left = assignments.get(first)
        right = assignments.get(second)
        if first[0] == second[0]:
            raise ValueError("A matching edge must connect different images.")
        if left is None and right is None:
            index = len(tracks)
            tracks.append({first[0]: first[1], second[0]: second[1]})
            assignments[first] = index
            assignments[second] = index
        elif left is None or right is None:
            index = right if left is None else left
            observation = first if left is None else second
            if observation[0] in tracks[index]:
                conflicts += 1
                continue
            tracks[index][observation[0]] = observation[1]
            assignments[observation] = index
        elif left != right:
            if set(tracks[left]).intersection(tracks[right]):
                conflicts += 1
                continue
            if len(tracks[left]) < len(tracks[right]):
                left, right = right, left
            for image, feature in tracks[right].items():
                tracks[left][image] = feature
                assignments[(image, feature)] = left
            tracks[right] = {}
    return [sorted(track.items()) for track in tracks if track], conflicts


def triangulate_feature_track(pixels, poses):
    """All-view homogeneous DLT, followed by explicit depth and pixel diagnostics."""
    rows = []
    for pixel, pose in zip(pixels, poses):
        ray = np.linalg.solve(pose["K"], [pixel[0], pixel[1], 1.])
        P = np.column_stack((pose["R"], pose["t"]))
        rows.append(ray[0] * P[2] - P[0])
        rows.append(ray[1] * P[2] - P[1])
    _, _, vectors = np.linalg.svd(np.asarray(rows))
    homogeneous = vectors[-1]
    if abs(homogeneous[3]) <= 1e-12:
        return np.full(3, np.nan), np.full(len(poses), np.nan), np.full(len(poses), np.inf)
    point = homogeneous[:3] / homogeneous[3]
    depths, errors = [], []
    for pixel, pose in zip(pixels, poses):
        camera = pose["R"] @ point + pose["t"]
        depths.append(camera[2])
        projected = pose["K"] @ camera
        if abs(projected[2]) <= 1e-12:
            errors.append(np.inf)
        else:
            errors.append(float(np.linalg.norm(projected[:2] / projected[2] - pixel)))
    return point, np.asarray(depths), np.asarray(errors)


def read_openmvs_roundtrip(folder):
    """Read the installed importer's PINHOLE binary export, preserving its pixel origin."""
    folder = Path(folder)
    cameras = {}
    with open(folder / "cameras.bin", "rb") as file:
        count = struct.unpack("<Q", file.read(8))[0]
        for _ in range(count):
            camera_id, model, width, height = struct.unpack("<IiQQ", file.read(24))
            if model != 1:
                raise ValueError("Expected PINHOLE round-trip calibration.")
            fx, fy, cx, cy = struct.unpack("<4d", file.read(32))
            cameras[camera_id] = {"K": np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]),
                                  "width": width, "height": height}
        if file.tell() != (folder / "cameras.bin").stat().st_size:
            raise ValueError("Unexpected binary camera payload.")
    images = {}
    with open(folder / "images.bin", "rb") as file:
        count = struct.unpack("<Q", file.read(8))[0]
        for _ in range(count):
            values = struct.unpack("<I7dI", file.read(64))
            q = np.array(values[1:5])
            name = bytearray()
            while True:
                character = file.read(1)
                if not character:
                    raise ValueError("Truncated binary image name.")
                if character == b"\0":
                    break
                name.extend(character)
            observations = struct.unpack("<Q", file.read(8))[0]
            file.seek(observations * 24, 1)
            pose = cameras[values[-1]].copy()
            pose.update({"image_id": values[0], "R": Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(),
                         "t": np.array(values[5:8])})
            images[Path(name.decode("utf-8")).stem] = pose
        if file.tell() != (folder / "images.bin").stat().st_size:
            raise ValueError("Truncated or unexpected binary image payload.")
    return images


def read_openmvs_depth(path):
    """Read the documented OpenMVS2.2 DR header and depth, retaining zero missing values."""
    path = Path(path)
    with open(path, "rb") as file:
        signature, flags, _, image_width, image_height, width, height, dmin, dmax = struct.unpack("<2sBB4I2f", file.read(28))
        if signature != b"DR" or not flags & 1:
            raise ValueError("Not an OpenMVS depth payload.")
        name_size = struct.unpack("<H", file.read(2))[0]
        name = file.read(name_size).decode("utf-8")
        id_count = struct.unpack("<I", file.read(4))[0]
        ids = np.frombuffer(file.read(id_count * 4), dtype="<u4").copy()
        K = np.frombuffer(file.read(72), dtype="<f8").reshape(3, 3).copy()
        R = np.frombuffer(file.read(72), dtype="<f8").reshape(3, 3).copy()
        center = np.frombuffer(file.read(24), dtype="<f8").copy()
        depth = np.fromfile(file, dtype="<f4", count=width * height).reshape(height, width)
        remaining_bytes = 0
        if flags & 2:
            remaining_bytes += width * height * 12
        if flags & 4:
            remaining_bytes += width * height * 4
        if flags & 8:
            remaining_bytes += width * height * 4
        if file.tell() + remaining_bytes != path.stat().st_size:
            raise ValueError("Unexpected depth-map size or truncated additional attributes.")
    return {"name": name, "ids": ids, "K": K, "R": R, "center": center, "depth": depth,
            "image_size": (image_width, image_height), "depth_range": (dmin, dmax), "flags": flags}


def read_openmvs_visibility(path, selected_indices, expected_count, image_count):
    """Return CSR observations for sorted selected vertices; view IDs remain zero-based."""
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if selected_indices.ndim != 1 or not len(selected_indices):
        raise ValueError("Nonempty selected indices are required.")
    if selected_indices[0] < 0 or selected_indices[-1] >= expected_count or np.any(np.diff(selected_indices) <= 0):
        raise ValueError("Selected vertex indices must be sorted, unique and in range.")
    offsets, views = [0], []
    selected = 0
    with open(path, "rb") as file:
        count = struct.unpack("<Q", file.read(8))[0]
        if count != expected_count:
            raise ValueError("Visibility and vertex counts disagree.")
        for index in range(count):
            view_count = struct.unpack("<I", file.read(4))[0]
            if view_count > image_count:
                raise ValueError("Invalid number of supporting cameras.")
            if selected < len(selected_indices) and index == selected_indices[selected]:
                values = np.frombuffer(file.read(view_count * 4), dtype="<u4").copy()
                if len(values) != view_count or np.any(values >= image_count) or len(np.unique(values)) != view_count:
                    raise ValueError("Invalid or truncated visibility indices.")
                views.extend(values.tolist())
                offsets.append(len(views))
                selected += 1
            else:
                file.seek(view_count * 4, 1)
        if file.tell() != Path(path).stat().st_size or selected != len(selected_indices):
            raise ValueError("Truncated or trailing visibility payload.")
    return np.asarray(offsets, dtype=np.int64), np.asarray(views, dtype=np.int32)
