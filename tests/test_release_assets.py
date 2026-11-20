"""Check portable inputs, byte-preserved results and the complete demo video."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from data_io import load_openmvg_sequence
from ply_utils import read_vertex_ply


def run_checks():
    with open(ROOT / "docs/source_manifest.json", encoding="utf-8") as file:
        manifest = json.load(file)
    for row in manifest["result_assets"]:
        path = ROOT / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path

    for experiment in ("translation", "deformation"):
        data = ROOT / "examples/data" / experiment
        for directory, count in ((data, 16), (data / "evaluation", 42)):
            cameras = load_openmvg_sequence(str(directory / "sfm.json"),
                views_manifest=str(directory / "views_manifest.json"), device="cpu")
            assert len(cameras) == count
            assert len({(cam.frame_idx, cam.camera_id) for cam in cameras}) == count
        results = ROOT / "examples/results/controlled_demo" / experiment / "sensitivity/fixed_on"
        sequence = results / "independent_evaluation"
        with open(sequence / "sequence_manifest.json", encoding="utf-8") as file:
            frames = json.load(file)["frames"]
        assert [row["frame_idx"] for row in frames] == list(range(7))
        np.testing.assert_allclose([row["timestamp"] for row in frames], np.arange(7) * 0.05)
        for row in frames:
            vertex = read_vertex_ply(str(sequence / row["file"]))
            xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"]))
            assert xyz.shape == (12000, 3)
            assert len(np.unique(xyz, axis=0)) == 12000
            assert np.isfinite(xyz).all()
            for name in ("opacity", "opacity_concentration", "opacity_variance"):
                assert np.isfinite(vertex[name]).all()
            expected = vertex["opacity"] * (1.0 - vertex["opacity"])
            expected = expected / (vertex["opacity_concentration"] + 1.0)
            np.testing.assert_allclose(vertex["opacity_variance"], expected, rtol=2e-6)

    with open(ROOT / "media/video_manifest.json", encoding="utf-8") as file:
        video = json.load(file)
    capture = cv2.VideoCapture(str(ROOT / "media/4dgs_sequence.mp4"))
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FPS) == video["fps"]
    count = 0
    first = None
    changed = False
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape == (video["height"], video["width"], 3)
        if first is None:
            first = frame.copy()
        else:
            changed = changed or not np.array_equal(frame, first)
        # Inspect the six rendering panels, excluding headings and colorbars.
        for y in (170, 580):
            for x in (24, 496, 968):
                panel = frame[y:y + 250, x:x + 448]
                assert float(panel.std()) > 5.0
        count += 1
    capture.release()
    assert count == video["frames"] == 252
    assert changed
    print("PASS: asset hashes, both camera datasets, 14 Gaussian frames and all 252 video frames.")


if __name__ == "__main__":
    run_checks()
