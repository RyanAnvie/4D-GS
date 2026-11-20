"""Focused viewer checks. Run directly in envtorch; no GUI is opened by default."""

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from open3d.visualization import gui

import view_pointclouds as view
from ply_utils import write_vertex_ply


def run_checks():
    assert view.DEFAULT_EXPERIMENT == "Surface-constrained"
    assert list(view.PUBLIC_EXPERIMENTS) == ["Adaptive densification", "Surface-constrained"]
    assert Path(view.IDE_CONFIG.corrected_root) == view.PUBLIC_EXPERIMENTS[view.DEFAULT_EXPERIMENT]
    assert view.PUBLIC_EXPERIMENTS["Adaptive densification"].name == "dense_adaptive"
    assert view.PUBLIC_EXPERIMENTS["Surface-constrained"].name == "surface_adaptive"
    assert all(path.is_absolute() for path in view.PUBLIC_EXPERIMENTS.values())
    assert view.IDE_CONFIG.mode == "corrected" and view.IDE_CONFIG.color_mode == "true"
    assert view.IDE_CONFIG.corrected_display == "gaussians" and view.IDE_CONFIG.show_colorbar
    assert view.IDE_CONFIG.fps is None and view.IDE_CONFIG.playback_speed == 0.25

    # A 90-degree rotation about +Z preserves the [w,x,y,z] convention.
    q = np.asarray([[1, 0, 0, 0], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]])
    rotations = view._quaternion_matrices(q)
    np.testing.assert_allclose(rotations[0], np.eye(3))
    np.testing.assert_allclose(rotations[1] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    values = np.asarray([0.0, 0.25, 0.5, 1.0, np.nan])
    normalized, lo, hi = view._normalize_scalar(values, (0, 1))
    np.testing.assert_allclose(normalized[:4], values[:4])
    assert (lo, hi) == (0, 1)
    normalized, lo, hi = view._normalize_scalar(np.asarray([4.0, 4.0]))
    np.testing.assert_allclose(normalized, [0.5, 0.5])
    assert (lo, hi) == (4.0, 4.0)

    # Nonuniform, nonzero timestamps must drive playback without assumed 30 fps.
    times = np.asarray([2.0, 2.1, 2.4])
    assert view._playback_index(times, 2.05, False) == (0, False)
    assert view._playback_index(times, 2.15, False) == (1, False)
    assert view._playback_index(times, 2.45, False) == (2, False)
    assert view._playback_index(times, 2.71, False) == (2, True)
    assert view._playback_index(times, 2.71, True) == (0, False)
    assert view._playback_index(np.asarray([2.0]), 9.0, True) == (0, True)

    vertex = {
        "x": np.asarray([0, 1, 2], dtype=np.float32),
        "y": np.zeros(3, dtype=np.float32), "z": np.zeros(3, dtype=np.float32),
        "scale_0": np.ones(3, dtype=np.float32),
        "scale_1": np.full(3, 0.5, dtype=np.float32),
        "scale_2": np.full(3, 0.1, dtype=np.float32),
        "rot_0": np.ones(3, dtype=np.float32),
        "rot_1": np.zeros(3, dtype=np.float32),
        "rot_2": np.zeros(3, dtype=np.float32),
        "rot_3": np.zeros(3, dtype=np.float32),
        "red": np.full(3, 200, dtype=np.uint8),
        "green": np.full(3, 100, dtype=np.uint8),
        "blue": np.full(3, 50, dtype=np.uint8),
        "opacity": np.asarray([0.1, 0.5, 1], dtype=np.float32),
        "opacity_concentration": np.asarray([2, 4, 8], dtype=np.float32),
        "opacity_variance": np.asarray([0.01, 0.02, 0.03], dtype=np.float32),
    }
    before = {name: value.copy() for name, value in vertex.items()}
    keep = view._display_mask(vertex, 0.2, 2)
    np.testing.assert_array_equal(keep, [False, True, True])
    for mode in ("true", "confidence", "uncertainty"):
        cloud = view._filter_corrected("", 0.2, 2, mode, vertex=vertex)
        assert len(cloud.points) == 2
        ellipsoids = view._gaussian_wireframes("", 0.2, 2, 1, 8, mode, vertex=vertex)
        assert len(ellipsoids.points) == 2 * 3 * 8
        assert len(ellipsoids.lines) == 2 * 3 * 8
    for name in before:
        np.testing.assert_array_equal(vertex[name], before[name])

    # Fixed range: identical concentration receives identical color across frames.
    colors = view._scalar_colors(vertex, np.ones(3, dtype=bool), "confidence", (0, 3))
    subset = {name: value[1:2] for name, value in vertex.items()}
    same_color = view._scalar_colors(subset, np.asarray([True]), "confidence", (0, 3))
    np.testing.assert_array_equal(colors[1], same_color[0])
    invalid = dict(vertex)
    invalid["opacity_concentration"] = np.asarray([np.nan, -1, 8])
    colors = view._scalar_colors(invalid, np.ones(3, dtype=bool), "confidence", (0, 3))
    np.testing.assert_array_equal(colors[:2], np.full((2, 3), 0.5))

    with tempfile.TemporaryDirectory(prefix="4dgs_viewer_test_") as directory:
        root = Path(directory)
        for frame in (10, 2):
            write_vertex_ply(str(root / f"corrected_frame_{frame}.ply"), vertex, text=False)
        config = SimpleNamespace(**vars(view.IDE_CONFIG))
        config.corrected_root = str(root)
        config.fps = 20.0
        viewer = view.SequenceViewer(config)
        assert viewer.frames == [2, 10]
        assert not viewer.has_originals
        np.testing.assert_allclose(viewer.times, [0.1, 0.5])
        np.savez(root / "dynamic_gaussians_final.npz", frame_indices=[2, 2, 10, 10],
                 t_ref=[3.0, 3.0, 3.4, 3.4])
        viewer = view.SequenceViewer(config)
        np.testing.assert_allclose(viewer.times, [3.0, 3.4])
        assert viewer.scalar_ranges["confidence"] == (np.log(2), np.log(8))
        original_root = root / "originals"
        for frame in (2, 10):
            (original_root / str(frame)).mkdir(parents=True)
            write_vertex_ply(str(original_root / str(frame) / "robust_scene_dense.ply"), vertex, text=False)
        config.continue_frame_root = str(original_root)
        assert view.SequenceViewer(config).has_originals
        np.savez(root / "dynamic_gaussians_final.npz", frame_indices=[2, 10], t_ref=[1, 0])
        try:
            view.SequenceViewer(config)
        except ValueError as error:
            assert "strictly increasing" in str(error)
        else:
            raise AssertionError("Non-increasing timestamps were accepted.")

    with tempfile.TemporaryDirectory(prefix="4dgs_evaluated_sequence_") as directory:
        root = Path(directory)
        for frame in (0, 1, 2):
            write_vertex_ply(str(root / f"sequence_frame_{frame:06d}.ply"), vertex, text=False)
        with open(root / "sequence_manifest.json", "w", encoding="utf-8") as f:
            json.dump({"frames": [{"frame_idx": 0, "timestamp": 2.}, {"frame_idx": 1, "timestamp": 2.05},
                                  {"frame_idx": 2, "timestamp": 2.1}]}, f)
        config = SimpleNamespace(**vars(view.IDE_CONFIG))
        config.corrected_root = str(root)
        config.fps = None
        viewer = view.SequenceViewer(config)
        assert viewer.frames == [0, 1, 2]
        np.testing.assert_allclose(viewer.times, [2., 2.05, 2.1])

    # Keep the bundled regression fixture independent of external public data.
    config = SimpleNamespace(**vars(view.IDE_CONFIG))
    config.corrected_root = str(
        ROOT / "examples/results/controlled_demo/translation/sensitivity/fixed_on/independent_evaluation")
    viewer = view.SequenceViewer(config)
    assert viewer.frames == list(range(7))
    np.testing.assert_allclose(viewer.times, np.arange(7) * 0.05)
    with open(viewer.corrected_root / "summary.json", encoding="utf-8") as f:
        counts = {row["frame"]: row["active_gaussians"] for row in json.load(f)["frame_metrics"]}
    for frame in viewer.frames:
        expected = counts[frame]
        assert viewer.counts[frame] == (expected, expected)
    assert not viewer.has_originals
    print("PASS: geometry, color ranges, masks, timeline, original-cloud fixtures, actual seven-frame demo.")


def run_public_checks():
    """Inspect both saved public runs when the local benchmark data is present."""
    for root in view.PUBLIC_EXPERIMENTS.values():
        if not any(root.glob("corrected_frame_*.ply")):
            print(f"SKIP: optional public-result checks; no PLY sequence in {root}")
            return
    config_before = vars(view.IDE_CONFIG).copy()
    viewer = view.SequenceViewer(view.IDE_CONFIG, experiments=view.PUBLIC_EXPERIMENTS)
    assert viewer.experiment_names[viewer.experiment_index] == view.DEFAULT_EXPERIMENT
    # Switch repeatedly and preserve the selected color, geometry and playback speed.
    for attempt, index in enumerate((1, 0, 1)):
        viewer._on_experiment("", index)
        assert viewer.corrected_root == view.PUBLIC_EXPERIMENTS[viewer.experiment_names[index]].resolve()
        assert viewer.frames == list(range(30))
        np.testing.assert_allclose(viewer.times, np.arange(30) / 30, rtol=0, atol=1e-6)
        assert viewer.has_shapes and viewer.color_modes == ["true", "confidence", "uncertainty"]
        if attempt == 0:
            assert viewer.color_mode == "true" and viewer.display == "gaussians"
        else:
            assert viewer.color_mode == "uncertainty" and viewer.display == "points"
        assert viewer.speed == 0.25 and viewer.loop
        assert not viewer.playing and viewer.index == 0
        for frame in viewer.frames:
            assert viewer.counts[frame] == (100000, 100000)
        vertex = viewer._vertex(viewer.frames[0])
        keep = view._display_mask(vertex, viewer.args.min_opacity, viewer.args.max_scale)
        for mode in ("true", "confidence", "uncertainty"):
            cloud = view._filter_corrected("", 0.0, 1e9, mode, vertex=vertex,
                                           value_range=viewer.scalar_ranges.get(mode))
            assert len(cloud.points) == 100000
            assert np.isfinite(np.asarray(cloud.colors)).all()
            if mode != "true":
                values = view._scalar_values(vertex, mode)[keep]
                assert np.isfinite(values).all()
                lo, hi = viewer.scalar_ranges[mode]
                assert lo <= values.min() <= values.max() <= hi
        ellipsoids = view._gaussian_wireframes(
            "", 0.0, 1e9, 1.0, viewer.args.circle_segments, "true", vertex=vertex)
        assert len(ellipsoids.points) == 100000 * 3 * viewer.args.circle_segments
        assert np.isfinite(np.asarray(ellipsoids.points)).all()
        del ellipsoids
        assert view._playback_index(viewer.times, 0.51, False) == (15, False)
        assert view._playback_index(viewer.times, 1.01, False) == (29, True)
        assert view._playback_index(viewer.times, 1.01, True) == (0, False)
        viewer.color_mode = "uncertainty"
        viewer.display = "points"
    assert vars(view.IDE_CONFIG) == config_before
    print("PASS: cook_spinach defaults, both 30-frame runs, 100,000 Gaussians/frame, all colors and switching.")


def run_gui_checks():
    """Optional native-window checks; exercises close/reopen in one Python process."""
    for attempt in range(2):
        config = SimpleNamespace(**vars(view.IDE_CONFIG))
        config.width = 1000
        config.height = 700
        viewer = view.SequenceViewer(config, experiments=view.PUBLIC_EXPERIMENTS)
        viewer.create_window()
        app = gui.Application.instance
        for _ in range(5):
            assert app.run_one_tick()
        camera_before = viewer.scene.scene.camera.get_view_matrix().copy()
        viewer._on_seek(3)
        assert viewer.index == 3
        for index in range(3):
            viewer._on_color("", index)
            assert app.run_one_tick()
        viewer._on_display("", 1)
        if viewer.has_originals:
            viewer._on_mode("", 2)
            assert app.run_one_tick()
            viewer._on_mode("", 0)
        viewer._on_display("", 0)
        np.testing.assert_array_equal(camera_before, viewer.scene.scene.camera.get_view_matrix())
        viewer._on_seek(0)
        viewer._toggle_play()
        viewer.play_started = time.perf_counter() - 0.2
        viewer._on_tick()
        assert viewer.index > 0 and viewer.playing
        viewer._on_seek(5)
        assert viewer.index == 5 and not viewer.playing
        viewer.loop = False
        viewer._toggle_play()
        viewer.play_started = time.perf_counter() - 2.0 / viewer.speed
        viewer._on_tick()
        assert viewer.index == len(viewer.frames) - 1 and not viewer.playing
        viewer.window.close()
        while app.run_one_tick():
            pass
        assert viewer.closed
        print(f"PASS: native GUI open/seek/color/geometry/play/stop/close, attempt {attempt + 1}.")


if __name__ == "__main__":
    run_checks()
    run_public_checks()
