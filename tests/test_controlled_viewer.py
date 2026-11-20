"""Latest-run loading and switching checks; no training or output changes."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from open3d.visualization import gui

import view_controlled_results as controlled
import view_pointclouds as view


def run_working_directory_checks():
    script = Path(controlled.__file__).resolve()
    code = """
import runpy
import sys
from pathlib import Path

script = sys.argv[1]
working_directory = Path.cwd()
for attempt in range(2):
    namespace = runpy.run_path(script, run_name='viewer_import_check')
    assert Path.cwd() == working_directory
    assert sys.path.count(str(Path(script).parent)) == 1
    assert 'VIEWER' not in namespace
    config, experiments = namespace['load_latest_results'](
        namespace['BENCHMARK_ROOT'], 'Deformation')
    assert Path(config.corrected_root) == experiments['Deformation']
    assert all(path.is_absolute() for path in experiments.values())
print('PASS: repeated script loading from an unrelated working directory.')
"""
    with tempfile.TemporaryDirectory(prefix="controlled_viewer_cwd_") as folder:
        result = subprocess.run(
            [sys.executable, "-I", "-c", code, str(script)],
            cwd=folder, capture_output=True, text=True, timeout=120,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())


def write_sequence_fixture(root, modified_at):
    root.mkdir(parents=True)
    records = []
    metrics = []
    for frame in (0, 2):
        filename = f"sequence_frame_{frame:06d}.ply"
        (root / filename).write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n"
            "property float y\nproperty float z\nend_header\n0 0 0\n", encoding="ascii")
        records.append({"frame_idx": frame, "timestamp": frame * .05, "file": filename})
        metrics.append({"frame": frame})
    for filename, data in (("sequence_manifest.json", {"frames": records}),
                           ("summary.json", {"frame_metrics": metrics})):
        path = root / filename
        path.write_text(json.dumps(data), encoding="utf-8")
        os.utime(path, (modified_at, modified_at))


def run_discovery_checks():
    with tempfile.TemporaryDirectory(prefix="controlled_viewer_") as folder:
        root = Path(folder)
        try:
            controlled.find_latest_experiments(root)
        except FileNotFoundError as error:
            assert "No complete translation/deformation pair" in str(error)
        else:
            raise AssertionError("An empty results folder must be reported.")
        old = root / "controlled_z_old"
        latest = root / "controlled_a_latest"
        for run, modified_at in ((old, 100), (latest, 200)):
            for kind in ("translation", "deformation"):
                write_sequence_fixture(run / kind / "tangent/independent_evaluation", modified_at)
        expected = {"Translation": latest / "translation/tangent/independent_evaluation",
                    "Deformation": latest / "deformation/tangent/independent_evaluation"}
        assert controlled.find_latest_experiments(root) == expected
        original_config = vars(view.IDE_CONFIG).copy()
        for default in ("Translation", "Deformation"):
            config, experiments = controlled.load_latest_results(root, default)
            assert experiments == expected
            assert Path(config.corrected_root) == expected[default]
            assert config.fps is None and config.continue_frame_root is None
            assert config.mode == "corrected" and config.show_colorbar
            assert config.color_mode == controlled.COLOR_MODE
            assert config.corrected_display == controlled.DISPLAY_MODE
            assert config.playback_speed == controlled.PLAYBACK_SPEED
            assert config.loop == controlled.LOOP
            assert config.width == controlled.WINDOW_WIDTH
            assert config.height == controlled.WINDOW_HEIGHT
            assert config.window_x == controlled.WINDOW_X
            assert config.window_y == controlled.WINDOW_Y
        assert vars(view.IDE_CONFIG) == original_config
        try:
            controlled.load_latest_results(root, "Unknown")
        except ValueError as error:
            assert "DEFAULT_EXPERIMENT" in str(error)
        else:
            raise AssertionError("An invalid initial experiment must be reported.")

        # A newer unmatched method must not be mixed with an older counterpart.
        unmatched = root / "controlled_unmatched"
        write_sequence_fixture(unmatched / "translation/adaptive/independent_evaluation", 300)
        write_sequence_fixture(unmatched / "deformation/tangent/independent_evaluation", 300)
        assert controlled.find_latest_experiments(root) == expected

        incomplete = root / "controlled_incomplete"
        for kind in ("translation", "deformation"):
            write_sequence_fixture(incomplete / kind / "tangent/independent_evaluation", 400)
        broken = incomplete / "deformation/tangent/independent_evaluation"
        summary_path = broken / "summary.json"
        summary_text = summary_path.read_text(encoding="utf-8")
        summary_path.unlink()
        assert controlled.find_latest_experiments(root) == expected
        summary_path.write_text("{", encoding="utf-8")
        assert controlled.find_latest_experiments(root) == expected
        summary_path.write_text(summary_text, encoding="utf-8")
        ply_path = broken / "sequence_frame_000002.ply"
        ply_text = ply_path.read_text(encoding="ascii")
        ply_path.unlink()
        assert controlled.find_latest_experiments(root) == expected
        ply_path.write_text("", encoding="ascii")
        assert controlled.find_latest_experiments(root) == expected
        ply_path.write_text(ply_text, encoding="ascii")
        summary_path.write_text(json.dumps({"frame_metrics": [{"frame": 0}]}), encoding="utf-8")
        assert controlled.find_latest_experiments(root) == expected
        summary_path.write_text(summary_text, encoding="utf-8")
        manifest_path = broken / "sequence_manifest.json"
        manifest_text = manifest_path.read_text(encoding="utf-8")
        for timestamp in (None, float("nan"), float("inf"), 0.0, -0.1):
            manifest = json.loads(manifest_text)
            manifest["frames"][1]["timestamp"] = timestamp
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            assert controlled.find_latest_experiments(root) == expected
        manifest = json.loads(manifest_text)
        del manifest["frames"][1]["timestamp"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        assert controlled.find_latest_experiments(root) == expected
        manifest_path.write_text(manifest_text, encoding="utf-8")
        selected = controlled.find_latest_experiments(root)
        assert selected["Deformation"] == broken
        assert selected["Translation"] == incomplete / "translation/tangent/independent_evaluation"
        config, experiments = controlled.load_latest_results(root, "Deformation")
        assert experiments == selected and Path(config.corrected_root) == broken
        nested = root / "controlled_nested"
        modified_at = max(path.stat().st_mtime for path in broken.iterdir()) + 1
        write_sequence_fixture(nested / "uncapped/translation/equal/fixed_on/independent_evaluation", modified_at)
        write_sequence_fixture(nested / "cap2/deformation/equal/fixed_on/independent_evaluation", modified_at)
        assert controlled.find_latest_experiments(root) == selected
        write_sequence_fixture(nested / "uncapped/deformation/equal/fixed_on/independent_evaluation", modified_at)
        pair = controlled.find_latest_experiments(root)
        assert pair == {"Translation":nested / "uncapped/translation/equal/fixed_on/independent_evaluation",
                        "Deformation":nested / "uncapped/deformation/equal/fixed_on/independent_evaluation"}
    print("PASS: newest matched pair, missing/malformed exports, valid timestamps, complete frame coverage and empty-folder error.")


def run_checks():
    config, experiments = controlled.load_latest_results(
        controlled.BENCHMARK_ROOT, controlled.DEFAULT_EXPERIMENT)
    assert experiments == controlled.find_latest_experiments(controlled.BENCHMARK_ROOT)
    hashes = {}
    for root in experiments.values():
        for path in sorted(root.glob("sequence_frame_*.ply")):
            hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    viewer = view.SequenceViewer(config, experiments=experiments)
    assert viewer.experiment_names == ["Translation", "Deformation"]
    assert viewer.experiment_index == viewer.experiment_names.index(controlled.DEFAULT_EXPERIMENT)
    assert viewer.window is None
    assert viewer.speed == controlled.PLAYBACK_SPEED and viewer.mode == "corrected"
    assert viewer.color_mode == controlled.COLOR_MODE
    assert viewer.display == controlled.DISPLAY_MODE
    assert viewer.loop == controlled.LOOP and viewer.args.show_colorbar
    config_before = vars(config).copy()
    viewer.color_mode = "uncertainty"
    viewer.display = "points"
    viewer.speed = .5
    viewer.loop = False
    switch_order = [1 - viewer.experiment_index, viewer.experiment_index]
    for index in switch_order * 2:
        viewer.index = 4
        viewer.playing = True
        viewer._on_experiment("", index)
        assert viewer.experiment_index == index
        assert viewer.corrected_root == experiments[viewer.experiment_names[index]].resolve()
        assert viewer.index == 0 and not viewer.playing
        assert viewer.color_mode == "uncertainty" and viewer.display == "points"
        assert viewer.speed == .5 and not viewer.loop
        assert viewer.frames == list(range(7))
        np.testing.assert_allclose(viewer.times, np.arange(7)*.05, rtol=0, atol=1e-15)
        assert viewer.has_shapes and viewer.color_modes == ["true", "confidence", "uncertainty"]
        with open(viewer.corrected_root / "summary.json", encoding="utf-8") as file:
            evaluation = json.load(file)
        exported_counts = {row["frame"]: row["active_gaussians"] for row in evaluation["frame_metrics"]}
        for frame in viewer.frames:
            vertex = viewer._vertex(frame)
            count = len(vertex["x"])
            assert count == exported_counts[frame] and viewer.counts[frame] == (count, count)
            assert np.isfinite(view._scalar_values(vertex, "confidence")).all()
            assert np.isfinite(view._scalar_values(vertex, "uncertainty")).all()
        old_cache = viewer.cache
        viewer.index = 3
        viewer._on_experiment("", index)
        assert viewer.index == 3 and viewer.cache is old_cache
    assert vars(config) == config_before
    for path, digest in hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert len(hashes) == 14
    print("PASS: latest defaults, both experiments, repeated switches, timestamps, counts, colors and immutable PLYs/config.")


def run_gui_checks():
    for attempt in range(2):
        default = ("Translation", "Deformation")[attempt]
        config, experiments = controlled.load_latest_results(
            controlled.BENCHMARK_ROOT, default)
        viewer = view.SequenceViewer(config, experiments=experiments)
        viewer.create_window()
        app = gui.Application.instance
        for _ in range(5):
            assert app.run_one_tick()
        assert viewer.experiment_box.number_of_items == 2
        for index in (1 - viewer.experiment_index, viewer.experiment_index):
            viewer._on_color("", 2)
            viewer._on_display("", 1)
            viewer._on_speed("", viewer.speeds.index(.5))
            viewer.axes_box.checked = True
            viewer._on_axes(True)
            viewer._toggle_play()
            viewer._on_experiment("", index)
            assert app.run_one_tick()
            assert viewer.experiment_box.selected_index == index
            assert viewer.color_box.selected_index == 2
            assert viewer.display_box.enabled and viewer.scene.scene.has_geometry("axes")
            assert viewer.slider.int_value == 0 and viewer.play_button.text == ">"
            assert not viewer.playing and viewer.speed == .5
            assert viewer.legend_title.visible and viewer.legend_title.text == "Opacity variance"
            for color in range(3):
                viewer.color_box.selected_index = color
                viewer._on_color("", color)
                assert app.run_one_tick()
            viewer._on_display("", 0)
            viewer._on_seek(0)
            viewer._toggle_play()
            viewer.play_started = time.perf_counter()-.22
            viewer._on_tick()
            assert viewer.index > 0 and viewer.playing
            viewer._on_seek(6)
            assert viewer.index == 6 and not viewer.playing
        viewer.window.close()
        while app.run_one_tick():
            pass
        assert viewer.closed
        print(f"PASS: native experiment switch, colorbar, geometry, axes, play/seek/close/reopen, attempt {attempt+1}.")


if __name__ == "__main__":
    run_working_directory_checks()
    run_discovery_checks()
    run_checks()
