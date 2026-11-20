"""Press F5 in Spyder to view the latest completed controlled experiment pair.

The Experiment menu switches between Translation and Deformation. The Color
menu selects RGB, confidence or uncertainty; the timeline controls playback.
"""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

# Spyder's working directory need not be the directory containing this script.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import view_pointclouds


# %% Spyder settings
BENCHMARK_ROOT = ROOT / "examples/results"
# Edit these values, then press F5 in Spyder. No command-line arguments are needed.
DEFAULT_EXPERIMENT = "Translation"  # "Translation" or "Deformation"
COLOR_MODE = "true"  # "true" (RGB), "confidence", "uncertainty"
DISPLAY_MODE = "gaussians"  # "gaussians" (ellipsoids) or "points" (centers)
PLAYBACK_SPEED = 0.25
LOOP = True
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 720
WINDOW_X = 40
WINDOW_Y = 40


def sequence_is_complete(root):
    """Require the finished evaluation and all frames referenced by its manifest."""
    manifest_path = root / "sequence_manifest.json"
    summary_path = root / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return False
    try:
        with open(manifest_path, encoding="utf-8") as file:
            manifest = json.load(file)
        with open(summary_path, encoding="utf-8") as file:
            summary = json.load(file)
        records = manifest["frames"]
        frames = [int(row["frame_idx"]) for row in records]
        timestamps = [float(row["timestamp"]) for row in records]
        evaluated_frames = [int(row["frame"]) for row in summary["frame_metrics"]]
        filenames = [row["file"] for row in records]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"Skipping incomplete sequence metadata: {root} ({error})")
        return False
    if not frames or len(set(frames)) != len(frames):
        return False
    if sorted(frames) != sorted(evaluated_frames):
        return False
    ordered_times = [timestamp for _, timestamp in sorted(zip(frames, timestamps))]
    if not all(math.isfinite(timestamp) for timestamp in ordered_times):
        return False
    for previous, current in zip(ordered_times, ordered_times[1:]):
        if current <= previous:
            return False
    expected_files = [f"sequence_frame_{frame:06d}.ply" for frame in frames]
    if filenames != expected_files:
        return False
    if set(expected_files) != {path.name for path in root.glob("sequence_frame_*.ply")}:
        return False
    for filename in expected_files:
        path = root / filename
        if not path.is_file() or path.stat().st_size == 0:
            return False
    return True


def find_latest_experiments(benchmark_root):
    """Select a completed pair with the same run and method, by export-file mtime."""
    candidates = []
    for run_root in sorted(Path(benchmark_root).glob("controlled_*")):
        for translation_root in sorted(run_root.rglob("translation")):
            if not translation_root.is_dir():
                continue
            for manifest in translation_root.rglob("sequence_manifest.json"):
                translation = manifest.parent
                if translation.name != "independent_evaluation":
                    continue
                method_path = translation.relative_to(translation_root)
                deformation = translation_root.parent / "deformation" / method_path
                completion_files = [manifest, translation / "summary.json",
                                    deformation / "sequence_manifest.json", deformation / "summary.json"]
                if not all(path.is_file() for path in completion_files):
                    continue
                completed_at = max(path.stat().st_mtime_ns for path in completion_files)
                candidates.append((completed_at, str(translation), translation, deformation))
    for _, _, translation, deformation in sorted(candidates, reverse=True):
        if sequence_is_complete(translation) and sequence_is_complete(deformation):
            return {"Translation": translation, "Deformation": deformation}
        print(f"Skipping incomplete experiment pair: {translation}")
    raise FileNotFoundError(
        f"No complete translation/deformation pair found in {benchmark_root}. "
        "Each independent_evaluation folder needs sequence_manifest.json, "
        "summary.json and every exported sequence_frame_*.ply."
    )


def load_latest_results(benchmark_root, default_experiment):
    """Resolve a fresh matched pair without opening a window or changing exports."""
    if default_experiment not in ("Translation", "Deformation"):
        raise ValueError("DEFAULT_EXPERIMENT must be 'Translation' or 'Deformation'.")
    experiments = find_latest_experiments(benchmark_root)
    config = SimpleNamespace(**vars(view_pointclouds.IDE_CONFIG))
    config.corrected_root = str(experiments[default_experiment])
    config.continue_frame_root = None
    config.mode = "corrected"
    config.color_mode = COLOR_MODE
    config.corrected_display = DISPLAY_MODE
    config.fps = None  # Exact observed and withheld timestamps come from the manifest.
    config.playback_speed = PLAYBACK_SPEED  # Physical timestamps remain unchanged.
    config.loop = LOOP
    config.show_colorbar = True
    config.width = WINDOW_WIDTH
    config.height = WINDOW_HEIGHT
    config.window_x = WINDOW_X
    config.window_y = WINDOW_Y
    return config, experiments


# %% Open viewer (F5 runs the complete file)
if __name__ == "__main__":
    # Resolve again on every F5 run; importing this module does not read results.
    CONFIG, EXPERIMENTS = load_latest_results(BENCHMARK_ROOT, DEFAULT_EXPERIMENT)
    RESULT = EXPERIMENTS[DEFAULT_EXPERIMENT]
    RESULT_ROOT = BENCHMARK_ROOT / RESULT.relative_to(BENCHMARK_ROOT).parts[0]
    print(f"Latest completed result pair: {RESULT_ROOT}")
    for name, path in EXPERIMENTS.items():
        print(f"  {name}: {path}")
    print(f"Opening: {DEFAULT_EXPERIMENT}")
    print("Latest means most recently exported, not best-performing or validated.")
    print("Experiment: Translation = rigid translation; Deformation = changing surface shape.")
    VIEWER = view_pointclouds.SequenceViewer(CONFIG, experiments=EXPERIMENTS)
    VIEWER.run()
