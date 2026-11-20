"""Repository-relative paths shared by the Spyder examples."""

from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "examples/results/controlled_demo"


def load_training_values(experiment):
    if experiment not in ("translation", "deformation"):
        raise ValueError("Choose translation or deformation.")
    run = RESULTS / experiment / "sensitivity/fixed_on"
    with open(run / "train_config.json", encoding="utf-8") as file:
        values = json.load(file)
    for name in ("sfm_json", "sequence_cloud", "views_manifest", "surface_origin_path", "output"):
        values[name] = str(ROOT / values[name])
    return values


def new_output(label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "outputs" / (label + "_" + timestamp)
    output.mkdir(parents=True)
    return output
