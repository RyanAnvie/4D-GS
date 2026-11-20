"""Spyder F5: run the packaged regression tests in fresh Python processes."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from demo_config import new_output


def main():
    tests = sorted((ROOT / "tests").glob("test_*.py"))
    if not tests:
        raise FileNotFoundError("No test scripts found in " + str(ROOT / "tests"))
    output = new_output("checks")
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_THREADING_LAYER="SEQUENTIAL")
    results = []
    for path in tests:
        with open(output / (path.stem + ".log"), "w", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, "-X", "faulthandler", "-u", str(path)],
                cwd=output, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, check=False)
        results.append({"test": path.name, "returncode": result.returncode})
        print(path.name, "PASS" if result.returncode == 0 else "FAIL", flush=True)
    passed = all(row["returncode"] == 0 for row in results)
    with open(output / "summary.json", "w", encoding="utf-8") as file:
        json.dump({"passed": passed, "python": sys.executable, "tests": results}, file, indent=2)
    if not passed:
        raise RuntimeError("Check the individual test logs in " + str(output))
    print("All", len(results), "tests passed.", output)
    return output


if __name__ == "__main__":
    OUTPUT = main()
