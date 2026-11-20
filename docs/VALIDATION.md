# Local Validation

Executed on 2026-11-20 using the existing Windows `envtorch` environment:
Python 3.10.18, PyTorch 2.8.0+cu129 and an NVIDIA RTX 4080 SUPER.

## Executed Checks

| Check | Result |
|---|---|
| `run_checks.py` from outside the repository working directory | All 14 test scripts passed |
| Relocated `tests/` suite | All 14 tests passed from a separate output working directory |
| Spyder kernel `%runfile` | Unified runner, direct public-video test, and repeated geometry-calibration test passed with changed working directories |
| Default translation quick training | 120 iterations, all five stages, four split events, model and PLY export completed |
| Deformation quick training | 120 iterations, all five stages, four split events, model and PLY export completed |
| Both quick-training outputs | 48,000 stored Gaussians across four reference frames, 12,000 per reference frame |
| `evaluate_results.py` | Both saved models evaluated at seven timestamps and 42 images each |
| Evaluation-image PSNR, SSIM and MAE | Matched the bundled numerical summaries |
| Seen-time and unseen-time mean mesh distances | Matched the bundled numerical summaries |
| `calibration_eval.py` | 22 evaluation views, confidence/error maps, JSON, CSV, PNG and SVG outputs completed |
| Model and result asset hashes | All entries matched `source_manifest.json` |
| Exported PLY attributes | Fourteen frames checked for 12,000 unique finite centers and the Beta opacity-variance formula |
| Video decoding | All 252 frames decoded; all six rendered panels contained visible changing results |
| FFprobe | H.264, 1440 x 960, 24 fps, 10.5 seconds |
| Public `cook_spinach` model replay | All 60 held-out saved renders matched fresh model renders exactly after uint8 encoding |
| Public reference-image check | All 30 original camera-00 images matched both runs' saved references |
| Public comparison and 3D videos | 720 decoded frames total; each video has 360 frames at 30 fps and lasts 12 seconds |
| Public orbit camera | Zero-angle pose, orthonormal rotation, pivot projection and both pixel-center conventions checked |

The regression suite covers camera and measurement conventions, temporal selection,
surface correspondence, adaptive splitting, surface-support constraints, scalar
color ranges, sequence manifests, model rendering, local disparity-plane fitting
and packaged assets. Viewer checks in `run_checks.py` exercise data and state
logic without opening a desktop window.

All new training and evaluation runs use timestamped `outputs/` folders. The
included 3,000-iteration models and result sequences retain their original bytes.

## Spyder Entry Points

Open `view_controlled_results.py` in the `envtorch` kernel and press F5 to inspect
both examples. Open `train_4dgs.py` and press F5 for the default quick training.
Open `run_checks.py` and press F5 to repeat the regression suite. Each entry point
resolves bundled paths relative to its own file.

Individual regression files are in `tests/` and can be run directly with F5.
After relocation, the numerical test function bodies were checked against the
original files; only the public-video asset-root expression changed inside a
test function. No new decorators or classes were introduced. The source manifest
retains original snapshot names and hashes, as described in `DATA.md`.
