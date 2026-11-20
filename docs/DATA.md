# Data And Outputs

## Coordinates And Time

Positions, Gaussian spatial scales, mesh vertices and geometric errors use
meters. Timestamps and exposure durations use seconds; velocities use meters
per second. Cameras use world-to-camera matrices with positive camera z forward
and OpenCV integer pixel centers. Quaternions are stored in w,x,y,z order.

## Included Examples

Each `examples/data/<experiment>` contains 16 FIT images, OpenMVG-style camera
JSON, a view/time/exposure manifest, four source clouds and the sensitivity
surface origins. `evaluation/` contains all42 camera/time images and seven
independently generated mesh/depth references. The training configuration reads
the FIT inputs; `evaluate_results.py` reads evaluation references separately.

Metadata paths are relative to their repository/data directories. JSON path
relocation preserves camera parameters, image pixels, timestamps and coordinates.
Saved model and PLY bytes match the source results recorded in the manifest.

The `source_modules_sha256` entries in `source_manifest.json` describe the
original source snapshot, before packaging adaptations. Its original
`test_*.py` names correspond to the relocated `tests/test_*.py` files; the
historical source hashes are retained unchanged.

## Attributed PLY Sequences

`sequence_manifest.json` gives the numerical frame order, physical timestamps,
and held-out-time flags. Each binary PLY includes XYZ, RGB, opacity, Beta
concentration/variance, Gaussian scales and rotations, and source identifiers.
The viewer reads all seven files and computes fixed whole-sequence color limits.

Confidence coloring uses `log(opacity_concentration)`. Uncertainty coloring uses
`opacity_variance = mu*(1-mu)/(kappa+1)`. Gaussian ellipsoid outlines use the
exported spatial scales and rotations with scale multiplier1 by default.

## Trained Model

`dynamic_gaussians_final.npz` stores reference positions/times, velocities,
SH features, opacity parameters, shape parameters, frame/source identifiers
and the temporal selection mode. `render_settings.json` specifies exposure
sampling, temporal support and clipping planes. `train_config.json` records the
3,000-iteration training setup. The included models use SH degree0.

## Evaluation

`summary.json` contains per-frame geometry, per-image PSNR/SSIM/MAE, grouped
metrics and confidence-error summaries. Geometry uses all active centers.
Completeness uses camera0-visible reference samples. The22 evaluation images
include every held-out timestamp and camera5 at observed timestamps. Camera2
at observed timestamps is recorded as a reserved measurement view.

All new entry-point outputs use fresh timestamped directories. The demo's
trained models and saved results are read as inputs, independently of new runs.
