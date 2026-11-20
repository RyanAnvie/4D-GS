# Public Dataset Results

The real multiview sequence is **cook_spinach**, from the
[Neural 3D Video Synthesis Dataset](https://github.com/facebookresearch/Neural_3D_Video),
released by Facebook Research under
[CC BY-NC 4.0](https://github.com/facebookresearch/Neural_3D_Video/blob/main/LICENSE).

## Saved Experiments

Both videos use the original project's completed
`benchmark_results/cook_spinach_30f_surface_v1/` runs:

| Run | Training | Active Gaussians per timestamp |
|---|---|---:|
| `dense_adaptive` | Adaptive densification, 6,000 iterations, SH degree 1 | 100,000 |
| `surface_adaptive` | Surface-constrained densification, 6,000 iterations, SH degree 1 | 100,000 |

Each model stores 30 reference frames. Training used cameras 01-20 at all 30
timestamps, for 600 images. Camera 00 was held out, yielding 30 evaluation images.
The processed image resolution is 480 x 360. Frame 00 corresponds to 0 seconds;
frame 29 corresponds to 29/30 seconds. Coordinates retain the released SfM scale.

Model files, per-frame counts, hashes and render-replay checks are recorded in
[video_manifest.json](../media/cook_spinach/video_manifest.json).

## Interactive 3D Viewer

Open `view_pointclouds.py` in Spyder with `envtorch` and press F5. The default
experiment is **Surface-constrained**; the **Experiment** selector switches to
**Adaptive densification** and back. Both sequences expose all 30 saved PLY
frames, with 100,000 Gaussians per frame and the original SfM coordinate scale.

Use the timeline, previous/next buttons and play/pause to inspect time. Playback
starts at 0.25x speed, with looping enabled. Mouse drag orbits the camera, the
wheel zooms, and Ctrl-drag pans. **Geometry** selects Gaussian ellipsoid outlines
or their centers. **Color** selects RGB, confidence `log(kappa)`, or uncertainty
as Beta opacity variance; scalar modes include a colorbar fixed over the selected
sequence. Switching experiments pauses playback and starts the selected run at
its first frame, retaining the color, geometry and speed selections.

`PUBLIC_PROJECT` near the top of the script points to the local project containing
`benchmark_results/cook_spinach_30f_surface_v1/`. The current machine's original
project is detected automatically; set this variable for a different data
location. `DEFAULT_EXPERIMENT` selects which of the two runs opens first.

## Comparison Video

[Watch the comparison](../media/cook_spinach/cook_spinach_comparison.mp4).

The columns are original held-out camera-00 imagery, adaptive densification, and
the surface-constrained model. The upper row retains the complete image. The
lower row enlarges the same rectangle in all columns: x = 120..428, y = 170..350
in the 480 x 360 image, using Pillow's exclusive right/bottom crop bounds.
Bilinear resizing is used only for presentation; source images are preserved.

Thirty timestamps are displayed at quarter speed and repeated three times:
12 seconds at 30 fps. This is a repeated presentation of the first second of the
saved reconstruction, rather than a 12-second reconstructed source sequence.

## Three-Dimensional Video

[Watch the camera orbit](../media/cook_spinach/cook_spinach_3d.mp4).

- Seconds 0-4: time playback with a fixed camera.
- Seconds 4-8: camera orbit at the fixed physical time t = 0.5 s.
- Seconds 8-12: time playback and camera orbit together.

The two saved models share the same camera path. The orbit spans -10 to +10
degrees around a pivot defined from the first model's median camera-space depth.
All active Gaussians remain in rendering; no opacity threshold or scale inflation
is applied. Positions, scales, rotations, opacity and spherical-harmonic appearance
are read from the saved model files. The RGB output is alpha-composited on black.

## Recreate Locally

Open `export_public_video.py` in Spyder with the `envtorch` interpreter and press
F5. `PUBLIC_PROJECT` identifies the original project containing the downloaded
`benchmark_data/cook_spinach_30f_50k/` inputs and completed run directories.
Edit that variable when using a different local directory layout. New exports
are written into timestamped `outputs/cook_spinach_video_*` folders.

Each export checks all 60 saved evaluation renders against a fresh model reload,
checks the original reference images, and records model and image hashes.
