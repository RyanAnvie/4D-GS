# Third-Party Components

The renderer calls the separately installed
[Graphdeco differentiable Gaussian rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization).
That component is distributed under the upstream
[Gaussian-Splatting license](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/LICENSE.md).
Its source and compiled binary remain external dependencies.

The project uses PyTorch, NumPy, SciPy, Open3D, OpenCV, Pillow, Matplotlib,
plyfile and Spyder under their respective licenses. Video encoding uses the
user's installed FFmpeg build and its associated license.

The bundled example data consist of project-generated controlled textured
surfaces, rendered images, point clouds, trained model parameters and evaluation
references. Their source experiment identifiers and result hashes are recorded
in `docs/source_manifest.json`.

## Public Video Dataset

Reference imagery and derived reconstruction videos in `media/cook_spinach/`
use the **Neural 3D Video Synthesis Dataset**, released by Facebook Research for
*Neural 3D Video Synthesis from Multi-View Video*.

- [Dataset and attribution](https://github.com/facebookresearch/Neural_3D_Video)
- [Original release](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0)
- [CC BY-NC 4.0 license](https://github.com/facebookresearch/Neural_3D_Video/blob/main/LICENSE)

The presentation selects the first 30 timestamps, displays resized reference and
reconstruction images side by side, adds a labeled common detail crop, and renders
virtual camera views from the project's trained Gaussian models. Source filenames,
model hashes and presentation transformations are recorded in
`media/cook_spinach/video_manifest.json`. The upstream attribution and
noncommercial license apply to these dataset-derived materials.
