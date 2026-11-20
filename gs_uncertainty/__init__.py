# -*- coding: utf-8 -*-
"""Photogrammetry-guided uncertainty tools for dynamic 4DGS.

Import submodules directly, for example:

    from gs_uncertainty.evidence import build_photogrammetry_targets
    from gs_uncertainty.losses import kappa_evidence_loss

The package initializer stays lightweight so evidence and metric utilities can
be used without importing the CUDA Gaussian rasterizer.
"""
