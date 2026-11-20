"""Reference-time surfel constraints tied to unchanged training measurements."""

import math

import numpy as np
import torch


def build_surface_support(init, device, origin_path=None):
    points = torch.as_tensor(init["mu_ref"], dtype=torch.float32, device=device).clone()
    rotations = torch.as_tensor(init["rotations"], dtype=torch.float32, device=device)
    scales = torch.as_tensor(init["scales"], dtype=torch.float32, device=device)
    if not all(torch.isfinite(value).all() for value in (points, rotations, scales)):
        raise ValueError("Nonfinite surface support; measurements must not be silently filtered.")
    if (scales <= 0).any() or (torch.linalg.vector_norm(rotations, dim=1) < 1e-8).any():
        raise ValueError("Surface support requires positive spacing and valid normal rotations.")
    q = torch.nn.functional.normalize(rotations, dim=1)
    # Third column of R for the existing [w,x,y,z] convention: local normal axis.
    normals = torch.stack((2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2]),
                           2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1]),
                           1 - 2 * (q[:, 1].square() + q[:, 2].square())), dim=1)
    normals = torch.nn.functional.normalize(normals, dim=1)
    # initialize_sand_surface sets both tangent scales to mean k-neighbor spacing.
    spacing = scales[:, :2].mean(dim=1, keepdim=True)
    support = {"points": points, "normals": normals, "spacing": spacing,
            "frame_indices": torch.as_tensor(init["frame_indices"], dtype=torch.long, device=device),
            "t_ref": torch.as_tensor(init["t_ref"], dtype=torch.float32, device=device)}
    if origin_path is not None:
        # A derived plane is a regularization target, never a replacement measurement.
        with np.load(origin_path, allow_pickle=False) as data:
            raw = torch.as_tensor(data["raw_points"], dtype=torch.float32, device=device)
            frames = torch.as_tensor(data["frames"], dtype=torch.long, device=device)
            original_normals = torch.as_tensor(data["normals"], dtype=torch.float32, device=device)
            origins = torch.as_tensor(data["origins"], dtype=torch.float32, device=device)
        if not torch.equal(raw, points) or not torch.equal(frames, support["frame_indices"]):
            raise ValueError("Derived surface origins do not match initialization source order/frames.")
        if original_normals.shape != normals.shape or not torch.allclose(original_normals, normals, rtol=0, atol=1e-6):
            raise ValueError("Derived surface origins use different source normals.")
        if origins.shape != points.shape or not torch.isfinite(origins).all():
            raise ValueError("Derived surface origins must be finite [N,3] coordinates.")
        displacement = origins-points
        height = torch.sum(displacement*normals, dim=1, keepdim=True)
        tangent = displacement-height*normals
        tolerance = 8*torch.finfo(torch.float32).eps*points.abs().amax(dim=1).clamp(min=1)
        if torch.any(torch.linalg.vector_norm(tangent, dim=1) > tolerance):
            raise ValueError("Derived surface origins must change only the normal coordinate.")
        support["raw_points"] = points
        support["points"] = origins.clone()
    return support


def surface_support_loss(model, support, indices):
    """All active centers, associated support planes, dimensionless normal distance."""
    source = model.source_indices[indices]
    displacement = model.mu_ref[indices] - support["points"][source]
    height = torch.sum(displacement * support["normals"][source], dim=1, keepdim=True)
    return (height.abs() / support["spacing"][source]).mean()


def project_surface_centers(model, optimizer, support, config, report=False):
    """Projected Adam: cylinder around each root surfel, at its own reference time.

    All centers are constrained, including inactive frames with residual Adam
    momentum. The legacy policy resets moments only on corrected position rows;
    the explicit retention control preserves them. Measurements,
    velocities, appearance, opacity, evidence IDs and timestamps are unchanged.
    """
    with torch.no_grad():
        source = model.source_indices
        reference = support["points"][source]
        normal = support["normals"][source]
        spacing = support["spacing"][source]
        radius = config.surface_tangent_radius * spacing
        thickness = config.surface_normal_band * spacing
        displacement = model.mu_ref - reference
        height = torch.sum(displacement * normal, dim=1, keepdim=True)
        tangent = displacement - height * normal
        tangent_length = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)
        outside = ((height.abs() > thickness) | (tangent_length > radius)).reshape(-1)
        tangent = tangent * (radius / tangent_length.clamp(min=1e-12)).clamp(max=1)
        height_clipped = torch.maximum(torch.minimum(height, thickness), -thickness)
        projected = reference + tangent + height_clipped * normal
        # Do not repeatedly round already valid centers.
        model.mu_ref[outside] = projected[outside]
        state = optimizer.state.get(model.mu_ref) if config.surface_projection_reset_moments else None
        if state is not None:
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state:
                    state[key][outside] = 0
        if report:
            return {"projected_centers": int(outside.sum()),
                    "pre_max_normal_over_spacing": float((height.abs() / spacing).max()),
                    "pre_max_tangent_over_spacing": float((tangent_length / spacing).max())}
    return None


def constrain_split_offsets(model, parents, displacement, support, config):
    """Fit a distinct symmetric pair inside the root surfel's tangent disk.

    Recenter parents near the rim to leave split room; do not clamp both children
    to one boundary point. The constrained split does NOT preserve parent moments.
    """
    source = model.source_indices[parents]
    reference = support["points"][source]
    normal = support["normals"][source]
    spacing = support["spacing"][source]
    radius = config.surface_tangent_radius * spacing
    thickness = config.surface_normal_band * spacing
    relative = model.mu_ref[parents] - reference
    height = torch.sum(relative * normal, dim=1, keepdim=True)
    tangent = relative - height * normal
    length = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)
    inner_radius = config.surface_split_center_fraction * radius
    tangent = tangent * (inner_radius / length.clamp(min=1e-12)).clamp(max=1)
    height = torch.maximum(torch.minimum(height, thickness), -thickness)
    direction = displacement - torch.sum(displacement * normal, dim=1, keepdim=True) * normal
    proposed_distance = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    if (proposed_distance <= 1e-12).any():
        raise ValueError("Split direction is not tangential; check frozen surface rotations.")
    direction = direction / proposed_distance
    along = torch.sum(tangent * direction, dim=1, keepdim=True)
    # Largest +/- distance that fits both children in a disk of radius r.
    room = torch.sqrt((along.square() + radius.square() - tangent.square().sum(dim=1, keepdim=True)).clamp(min=0)) - along.abs()
    distance = torch.minimum(proposed_distance, config.surface_split_room_fraction * room)
    center = reference + tangent + height * normal
    children = torch.stack((center - distance * direction, center + distance * direction), dim=1)
    if (children[:, 0] == children[:, 1]).all(dim=1).any():
        raise ValueError("Surface split cannot produce distinct float32 centers at this coordinate scale.")
    offsets = children - model.mu_ref[parents].unsqueeze(1)
    return offsets.reshape(-1, 3)


def project_surface_scales(model, optimizer, support, config):
    """Bound tangent standard deviations; restore original normal thickness exactly."""
    with torch.no_grad():
        original = support["log_scales"][model.source_indices]
        lower = original[:, 2:3] - math.log(config.normal_ratio_limit)
        tangents = torch.maximum(torch.minimum(model.scaling[:, :2], original[:, :2]), lower)
        projected = torch.cat((tangents, original[:, 2:3]), dim=1)
        corrected = model.scaling != projected
        model.scaling[corrected] = projected[corrected]
        state = optimizer.state.get(model.scaling)
        if state is not None:
            # Normal moments are always zero, including after a split or stale momentum.
            corrected[:, 2] = True
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state:
                    state[key][corrected] = 0
