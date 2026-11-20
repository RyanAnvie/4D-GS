"""Training-gradient-driven, frame-local splitting of surface Gaussians."""

import math

import torch
from torch import nn

from model import inverse_sigmoid
from surface_support import constrain_split_offsets


def new_densification_stats(model):
    return {name: torch.zeros(model.n_gaussians, device=model.mu_ref.device)
            for name in ("gradient_sum", "observations", "max_radius")}


def accumulate_screen_gradients(stats, screen_samples):
    """Gradients already include shutter averaging; count fractional observations."""
    with torch.no_grad():
        for sample in screen_samples:
            gradient = sample["viewspace"].grad
            if gradient is None or sample["radii"] is None:
                raise RuntimeError("Densification requires rasterizer screen gradients and radii.")
            magnitude = torch.linalg.vector_norm(gradient[:, :2], dim=1)
            if not torch.isfinite(magnitude).all():
                raise ValueError("Nonfinite screen gradient; refusing to densify.")
            visible = sample["radii"] > 0
            idx = sample["indices"][visible]
            stats["gradient_sum"].index_add_(0, idx, magnitude[visible])
            weight = torch.full_like(magnitude[visible], 1.0 / len(screen_samples))
            stats["observations"].index_add_(0, idx, weight)
            radius = sample["radii"][visible].float()
            stats["max_radius"][idx] = torch.maximum(stats["max_radius"][idx], radius)


def select_split_parents(model, stats, config, support=None):
    """Rank positive mean gradients within each frame, subject to explicit budgets."""
    source_limit = getattr(config, "densify_max_per_source", 0)
    if type(source_limit) is not int or source_limit < 0:
        raise ValueError("densify_max_per_source must be a nonnegative integer.")
    with torch.no_grad():
        score = stats["gradient_sum"] / stats["observations"].clamp(min=1e-8)
        eligible = (stats["observations"] >= config.densify_min_observations)
        eligible &= score > config.densify_grad_threshold
        eligible &= stats["max_radius"] >= config.densify_min_radius
        eligible &= model.get_opacity_mean().reshape(-1) >= config.densify_min_opacity
        before_geometry = eligible.clone()
        if config.densify_geometry_filter:
            if support is None:
                raise ValueError("Geometry-aware parent selection needs original source surfels.")
            source = model.source_indices
            displacement = model.mu_ref - support["points"][source]
            normal_offset = torch.sum(displacement * support["normals"][source], dim=1).abs()
            # An algorithmic support test, not a measurement-error confidence bound.
            band = config.surface_normal_band * support["spacing"][source].reshape(-1)
            eligible &= normal_offset <= band
        selected = []
        rows = []
        if source_limit:
            remaining = (source_limit - torch.bincount(model.source_indices)).cpu().tolist()
        for frame in torch.unique(model.frame_indices, sorted=True):
            frame_mask = model.frame_indices == frame
            count = int(frame_mask.sum())
            candidates = torch.nonzero(frame_mask & eligible, as_tuple=False).reshape(-1)
            budget = min(config.densify_max_per_frame - count,
                         int(count * config.densify_fraction), len(candidates))
            budget = max(budget, 0)
            source_skipped = 0
            if budget:
                order = torch.argsort(score[candidates], descending=True, stable=True)
                if source_limit:
                    ranked = candidates[order]
                    roots = model.source_indices[ranked].cpu().tolist()
                    chosen = []
                    # One split adds one descendant; reserve slots across the entire batch.
                    for parent, source in zip(ranked.cpu().tolist(), roots):
                        if remaining[source] <= 0:
                            source_skipped += 1
                            continue
                        chosen.append(parent)
                        remaining[source] -= 1
                        if len(chosen) == budget:
                            break
                    budget = len(chosen)
                    if budget:
                        selected.append(torch.tensor(chosen, dtype=torch.long, device=score.device))
                else:
                    selected.append(candidates[order[:budget]])
            row = {"frame": int(frame), "before": count, "eligible": len(candidates),
                   "split_parents": budget, "after": count + budget}
            if config.densify_geometry_filter:
                row["geometry_rejected"] = int((frame_mask & before_geometry & ~eligible).sum())
            if source_limit:
                row["source_cap_skipped"] = source_skipped
            rows.append(row)
        parents = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long, device=score.device)
    return parents, rows


def split_surface_gaussians(model, optimizer, parents, support=None, config=None):
    """Replace parents with distinct children; optionally retain each parent center."""
    position_mode = getattr(config, "densify_position_mode", "symmetric")
    if position_mode not in ("symmetric", "parent_preserving"):
        raise ValueError("Unknown densify_position_mode.")
    if model.mode != "flow_sand":
        raise ValueError("Surface splitting is only implemented for flow_sand, not tracked identities.")
    if parents.numel() == 0:
        return
    with torch.no_grad():
        n = model.n_gaussians
        if parents.min() < 0 or parents.max() >= n or len(torch.unique(parents)) != len(parents):
            raise ValueError("Split parents must be unique valid Gaussian indices.")
        keep = torch.ones(n, dtype=torch.bool, device=parents.device)
        keep[parents] = False
        retained = torch.nonzero(keep, as_tuple=False).reshape(-1)
        child_sources = parents.repeat_interleave(2)
        source = torch.cat((retained, child_sources))
        child_start = len(retained)

        scales = model.scaling[parents].exp()
        # Local axes 0 and 1 are tangential; never split along the thin normal axis 2.
        axis = scales[:, :2].argmax(dim=1)
        local = torch.zeros_like(scales)
        row = torch.arange(len(parents), device=parents.device)
        local[row, axis] = scales[row, axis] / math.sqrt(2.0)
        q = torch.nn.functional.normalize(model.rotation[parents], dim=1)
        cross = torch.cross(q[:, 1:], local, dim=1)
        displacement = local + 2 * q[:, :1] * cross + 2 * torch.cross(q[:, 1:], cross, dim=1)
        offsets = torch.stack((-displacement, displacement), dim=1).reshape(-1, 3)
        if support is not None:
            offsets = constrain_split_offsets(model, parents, displacement, support, config)
        if position_mode == "parent_preserving":
            proposals = model.mu_ref[parents, None, :] + offsets.reshape(-1, 2, 3)
            squared = (proposals - model.mu_ref[parents, None, :]).square().sum(dim=2)
            chosen = squared.argmax(dim=1)
            ties = squared[:, 0] == squared[:, 1]
            chosen[ties] = model.track_ids[parents[ties]] % 2
            children = torch.stack((model.mu_ref[parents], proposals[row, chosen]), dim=1)
            if (children[:, 0] == children[:, 1]).all(dim=1).any():
                raise ValueError("Parent-preserving split cannot produce distinct float32 centers.")
            offsets = (children - model.mu_ref[parents, None, :]).reshape(-1, 3)
        child_scaling = model.scaling[child_sources].clone()
        child_scaling[torch.arange(len(child_sources), device=parents.device), axis.repeat_interleave(2)] -= math.log(2.0) / 2
        # Only the unconstrained symmetric split preserves parent mean/covariance.
        # Alpha transfer is a heuristic: it preserves transmittance only for coincident footprints.
        alpha = model.get_opacity_mean(child_sources)
        child_opacity = inverse_sigmoid(1.0 - torch.sqrt(1.0 - alpha))
        next_id = int(model.track_ids.max()) + 1
        parent_ids = model.track_ids[child_sources].clone()

        for group in optimizer.param_groups:
            name = group["name"]
            old = group["params"][0]
            values = old.detach()[source].clone()
            if name == "mu_ref":
                values[child_start:] += offsets
                if position_mode == "parent_preserving":
                    # Assign final positions directly, avoiding subtract/add cancellation.
                    values[child_start:] = children.reshape(-1, 3)
            elif name == "scaling":
                values[child_start:] = child_scaling
            elif name == "opacity_logit":
                values[child_start:] = child_opacity
            parameter = nn.Parameter(values, requires_grad=old.requires_grad)
            state = optimizer.state.pop(old, None)
            if state is not None:
                # Retained Adam moments continue unchanged; children start with zero moments.
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state:
                        moments = state[key][source].clone()
                        moments[child_start:] = 0
                        state[key] = moments
                optimizer.state[parameter] = state
            group["params"][0] = parameter
            setattr(model, name, parameter)

        for name in ("t_ref", "anchor_mu_ref", "anchor_velocity", "anchor_rotation", "track_ids",
                     "track_times", "track_points", "track_mask", "frame_indices",
                     "source_indices", "parent_ids", "generation"):
            values = getattr(model, name)[source].clone()
            if name == "anchor_mu_ref":
                values[child_start:] += offsets
            elif name == "track_ids":
                values[child_start:] = torch.arange(next_id, next_id + len(child_sources), device=parents.device)
            elif name == "parent_ids":
                values[child_start:] = parent_ids
            elif name == "generation":
                values[child_start:] += 1
            # Reference measurements and evidence IDs are inherited, not new observations.
            setattr(model, name, values)
