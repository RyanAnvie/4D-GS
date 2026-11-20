"""Loss consistency and gradients against complete plane/curved references."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from losses import nearest_point_indices, sand_surface_loss
from model import DynamicGaussianModel


def run_checks():
    rng = np.random.default_rng(42)
    for device in ("cpu", "cuda"):
        for shape in ("plane", "sphere"):
            points = rng.uniform(-1., 1., (2000, 3)).astype(np.float32)
            if shape == "plane":
                points[:, 2] = 2.
                normals = np.tile([0., 0., 1.], (len(points), 1)).astype(np.float32)
                displacement = 0.02
            else:
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                normals = points.copy()
                displacement = 0.01
            n = len(points)
            init = {"mu_ref": points, "velocity": np.zeros_like(points), "t_ref": np.zeros(n),
                    "colors": np.full_like(points, 0.5), "scales": np.full_like(points, 0.01),
                    "rotations": np.tile([1., 0., 0., 0.], (n, 1)), "track_ids": np.arange(n),
                    "track_times": np.zeros((n, 1)), "track_points": points[:, None, :],
                    "track_mask": np.ones((n, 1))}
            model = DynamicGaussianModel(init, device=device)
            target = torch.tensor(points, device=device)
            normal = torch.tensor(normals, device=device)
            query = target[::43]
            nearest = nearest_point_indices(query, target, block_size=127)
            np.testing.assert_array_equal(nearest.cpu().numpy(), np.arange(n)[::43])
            torch.manual_seed(42)
            exact, _ = sand_surface_loss(model, 0., target, normal, 0.1, sample_count=128,
                                         correspondence_mode="full_reference")
            assert float(exact.detach()) == 0.
            exact.backward()
            assert torch.count_nonzero(model.mu_ref.grad) == 0
            model.mu_ref.grad = None
            legacy, _ = sand_surface_loss(model, 0., target, normal, 0.1, sample_count=128)
            assert float(legacy.detach()) > 0.05
            with torch.no_grad():
                model.mu_ref.add_(displacement * normal)
            offset, _ = sand_surface_loss(model, 0., target, normal, 0.1, sample_count=128,
                                          correspondence_mode="full_reference")
            np.testing.assert_allclose(float(offset.detach()), 3 * displacement, atol=2e-6)
            offset.backward()
            assert torch.isfinite(model.mu_ref.grad).all()
            normal_derivative = torch.sum(model.mu_ref.grad * normal)
            assert abs(float(normal_derivative) - 3.) < 1e-5
            print(f"PASS {device} {shape}: exact self-match, zero loss/gradient, analytic offset and restoring gradient.")
            model.mu_ref.grad = None
            surface, _ = sand_surface_loss(model, 0., target, normal, 0.1, sample_count=128,
                correspondence_mode="full_reference", objective="plane_coverage")
            np.testing.assert_allclose(float(surface.detach()), 2 * displacement, atol=2e-6)
            surface.backward()
            assert torch.isfinite(model.mu_ref.grad).all()
            assert abs(float(torch.sum(model.mu_ref.grad * normal)) - 2.) < 1e-5
            if shape == "plane":
                with torch.no_grad():
                    model.mu_ref.copy_(target)
                model.mu_ref.grad = None
                surface, _ = sand_surface_loss(model, 0., target[:1000], normal[:1000], 0.1,
                    sample_count=2000, correspondence_mode="full_reference", objective="plane_coverage")
                euclidean, _ = sand_surface_loss(model, 0., target[:1000], normal[:1000], 0.1,
                    sample_count=2000, correspondence_mode="full_reference")
                assert float(surface.detach()) == 0.
                assert float(euclidean.detach()) > 0.
                surface.backward()
                assert torch.count_nonzero(model.mu_ref.grad) == 0
                print(f"PASS {device} plane: extra distinct tangent samples have zero loss/gradient.")
            print(f"PASS {device} {shape}: analytic plane/coverage normal offset and gradient.")


if __name__ == "__main__":
    run_checks()
