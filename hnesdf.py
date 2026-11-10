"""Smooth box SDF and batched box-to-oriented-point-cloud h-NESDF."""

import torch


def h_esdf_box(T, lx, ly, lz, p, h):
    """Box-to-point field; T (...,4,4), p (...,3), broadcastable.

    Dimensions are full side lengths; h and dimensions must be positive.
    T and p must be floating tensors on the same device with the same dtype.
    For B poses and K points, pass T[:, None] and p[None].
    """
    u = (T[..., :3, :3].transpose(-1, -2)
         @ (p - T[..., :3, 3]).unsqueeze(-1)).squeeze(-1)
    a = p.new_tensor([lx, ly, lz]) / 2
    q = torch.logsumexp(torch.stack((u / h, -u / h), -1), -1) - a / h
    positive = torch.logsumexp(torch.stack((q, torch.zeros_like(q)), -1), -1)
    maximum = torch.logsumexp(q, -1)
    negative = torch.logsumexp(
        torch.stack((-maximum, torch.zeros_like(maximum)), -1), -1)
    # Norm radius h**2 / ||a|| vanishes faster than the LSE bandwidth h.
    outside = torch.sqrt(positive.square().sum(-1) + (h / a.norm()).square())
    return h * (outside - negative)


@torch.enable_grad()
def hNESDF(batch_T, lx, ly, lz, h, list_points, *,
           return_tensors=False, compute_gradient=True,
           gradient_method="autograd", fd_steps=(1e-3, 1e-3)):
    """Return distance list and list of 6D pose differentials.

    batch_T: (B,4,4) floating tensor of box poses.
    list_points: (K,6) tensor [px,py,pz,nx,ny,nz], shared by all poses.
        Points are approximately uniform boundary samples of the other body;
        normals are outward unit normals. Both are in world coordinates.
    Each differential L uses xi=[vx,vy,vz,wx,wy,wz], with world-frame
    origin velocity s_dot=v and angular velocity R_dot=[w]_x R.
    Set return_tensors=True to keep detached outputs on the input device.
    Set compute_gradient=False for distance-only evaluation; L is then None.
    gradient_method: "autograd" or "finite_difference" for the pose gradient.
    fd_steps: (translation step in length units, rotation step in radians).
        Central differences evaluate perturbations sequentially. This avoids
        second-order autograd; the spatial derivative still uses autograd.
    """
    if batch_T.ndim != 3 or batch_T.shape[1:] != (4, 4):
        raise ValueError("batch_T must have shape (B,4,4).")
    if gradient_method not in ("autograd", "finite_difference"):
        raise ValueError("gradient_method must be 'autograd' or 'finite_difference'.")
    if compute_gradient and gradient_method == "finite_difference":
        if len(fd_steps) != 2 or not all(0 < x < float("inf") for x in fd_steps):
            raise ValueError("fd_steps must contain two positive steps.")
        if batch_T.dtype not in (torch.float32, torch.float64):
            raise ValueError("Use float32 or float64 for finite differences.")
        T = batch_T.detach()
        distance, _ = hNESDF(T, lx, ly, lz, h, list_points,
                            return_tensors=True, compute_gradient=False)
        L = T.new_empty((len(T), 6))
        for i in range(6):
            eps = fd_steps[i // 3]
            values = []
            for sign in (1, -1):
                perturbed = T.clone()
                if i < 3:
                    perturbed[:, i, 3] += sign * eps
                else:
                    W = T.new_zeros((3, 3))
                    a, b = (i - 2) % 3, (i - 1) % 3
                    W[a, b], W[b, a] = -sign * eps, sign * eps
                    perturbed[:, :3, :3] = torch.matrix_exp(W) @ T[:, :3, :3]
                value, _ = hNESDF(perturbed, lx, ly, lz, h, list_points,
                                 return_tensors=True, compute_gradient=False)
                values.append(value)
            L[:, i] = (values[0] - values[1]) / (2 * eps)
        if return_tensors:
            return distance, L
        return distance.cpu().tolist(), L.cpu().tolist()
    T = batch_T.detach().clone().requires_grad_(compute_gradient)
    cloud = torch.as_tensor(list_points, dtype=T.dtype, device=T.device)
    if cloud.ndim != 2 or cloud.shape[1] != 6 or cloud.shape[0] == 0:
        raise ValueError("list_points must be a nonempty (K,6) oriented cloud.")
    p = cloud[None, :, :3].expand(T.shape[0], -1, -1)
    p = p.detach().clone().requires_grad_(True)

    D = h_esdf_box(T[:, None], lx, ly, lz, p, h)
    d = h_esdf_box(T, lx, ly, lz, T[:, :3, 3], h)
    E = 0.5 * (D - d[:, None]).square()  # Constant -d**2/2 cancels.
    grad_E = torch.autograd.grad(E.sum(), p, create_graph=compute_gradient)[0]
    if not compute_gradient:
        E = E.detach()  # No pose derivative is needed after the spatial derivative.
    weights = torch.softmax(-E / h**2, dim=-1)
    projection = (grad_E * cloud[None, :, 3:]).sum(-1)
    distance = d - (weights * projection).sum(-1)

    L = None
    if compute_gradient:
        G = torch.autograd.grad(distance.sum(), T)[0]
        M = G[:, :3, :3] @ T[:, :3, :3].transpose(-1, -2)
        angular = torch.stack((M[:, 2, 1] - M[:, 1, 2],
                               M[:, 0, 2] - M[:, 2, 0],
                               M[:, 1, 0] - M[:, 0, 1]), -1)
        L = torch.cat((G[:, :3, 3], angular), -1).detach()
    distance = distance.detach()
    if return_tensors:
        return distance, L
    return distance.cpu().tolist(), None if L is None else L.cpu().tolist()
