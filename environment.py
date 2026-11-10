"""Box-environment sampling and a furnished house scene."""

import numpy as np
import torch


def sample_box_environment(scenarios, delta, *, device=None,
                           dtype=torch.float32, tol=1e-9):
    """
    scenarios: list of (T, lx, ly, lz), with full side lengths.
    delta: maximum grid spacing along each face direction.
    tol: geometric tolerance, in the same units as the lengths.

    Returns a (K,6) tensor of world-frame points and outward unit normals.
    """
    if not np.isfinite(delta) or delta <= 0 or not np.isfinite(tol) or tol < 0:
        raise ValueError("delta must be positive and tol nonnegative.")

    boxes = []
    for T, lx, ly, lz in scenarios:
        T = np.asarray(T, dtype=float)  # Also accepts np.matrix.
        lengths = np.array([lx, ly, lz], dtype=float)
        if (T.shape != (4, 4) or not np.isfinite(T).all()
                or not np.isfinite(lengths).all() or np.any(lengths <= 0)):
            raise ValueError("Each box needs an SE(3) pose and positive lengths.")
        boxes.append((T[:3, :3], T[:3, 3], lengths / 2))

    samples = []
    for i, (R, center, half) in enumerate(boxes):
        for axis in range(3):
            tangents = [k for k in range(3) if k != axis]
            count = np.ceil(2 * half[tangents] / delta).astype(int)

            # Cell centers avoid sampling edges with ambiguous normals.
            grids = [
                (np.arange(n) + 0.5) * (2 * half[k] / n) - half[k]
                for k, n in zip(tangents, count)
            ]
            grid = np.stack(
                np.meshgrid(*grids, indexing="ij"), -1
            ).reshape(-1, 2)

            for side in (-1, 1):
                local = np.zeros((len(grid), 3))
                local[:, tangents] = grid
                local[:, axis] = side * half[axis]

                points = local @ R.T + center
                normal = side * R[:, axis]
                keep = np.ones(len(points), dtype=bool)

                for j, (Rj, cj, hj) in enumerate(boxes):
                    if j == i or not keep.any():
                        continue

                    ids = np.flatnonzero(keep)
                    q = (points[ids] - cj) @ Rj
                    gap = hj - np.abs(q)

                    inside = np.all(gap > tol, axis=1)
                    closed = np.all(gap >= -tol, axis=1)
                    boundary = closed & ~inside

                    # Check whether a coincident face points outward
                    # in the same direction as the sampled face.
                    direction = normal @ Rj
                    same_normal = np.any(
                        (np.abs(gap) <= tol)
                        & (np.sign(q) * direction > 1 - 1e-10),
                        axis=1,
                    )

                    # Remove buried/glued points. Shared exposed patches
                    # belong to the first box, preventing double density.
                    hidden = inside | (
                        boundary & (~same_normal | (j < i))
                    )
                    keep[ids[hidden]] = False

                points = points[keep]
                if len(points):
                    normals = np.broadcast_to(normal, points.shape)
                    samples.append(np.concatenate((points, normals), axis=1))

    cloud = np.concatenate(samples) if samples else np.empty((0, 6))
    return torch.as_tensor(cloud, dtype=dtype, device=device)

# A furnished 12 m x 10 m house made entirely of box tuples.


def create_house_environment(include_ceiling=False):
    """Return [(T, lx, ly, lz), ...], with 4x4 np.matrix poses.

    Units are metres. The floor's upper surface is z=0. The house occupies
    x in [-6,6], y in [-5,5], with 2.8 m high walls. Its 1.6 m wide central
    corridor connects a 1.2 m entrance to four 1.0 m room doorways; all
    doorway clear heights are 2.1 m. Four side windows are 1.6 x 1.0 m.
    Doorways/windows are gaps in the box geometry, with no door leaves/glass.
    Set include_ceiling=True to add a ceiling at z=2.8 m.
    """
    boxes = []

    def add(center, size, yaw=0.0):
        c, s = np.cos(yaw), np.sin(yaw)
        T = np.eye(4)
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        T[:3, 3] = center
        boxes.append((np.matrix(T), *map(float, size)))

    # Floor, with its top at z=0.
    add((0, 0, -0.10), (12, 10, 0.20))

    # West/east walls. Sills and lintels leave windows at y=-2.0 and 2.2,
    # with clear z range [1.0,2.0].
    for x in (-5.9, 5.9):
        add((x, 0, 0.5), (0.2, 9.6, 1.0))
        add((x, 0, 2.4), (0.2, 9.6, 0.8))
        for lo, hi in [(-4.8, -2.8), (-1.2, 1.4), (3.0, 4.8)]:
            add((x, (lo + hi) / 2, 1.5), (0.2, hi - lo, 1.0))

    # North wall and south entrance: x in [-0.6,0.6] remains open below 2.1.
    add((0, 4.9, 1.4), (12, 0.2, 2.8))
    for x in (-3.3, 3.3):
        add((x, -4.9, 1.4), (5.4, 0.2, 2.8))
    add((0, -4.9, 2.45), (1.2, 0.2, 0.7))

    # Corridor: clear x range [-0.8,0.8]. Room doors at y=-2.0 and 2.2.
    for x in (-0.9, 0.9):
        for lo, hi in [(-4.8, -2.5), (-1.5, 1.7), (2.7, 4.8)]:
            add((x, (lo + hi) / 2, 1.4), (0.2, hi - lo, 2.8))
        for y in (-2.0, 2.2):
            add((x, y, 2.45), (0.2, 1.0, 0.7))

    # Separate the front/back rooms while leaving the corridor continuous.
    for x in (-3.4, 3.4):
        add((x, 0.1, 1.4), (4.8, 0.2, 2.8))

    # Southwest: dining table with four legs.
    add((-3.4, -2.2, 0.76), (1.5, 0.9, 0.08))
    for dx in (-0.62, 0.62):
        for dy in (-0.33, 0.33):
            add((-3.4 + dx, -2.2 + dy, 0.36), (0.08, 0.08, 0.72))

    # Sofa: base/seat, back and two armrests.
    add((-5.25, -1.45, 0.30), (0.9, 2.2, 0.6))
    add((-5.65, -1.45, 0.80), (0.12, 2.2, 0.4))
    for y in (-2.50, -0.40):
        add((-5.25, y, 0.75), (0.9, 0.12, 0.3))

    # TV cabinet, pedestal and screen facing the sofa.
    add((-1.45, -0.65, 0.25), (0.65, 1.1, 0.5))
    add((-1.45, -0.65, 0.55), (0.08, 0.2, 0.1))
    add((-1.45, -0.65, 0.95), (0.04, 0.95, 0.7))

    # Four dining chairs and one slightly rotated desk chair.
    # Each chair comprises a seat, a backrest and four legs.
    parts = [((0, 0, 0.45), (0.44, 0.44, 0.08)),
             ((0, 0.20, 0.74), (0.44, 0.04, 0.50))]
    parts += [((dx, dy, 0.205), (0.045, 0.045, 0.41))
              for dx in (-0.16, 0.16) for dy in (-0.16, 0.16)]
    chairs = [(-3.9, -3.0, np.pi), (-2.9, -3.0, np.pi),
              (-3.9, -1.35, 0), (-2.9, -1.35, 0),
              (4.2, 2.75, np.pi + np.pi / 12)]
    for x, y, yaw in chairs:
        c, s = np.cos(yaw), np.sin(yaw)
        for (dx, dy, z), size in parts:
            add((x + c * dx - s * dy, y + s * dx + c * dy, z), size, yaw)

    # Northwest: bed frame, mattress, headboard, two pillows and storage.
    add((-3.2, 3.3, 0.20), (1.8, 2.2, 0.4))
    add((-3.2, 3.3, 0.50), (1.7, 2.1, 0.2))
    add((-3.2, 4.45, 0.70), (1.9, 0.12, 1.4))
    for x in (-3.65, -2.85):
        add((x, 4.0, 0.65), (0.55, 0.4, 0.1))
    add((-5.2, 3.7, 1.0), (1.0, 1.5, 2.0))
    add((-1.65, 4.0, 0.30), (0.55, 0.55, 0.6))

    # Southeast: L-shaped kitchen counter, island and refrigerator.
    add((3.4, -4.45, 0.45), (4.2, 0.7, 0.9))
    add((5.45, -2.75, 0.45), (0.7, 2.7, 0.9))
    add((3.3, -2.4, 0.45), (1.1, 1.4, 0.9))
    add((5.25, -0.55, 0.95), (1.0, 0.9, 1.9))

    # Northeast: desk with legs, and an open bookcase with four shelves.
    add((4.2, 3.55, 0.78), (1.8, 0.8, 0.08))
    for dx in (-0.76, 0.76):
        for dy in (-0.26, 0.26):
            add((4.2 + dx, 3.55 + dy, 0.37), (0.08, 0.08, 0.74))
    for x in (1.44, 2.96):
        add((x, 4.45, 0.9), (0.08, 0.4, 1.8))
    for z in (0.04, 0.60, 1.16, 1.76):
        add((2.2, 4.45, z), (1.44, 0.4, 0.08))
    add((2.2, 4.61, 0.9), (1.44, 0.08, 1.8))

    if include_ceiling:
        add((0, 0, 2.9), (12, 10, 0.2))

    return boxes
