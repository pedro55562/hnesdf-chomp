"""Batched h-NESDF path optimization with Nesterov-style SE(3) updates."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
import uaibot as ub

from hnesdf import hNESDF


def _interpolate(poses, subdivisions):
    """Linear translation and shortest-arc SLERP on each segment."""
    poses = np.asarray(poses)
    times = np.arange(len(poses), dtype=float)
    query = np.linspace(0, times[-1], (len(poses) - 1) * subdivisions + 1)
    result = np.repeat(np.eye(4)[None], len(query), axis=0)
    result[:, :3, :3] = Slerp(
        times, Rotation.from_matrix(poses[:, :3, :3])
    )(query).as_matrix()
    for j in range(3):
        result[:, j, 3] = np.interp(query, times, poses[:, j, 3])
    result[[0, -1]] = poses[[0, -1]]
    return result


@torch.no_grad()
def _move(H, increments):
    """World-frame increments [dt,domega], with t_dot=v, R_dot=[w] R."""
    w = increments[:, 3:]
    W = H.new_zeros((len(H), 3, 3))
    W[:, 0, 1], W[:, 1, 0] = -w[:, 2], w[:, 2]
    W[:, 0, 2], W[:, 2, 0] = w[:, 1], -w[:, 1]
    W[:, 1, 2], W[:, 2, 1] = -w[:, 0], w[:, 0]
    result = H.clone()
    result[:, :3, :3] = torch.matrix_exp(W) @ H[:, :3, :3]
    result[:, :3, 3] += increments[:, :3]
    result[[0, -1]] = H[[0, -1]]
    return result


@torch.enable_grad()
def path_objective(H, lx, ly, lz, h, cloud, threshold, G=10.0,
                   softmin_h=None, rotation_scale=1.0, batch_size=16,
                   gradient=True, gradient_method="autograd",
                   fd_steps=(1e-3, 1e-3), proximity_gain=10.0,
                   proximity_epsilon=0.05):
    """Return (objective, distances, world-frame gradient, max_step).

    Invalid spacing returns (-inf, None, None, max_step). The gradient
    has shape (N+1,6), with fixed-endpoint rows zero. Tensor outputs are
    detached and remain on H.device; objective/max_step are Python floats.
    CHOMP's proximity penalty is integrated along pose arc length with
    trapezoidal weights. G retains the original soft-min clearance reward.
    """
    if (not np.isfinite(proximity_gain) or proximity_gain < 0
            or not np.isfinite(proximity_epsilon) or proximity_epsilon <= 0):
        raise ValueError("proximity_gain must be nonnegative and proximity_epsilon positive.")
    tau = h if softmin_h is None else softmin_h
    X = H.detach().clone().requires_grad_(gradient)
    dt = X[1:, :3, 3] - X[:-1, :3, 3]
    dR = X[1:, :3, :3] - X[:-1, :3, :3]
    s = dt.square().sum(-1) + rotation_scale**2 * dR.square().sum((-1, -2))
    max_step = float(s.detach().max().sqrt())
    if max_step >= threshold:
        return -float("inf"), None, None, max_step
    reward = torch.log1p(-s / threshold**2).sum()

    distances, differentials = [], []
    chunk = len(X) if batch_size is None else batch_size
    for part in X.split(chunk):
        d, L = hNESDF(part, lx, ly, lz, h, cloud,
                      return_tensors=True, compute_gradient=gradient,
                      gradient_method=gradient_method, fd_steps=fd_steps)
        distances.append(d)
        if gradient:
            differentials.append(L)
    d = torch.cat(distances)
    base = reward
    distance_weight = G * torch.softmax(-d / tau, dim=0)
    if proximity_gain > 0:
        eps = proximity_epsilon
        rho = torch.where(d < 0, -d + eps / 2,
                          (eps - d).clamp_min(0).square() / (2 * eps))
        # This norm has a finite zero subgradient at a repeated waypoint.
        length = torch.linalg.vector_norm(
            torch.cat((dt, rotation_scale * dR.flatten(1)), dim=1), dim=1)
        base = base - proximity_gain * (0.5 * (rho[:-1] + rho[1:]) * length).sum()
        # Each waypoint receives half the arc length of its adjacent edges.
        weights = d.new_zeros(d.shape)
        weights[:-1] += length.detach() / 2
        weights[1:] += length.detach() / 2
        minus_rho_prime = (1 - d / eps).clamp(0, 1)
        distance_weight += proximity_gain * weights * minus_rho_prime
    f = base - G * tau * torch.logsumexp(-d / tau, dim=0)

    grad = None
    if gradient:
        A = torch.autograd.grad(base, X)[0]
        M = A[:, :3, :3] @ X[:, :3, :3].transpose(-1, -2)
        angular = torch.stack((M[:, 2, 1] - M[:, 1, 2],
                               M[:, 0, 2] - M[:, 2, 0],
                               M[:, 1, 0] - M[:, 0, 1]), -1)
        grad = torch.cat((A[:, :3, 3], angular), -1).detach()
        grad += distance_weight[:, None] * torch.cat(differentials)
        grad[[0, -1]] = 0
    return float(f.detach()), d, grad, max_step


def chomp_se3(H_start, H_goal, lx, ly, lz, h, list_points, N=40, *,
              threshold=0.4, G=10.0, iterations=300, step_size=0.05,
              momentum=0.8, clearance=0.05, softmin_h=None,
              rotation_scale=1.0, batch_size=16, save_every=1,
              check_substeps=5, initial_path=None, precondition=True,
              stop_when_clear=True, gradient_method="autograd",
              fd_steps=(1e-3, 1e-3), proximity_gain=10.0,
              proximity_epsilon=None, echo = True):
    """Plan a collision-avoiding SE(3) path for a box with a CHOMP-like optimizer.

    The box (full side lengths lx, ly, lz) must go from H_start to H_goal
    through an environment given as an oriented point cloud. The path is
    discretized into N segments, i.e. N+1 poses H[0..N]. The endpoints
    H[0]=H_start and H[N]=H_goal are fixed; the N-1 inner poses are
    optimized by gradient ascent with Nesterov-style momentum on SE(3).
    Clearance is measured with the h-NESDF between the box and the cloud
    (see hnesdf.hNESDF). This is a LOCAL optimizer: it can fail from a
    poor initial path.

    Objective (maximized; paper eqs. 31-33)
    ---------------------------------------
        f = sum_k log(1 - s_k / threshold**2)            spacing barrier
            - G * tau * logsumexp_k(-d_k / tau)          soft-min clearance
            - proximity_gain * C                         optional CHOMP term

    - s_k = ||t[k+1]-t[k]||^2 + rotation_scale**2 * ||R[k+1]-R[k]||_F^2 is the
      squared pose distance between consecutive poses. If any sqrt(s_k)
      reaches threshold, the path is infeasible and f = -inf, so every
      accepted step keeps all consecutive poses closer than threshold.
    - d_k is the h-NESDF of pose k; tau = softmin_h (or h if None). The
      logsumexp term is a smooth approximation of the minimum clearance.
    - C = sum_k sqrt(s_k) * (rho(d_k) + rho(d_{k+1})) / 2 is CHOMP's proximity
      penalty integrated along the pose arc length (trapezoidal rule), with
      rho(d) = -d + eps/2 for d < 0, (eps - d)**2 / (2*eps) for 0 <= d <= eps,
      and 0 otherwise; eps = proximity_epsilon. Use proximity_gain=0 to drop
      it (as in the paper), or G=0 to use only this penalty plus the barrier.

    Mapping to the paper's Table IV: N = T, G = C_weight,
    threshold = C_threshold, rotation_scale**2 = C_factor, momentum = beta,
    step_size = Delta t.

    Algorithm
    ---------
    1. Initialization: initial_path, or linear translation plus
       shortest-arc SLERP between the endpoints. Raises ValueError if the
       initial path already violates the spacing bound, or if an endpoint
       has h-NESDF below clearance.
    2. Each iteration (up to `iterations`):
       a. If stop_when_clear and every pose has d_k >= clearance, stop.
       b. Look-ahead: Y = H moved by momentum * velocity. Evaluate f and its
          world-frame gradient [v, w] (6 per pose) at Y; endpoint rows are 0.
       c. Direction: the gradient, optionally preconditioned with a temporal
          Laplacian (precondition=True), which spreads each update smoothly
          along the path. Rotation rows are divided by rotation_scale**2, and
          the direction is scaled so that no pose moves more than 1 per unit step.
       d. Backtracking line search from H with increments
          momentum * velocity + alpha * direction. alpha starts at the last
          accepted step (at most step_size) and is halved up to 24 times
          until Armijo holds: f(trial) >= f(Y) + 1e-4 * alpha * slope.
       e. If the momentum attempt fails, retry once with zero momentum. If
          that also fails, stop with status "stalled".
       Poses are updated with the exponential map (rotation R <- exp([w]) R,
       translation t <- t + v), so they always stay valid SE(3) poses.
    3. Final check: the final path is densified with `check_substeps`
       substeps per segment (same linear/SLERP interpolation), and the
       h-NESDF is evaluated on every dense pose.

    Parameters
    ----------
    H_start, H_goal : (4,4) array-like
        Fixed start and goal poses (homogeneous transforms, world frame).
    lx, ly, lz : float
        Full side lengths of the box.
    h : float
        h-NESDF smoothing parameter (bandwidth).
    list_points : (K,6) tensor
        Environment cloud [px, py, pz, nx, ny, nz] (points + outward unit
        normals, world frame). Its device and dtype set where everything runs.
    N : int
        Number of segments (N+1 poses, N-1 free poses). N >= 2.
    threshold : float
        Maximum pose distance between consecutive poses (spacing barrier).
    G : float
        Weight of the soft-min clearance reward.
    iterations : int
        Maximum number of optimizer iterations.
    step_size : float
        Maximum line-search step alpha.
    momentum : float in [0, 1)
        Nesterov momentum coefficient.
    clearance : float
        Required h-NESDF margin, for early stopping and for success.
    softmin_h : float or None
        Temperature tau of the soft-min; None uses h.
    rotation_scale : float
        Length scale that weights rotation against translation in s_k.
    batch_size : int or None
        Poses per hNESDF call; None evaluates all poses at once.
    save_every : int
        Store the path every `save_every` iterations (for animation).
    check_substeps : int
        Interpolation substeps per segment in the final dense check.
    initial_path : (N+1,4,4) array or tensor, optional
        Custom initial path; its endpoints are replaced by H_start/H_goal.
    precondition : bool
        Use temporal-Laplacian preconditioning.
    stop_when_clear : bool
        Stop as soon as every pose reaches clearance. If False, keep
        optimizing until `iterations` or a stall.
    gradient_method : {"autograd", "finite_difference"}
        How the h-NESDF pose gradient is computed. Finite differences use
        central differences with fd_steps = (translation step, rotation step
        in radians); the spacing-barrier gradient is always exact.
    proximity_gain : float
        Weight of the CHOMP proximity penalty (0 disables it).
    proximity_epsilon : float or None
        eps of rho(d); None uses clearance if positive, else h.
    echo : bool
        Print the iteration number and minimum distance at each iteration.

    Returns
    -------
    paths : list of (N+1,4,4) NumPy arrays
        Saved paths, including the endpoints. paths[0] is the initial path
        and paths[-1] the final one (always saved).
    info : dict
        success : bool
            True if every dense pose has finite h-NESDF >= clearance.
            This is a SAMPLED check with the smoothed distance itself: it is
            NOT an exact or continuous (swept-volume) collision certificate.
        status : str
            "dense_sampled_clearance_reached" (success),
            "dense_check_failed" (all N+1 poses clear, a dense pose is not),
            "max_iterations" or "stalled" (stopped before clearing).
        iterations : int
            Completed iterations.
        saved_iterations : list of int
            Iteration index of each entry of `paths`.
        objective_history, min_distance_history, max_step_history : lists
            f, min_k d_k and max_k sqrt(s_k) after each accepted iteration.
        final_distances : (N+1,) array
            h-NESDF of each pose of the final path.
        dense_path : (N*check_substeps+1,4,4) array
            Densified final path used by the success check.
        dense_distances : array
            h-NESDF of each dense pose.
        min_dense_distance : float
            Minimum of dense_distances.
        clearance, threshold, proximity_gain, proximity_epsilon
            Values used in this run.

    Raises
    ------
    ValueError
        Invalid parameters, an initial path that violates the spacing bound,
        an endpoint below clearance, or an initial_path of the wrong shape.
    """
    tau = h if softmin_h is None else softmin_h
    eps = (clearance if clearance > 0 else h) if proximity_epsilon is None else proximity_epsilon
    if (N < 2 or int(N) != N or min(lx, ly, lz, h, tau, threshold,
                                    step_size, rotation_scale) <= 0
            or G < 0 or not 0 <= momentum < 1 or iterations < 0
            or save_every < 1 or check_substeps < 1
            or (batch_size is not None and batch_size < 1)):
        raise ValueError("Invalid dimensions, bandwidth, or optimization parameters.")
    if gradient_method not in ("autograd", "finite_difference"):
        raise ValueError("Unknown gradient_method.")
    if (not np.isfinite(proximity_gain) or proximity_gain < 0
            or not np.isfinite(eps) or eps <= 0):
        raise ValueError("proximity_gain must be nonnegative and proximity_epsilon positive.")
    N = int(N)
    cloud = torch.as_tensor(list_points).detach()
    if not cloud.is_floating_point():
        cloud = cloud.float()
    start = torch.as_tensor(H_start, dtype=cloud.dtype, device=cloud.device)
    goal = torch.as_tensor(H_goal, dtype=cloud.dtype, device=cloud.device)
    endpoints = torch.stack((start, goal)).detach().cpu().numpy()
    if isinstance(initial_path, torch.Tensor):
        initial_path = initial_path.detach().cpu().numpy()
    seed = _interpolate(endpoints, N) if initial_path is None else np.asarray(initial_path)
    if seed.shape != (N + 1, 4, 4):
        raise ValueError("initial_path must have shape (N+1,4,4).")
    H = torch.as_tensor(seed.copy(), dtype=cloud.dtype, device=cloud.device)
    H[[0, -1]] = torch.stack((start, goal))
    velocity = torch.zeros((N + 1, 6), dtype=H.dtype, device=H.device)

    # A positive-definite discrete Laplacian on the free waypoints.
    n = N - 1
    A = 2 * torch.eye(n, dtype=H.dtype, device=H.device)
    A += torch.diag(-torch.ones(n - 1, dtype=H.dtype, device=H.device), 1)
    A += torch.diag(-torch.ones(n - 1, dtype=H.dtype, device=H.device), -1)
    chol = torch.linalg.cholesky(A) if precondition else None

    def evaluate(X, gradient):
        return path_objective(X, lx, ly, lz, h, cloud, threshold, G,
                              tau, rotation_scale, batch_size, gradient,
                              gradient_method, fd_steps, proximity_gain, eps)

    f, d, _, max_step = evaluate(H, False)
    if d is None:
        raise ValueError(f"Initial max step is {max_step:.6g}; increase N or threshold.")
    if float(d[[0, -1]].min()) < clearance:
        raise ValueError("A fixed endpoint is below the requested h-NESDF clearance.")
    paths = [H.detach().cpu().numpy().copy()]
    saved_iterations = [0]
    objectives, minimums, maximums = [f], [float(d.min())], [max_step]
    status, completed, next_step = "max_iterations", 0, step_size

    for iteration in range(1, iterations + 1):
        if echo:
          print("i = "+str(iteration)+", d_min = "+str(float(d.min())))

        if stop_when_clear and float(d.min()) >= clearance:
            status = "waypoint_clearance_reached"
            break
        accepted = False
        for restart in range(2):
            inertia = momentum * velocity if restart == 0 else torch.zeros_like(velocity)
            Y = _move(H, inertia)
            fy, _, grad, _ = evaluate(Y, True)
            if grad is None or not np.isfinite(fy) or fy < f:
                continue
            direction = grad.clone()
            if precondition:
                direction[1:-1] = torch.cholesky_solve(direction[1:-1], chol)
            direction[:, 3:] /= rotation_scale**2
            norm = (direction[:, :3].square().sum(-1)
                    + rotation_scale**2 * direction[:, 3:].square().sum(-1)).sqrt().max()
            direction /= norm.clamp_min(1.0)
            slope = float((grad * direction).sum())
            if not np.isfinite(slope) or slope <= 1e-14:
                continue
            alpha = next_step
            for _ in range(24):
                increments = inertia + alpha * direction
                trial = _move(H, increments)
                ft, dt, _, mt = evaluate(trial, False)
                if np.isfinite(ft) and ft >= fy + 1e-4 * alpha * slope:
                    H, velocity, f, d, max_step = trial, increments, ft, dt, mt
                    next_step = min(step_size, 1.2 * alpha)
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                break
        if not accepted:
            status = "stalled"
            break
        completed = iteration
        objectives.append(f)
        minimums.append(float(d.min()))
        maximums.append(max_step)
        if iteration % save_every == 0:
            paths.append(H.detach().cpu().numpy().copy())
            saved_iterations.append(iteration)

    if saved_iterations[-1] != completed:
        paths.append(H.detach().cpu().numpy().copy())
        saved_iterations.append(completed)
    final = H.detach().cpu().numpy().copy()
    dense_path = _interpolate(final, check_substeps)
    dense_T = torch.as_tensor(dense_path, dtype=H.dtype, device=H.device)
    chunk = len(dense_T) if batch_size is None else batch_size
    dense_distances = torch.cat([
        hNESDF(part, lx, ly, lz, h, cloud,
               return_tensors=True, compute_gradient=False)[0]
        for part in dense_T.split(chunk)
    ]).cpu().numpy()
    success = bool(np.isfinite(dense_distances).all()
                   and dense_distances.min() >= clearance)
    if success:
        status = "dense_sampled_clearance_reached"
    elif float(d.min()) >= clearance:
        status = "dense_check_failed"
    info = dict(success=success, status=status, iterations=completed,
                saved_iterations=saved_iterations, objective_history=objectives,
                min_distance_history=minimums, max_step_history=maximums,
                final_distances=d.cpu().numpy(), dense_path=dense_path,
                dense_distances=dense_distances,
                min_dense_distance=float(dense_distances.min()),
                clearance=clearance, threshold=threshold,
                proximity_gain=proximity_gain, proximity_epsilon=eps)
    return paths, info

def interpolate_poses(H_start, H_goal, N):
    """Return N intermediate poses as np.matrix, excluding endpoints."""
    if N == 0:
        return []

    A, B = np.asarray(H_start), np.asarray(H_goal)
    t = np.linspace(0, 1, N + 2)[1:-1]

    rotations = Rotation.from_matrix(
        np.stack([A[:3, :3], B[:3, :3]])
    )
    H = np.tile(np.eye(4), (N, 1, 1))
    H[:, :3, :3] = Slerp([0, 1], rotations)(t).as_matrix()
    H[:, :3, 3] = (
        (1 - t[:, None]) * A[:3, 3]
        + t[:, None] * B[:3, 3]
    )

    return [np.matrix(pose) for pose in H]

def generate_poses(_pc, _lx, _ly, _lz):
  box_test = ub.Box(width=_lx, depth=_ly, height=_lz)
  cont = True
  while cont:

    try:
      H_start = ub.Utils.htm_rand([-4.5,-4.5,0.5],[4.5,4.5,2.5],rot=np.pi/2)
      H_goal = ub.Utils.htm_rand([-4.5,-4.5,0.5],[4.5,4.5,2.5],rot=np.pi/2)



      cont = np.linalg.norm(H_start[:3,3]-H_goal[:3,3])<2.0

      if not cont:
        box_test.set_ani_frame(htm=H_start)
        _, _, dist, _ = box_test.compute_dist(_pc)
        cont = dist<0.3

      if not cont:
        box_test.set_ani_frame(htm=H_goal)
        _, _, dist, _ = box_test.compute_dist(_pc)
        cont = dist<0.3

      if not cont:
        poses = interpolate_poses(H_start, H_goal, 30)
        j = 0
        cont_internal = True
        while cont_internal:
          j=j+1
          box_test.set_ani_frame(htm=poses[j])
          _, _, dist, _ = box_test.compute_dist(_pc)
          cont_internal =  (dist >= 1e-4) and (j<30)

        cont = dist>= 1e-4

    except Exception as e:
      cont = True

  return H_start, H_goal
