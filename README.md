# hnesdf-chomp

h-NESDF + CHOMP path optimization for a box in an underwater point-cloud scene.
Script version of `h-nesdf_chomp_underwater.ipynb`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python main.py
```

The animation is saved to `output/result.html`; open it in a browser.

## What is implemented

| File | Contents |
|---|---|
| `hnesdf.py` | `h_esdf_box`: smooth (log-sum-exp) SDF of a box at query points. `hNESDF`: smooth box-to-oriented-point-cloud distance for a batch of box poses, plus its 6D pose differential (autograd or central finite differences). |
| `chomp.py` | `chomp_se3`: CHOMP-style path optimizer on SE(3) (waypoint poses of the box). `path_objective`: spacing barrier + soft-min clearance reward + optional CHOMP proximity penalty. `_interpolate`/`interpolate_poses`: linear translation + SLERP. `generate_poses`: random start/goal sampling with rejection. |
| `environment.py` | `sample_box_environment` (oriented point cloud from a set of boxes) and `create_house_environment` (a furnished house scene). **Not used by `main.py`**; the scene comes from the `.npz`. |
| `main.py` | Loads the cloud, samples random start/goal pairs, runs `chomp_se3` until the first success (up to 100 attempts), saves the uaibot animation. |

### How it works

1. **Distance (`hNESDF`)**: for B poses and K cloud points, it evaluates `h_esdf_box` on a `(B, K)` grid. For each pose, a softmax over the K points (temperature `h²`) weights the points near the box. The distance is the SDF at the box center, corrected by the weighted projection of the spatial gradient onto the point normals. The pose differential comes from a second autograd pass (`create_graph=True`, i.e. second-order autograd), mapped to `[v, ω]` in the world frame.
2. **Objective (`path_objective`)**: `sum log(1 - s_k/threshold²)` keeps consecutive waypoints closer than `threshold`. `-G·τ·logsumexp(-d/τ)` rewards the minimum clearance. An optional term (`proximity_gain`) adds the CHOMP penalty, weighted by arc length. `main.py` uses `proximity_gain=0`.
3. **Optimizer (`chomp_se3`)**:
   - Nesterov-style momentum on SE(3), using `matrix_exp` for rotations.
   - Optional temporal-Laplacian preconditioning (Cholesky solve).
   - Armijo backtracking (up to 24 halvings, 2 restarts).
   - Endpoints stay fixed.
   - At the end, the path is densified (`check_substeps`) and the h-NESDF clearance is checked on the dense path. This is a sampled check, not a continuous collision certificate.
4. **Start/goal (`generate_poses`)**: random poses in `[-4.5, 4.5]² × [0.5, 2.5]`. A pair is accepted if the poses are at least 2 m apart, both endpoints are at least 0.3 m from the cloud, and the straight-line path collides (to force a non-trivial problem). The distances come from uaibot's exact `Box.compute_dist` (GJK, C++ backend).

### Data (`underwater_complex.npz`)

The file stores `point_cloud` (82 445 × 6: points + normals), `box_lengths` (0.9 × 0.5 × 0.35 m), `H_start`/`H_goal` and metadata (`scene_json`, `seed`, …). `main.py` keeps the points with `z ≥ -0.2`, which leaves **K = 54 570**. It does **not** use the stored `H_start`/`H_goal`; the poses are random on every run.

## GPU vs CPU

| Stage | Device | Notes |
|---|---|---|
| `h_esdf_box`, `hNESDF` (distance + gradient) | **GPU** (whatever device the tensors are on) | Pure torch, vectorized over B × K. |
| `path_objective` (barrier, soft-min, penalty) | **GPU** | Small tensors (N+1 poses). |
| `_move` (SE(3) update, `matrix_exp`), Cholesky preconditioning | **GPU** | Tiny (N−1 ≤ 29). |
| `_interpolate` (initial path, dense check path) | **CPU** (NumPy/SciPy `Slerp`) | Copied to the GPU afterwards. |
| Optimizer control flow (iterations, line search, restarts) | **CPU** (Python) | Each evaluation calls `float(...)` twice, so it syncs the GPU. |
| `generate_poses` (sampling + `compute_dist`) | **CPU** (uaibot C++) | About 10 ms per pair; it is not a bottleneck. |
| `sample_box_environment`, `create_house_environment` | **CPU** (NumPy) | Not used. |
| Visualization (`ub.Simulation`, HTML) | **CPU** | A separate copy of the cloud lives on the CPU. |

Everything runs in float32. With no CUDA available, `main.py` falls back to the CPU.

## Parallelism

- **Parallel (GPU)**: within one `hNESDF` call, all `B × K` box-point pairs run as single tensor ops.
- **Sequential**:
  - The `batch_size` chunks in `path_objective` and in the dense check are evaluated in a Python loop.
  - The line-search trials run one after another (up to 24 per restart, plus 2 restarts).
  - The optimizer iterations run one after another (this one is inherent).
  - The benchmark attempts in `main.py` run one after another. The loop stops at the first success.
  - The finite-difference gradient makes 12 extra evaluations, one by one.

### Measured cost (RTX 4050 Laptop, float32, K = 54 570)

| B (poses) | with gradient | distance only |
|---|---|---|
| 1 | 3.5 ms | 1.5 ms |
| 30 | 81 ms | 30 ms |
| 31 | 80 ms | 31 ms |
| 128 | 353 ms | 139 ms |

- **Scaling**: the time grows about linearly with B, so the GPU is already saturated at B ≈ 30. Larger batches will not make things faster; reducing K will.
- **Memory**: B = 128 with gradient peaks at about 3 GB of VRAM.
- **Finite differences** with B = 1: 24 ms (vs 3.5 ms with autograd).
- **Typical run**: one attempt takes 0.5–6 s. The first attempt also pays for CUDA warm-up, and `perf_counter` does not separate it.

## Potential improvements

### Performance

- **Prune the point cloud.** The softmax weights with temperature `h²` make far points irrelevant. Keep only the points within some margin of the path's bounding box, or use a KD-tree / voxel grid around each pose. The cost scales with K, so this is the biggest gain available.
- **Fix the chunking.** `batch_size=30` with `N+1 = 31` poses creates a 30-pose chunk plus a 1-pose chunk. Use `batch_size=None` or `N+1`.
- **Skip the fixed endpoints.** Their distance and gradient are computed every evaluation and then thrown away.
- **Batch the line search.** Evaluate several step sizes α in one batched call instead of halving one by one.
- **Batch the problems.** Run several start/goal pairs at once (benchmarking, multi-start).
- **Reduce GPU syncs.** Avoid the `float(...)` calls inside `path_objective` on every evaluation.
- **Fuse kernels.** `torch.compile` or CUDA graphs can cut launch overhead for the many small kernels.
- **Densify on the GPU.** Run `_interpolate` in torch to avoid the CPU round trip in the dense check.
- **Time it correctly.** Do a warm-up before timing, and call `torch.cuda.synchronize()` around the measurement.

### Correctness and experiment hygiene

- **The benchmark metric is wrong.** `main.py` stops at the first success and prints `no_success/30`, even though there are up to 100 attempts. To get a success rate, remove the `break` and divide by the number of attempts.
- **Silent errors.** `except Exception: pass` in `main.py` and `except Exception: cont = True` in `generate_poses` hide every error. If something always raises, `generate_poses` loops forever.
- **No seeds.** The runs are not reproducible. The `.npz` even ships a `seed` and a fixed start/goal pair that go unused.
- **Unused code.** `environment.py` is not used. `hNESDF`'s finite-difference path checks `list_points` only inside the recursive call.
- **Not a continuous certificate.** `info['success']` is a sampled check on the densified path, not a guarantee on the swept volume.
