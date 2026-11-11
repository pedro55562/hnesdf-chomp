"""Run h-NESDF CHOMP on the underwater scene and save each animation as HTML."""

import gc
import shutil
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import uaibot as ub

from chomp import chomp_se3, generate_poses

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
N_SCENARIOS = 10  # Number of random start/goal scenarios to run and save.
SEED = 0  # Same seed -> same scenarios on every run.
TRAIL_SIZE = 15


def save_animation(folder, name, pc, H_start, H_goal, paths, lx, ly, lz):
    sim = ub.Simulation([], background_color="#009dc4")
    sim.add(pc)

    box_start = ub.Box(htm=H_start, width=lx, depth=ly, height=lz, color='magenta', opacity=1)
    box_goal = ub.Box(htm=H_goal, width=lx, depth=ly, height=lz, color='yellow', opacity=1)
    sim.add(box_start)
    sim.add(box_goal)

    box_trail = []
    for j in range(TRAIL_SIZE):
        box_trail.append(ub.Box(width=lx, depth=ly, height=lz, color='red', opacity=0.3))
        sim.add(box_trail[j])

    box_start.add_ani_frame(0, H_start)
    box_goal.add_ani_frame(0, H_goal)

    tot_j = len(paths[0])
    for k in range(len(paths)):
        for j in range(TRAIL_SIZE):
            box_trail[j].add_ani_frame(k, paths[k][round(j * (tot_j - 4) / 14) + 3])

    folder.mkdir(parents=True, exist_ok=True)
    sim.save(str(folder), name)


def main():
    # Check if CUDA is available, otherwise default to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)
    data = np.load(ROOT / "underwater_complex.npz", allow_pickle=False)
    pc_points = torch.as_tensor(data["point_cloud"], dtype=torch.float32, device=device)
    lx, ly, lz = data["box_lengths"]

    pc_points = pc_points[pc_points[:, 2] >= -0.2]
    pc_numpy = np.matrix(pc_points.detach().cpu().numpy()[:, :3]).T
    print(np.shape(pc_numpy))
    pc = ub.PointCloud(points=pc_numpy, color='white', size=0.05)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Previous results are replaced on every run.
    shutil.rmtree(OUTPUT, ignore_errors=True)

    np.random.seed(SEED)
    results = []  # (scenario, success, status, time, iterations, min dense distance)
    skipped = 0
    for n in range(N_SCENARIOS):
        H_start, H_goal = generate_poses(pc, lx, ly, lz)

        start_time = perf_counter()
        try:
            paths, info = chomp_se3(
                H_start, H_goal,
                lx=lx, ly=ly, lz=lz,
                h=0.08, list_points=pc_points,
                N=30,
                threshold=0.6,  # 0.3 for N=60
                G=25.0,
                softmin_h=0.1,
                clearance=0.04,
                iterations=40,
                step_size=0.3,
                proximity_gain=0,
                proximity_epsilon=0.04,
                momentum=0.65,
                batch_size=None,  # Evaluate all N+1 poses in one call.
                save_every=1,
                check_substeps=10,
                gradient_method="autograd",
                fd_steps=(1e-3, 1e-3),
                echo=False
            )
        except ValueError as e:
            print(f"scenario {n:03d}: skipped ({e})")
            skipped += 1
            continue
        elapsed_time = perf_counter() - start_time

        results.append((n, info['success'], info['status'], elapsed_time,
                        info['iterations'], info['min_dense_distance']))
        folder = OUTPUT / ("success" if info['success'] else "failure")
        name = f"scenario_{n:03d}"
        save_animation(folder, name, pc, H_start, H_goal, paths, lx, ly, lz)
        print(f"scenario {n:03d}: success={info['success']} time={elapsed_time:.2f}s "
              f"iterations={info['iterations']} -> {folder / name}.html")

    print_report(results, skipped)


def print_report(results, skipped):
    print("\n===== Report =====")
    print(f"seed: {SEED} | scenarios: {N_SCENARIOS} | run: {len(results)} | skipped: {skipped}")
    if not results:
        return
    success = [r for r in results if r[1]]
    failure = [r for r in results if not r[1]]
    print(f"success: {len(success)}/{len(results)} = {len(success) / len(results):.0%}")
    print("status: " + ", ".join(f"{k}={v}" for k, v in Counter(r[2] for r in results).items()))
    times = np.array([r[3] for r in results])
    print(f"time (s): median {np.median(times):.2f} | mean {times.mean():.2f} | max {times.max():.2f}"
          " (the first scenario includes torch.compile)")
    for label, group in (("success", success), ("failure", failure)):
        if group:
            print(f"{label}: mean iterations {np.mean([r[4] for r in group]):.1f} | "
                  f"mean min dense distance {np.mean([r[5] for r in group]):+.3f} m | "
                  f"scenarios {[r[0] for r in group]}")
    print(f"HTML files: {OUTPUT / 'success'} and {OUTPUT / 'failure'}")


if __name__ == "__main__":
    main()
