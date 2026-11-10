"""Measure hNESDF and end-to-end chomp_se3 speed on seeded problems.

Usage: python bench_speed.py [--problems 30] [--seed 0]
Runs on any branch, so results can be compared between implementations.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import uaibot as ub

from chomp import _interpolate, chomp_se3, generate_poses
from hnesdf import hNESDF

ROOT = Path(__file__).resolve().parent
PARAMS = dict(h=0.1, N=30, threshold=0.6, G=25.0, softmin_h=0.1,
              clearance=0.04, iterations=20, step_size=0.2,
              proximity_gain=0, proximity_epsilon=0.04, momentum=0.95,
              batch_size=None, save_every=1, check_substeps=10,
              gradient_method="autograd", echo=False)


def timed(f, repeats=10):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(ROOT / "underwater_complex.npz", allow_pickle=False)
    cloud = torch.as_tensor(data["point_cloud"], dtype=torch.float32, device="cuda")
    cloud = cloud[cloud[:, 2] >= -0.2]
    lx, ly, lz = map(float, data["box_lengths"])
    pc = ub.PointCloud(points=np.matrix(cloud.cpu().numpy()[:, :3]).T)
    print(f"GPU: {torch.cuda.get_device_name()} | K = {len(cloud)} points")

    np.random.seed(args.seed)
    problems = [generate_poses(pc, lx, ly, lz) for _ in range(args.problems + 1)]

    # Warm-up (includes torch.compile time on the first run of each shape).
    start = time.perf_counter()
    chomp_se3(*problems[0], lx, ly, lz, list_points=cloud, **PARAMS)
    torch.cuda.synchronize()
    print(f"warm-up run: {time.perf_counter() - start:.1f} s")

    poses = torch.as_tensor(_interpolate(np.stack([np.asarray(H) for H in problems[1]]), PARAMS["N"]),
                            dtype=torch.float32, device="cuda")
    for gradient in (True, False):
        t = timed(lambda: hNESDF(poses, lx, ly, lz, 0.1, cloud, return_tensors=True,
                                 compute_gradient=gradient))
        print(f"hNESDF B={len(poses)} {'with gradient' if gradient else 'distance only'}: {t * 1e3:.2f} ms")

    times, successes, iterations = [], [], []
    for H_start, H_goal in problems[1:]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        _, info = chomp_se3(H_start, H_goal, lx, ly, lz, list_points=cloud, **PARAMS)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        successes.append(info["success"])
        iterations.append(info["iterations"])

    times = np.array(times)
    print(f"chomp_se3 over {len(times)} problems: mean {times.mean():.3f} s | "
          f"median {np.median(times):.3f} s | max {times.max():.3f} s | "
          f"success {np.mean(successes):.0%} | mean iterations {np.mean(iterations):.1f}")
    print(f"ms per iteration: {1e3 * times.sum() / max(sum(iterations), 1):.1f}")


if __name__ == "__main__":
    main()
