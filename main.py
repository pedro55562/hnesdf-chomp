"""Run h-NESDF CHOMP on the underwater scene and save the animation as HTML."""

import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import uaibot as ub

from chomp import chomp_se3, generate_poses

ROOT = Path(__file__).resolve().parent
TRAIL_SIZE = 15


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

    results_info = []
    no_success = 0
    path_success = []
    time_success = 0

    for n in range(100):
        try:
            H_start, H_goal = generate_poses(pc, lx, ly, lz)

            start_time = perf_counter()

            paths, info = chomp_se3(
                H_start, H_goal,
                lx=lx, ly=ly, lz=lz,
                h=0.1, list_points=pc_points,
                N=30,
                threshold=0.6,  # 0.3 for N=60
                G=25.0,
                softmin_h=0.1,
                clearance=0.04,
                iterations=20,
                step_size=0.2,
                proximity_gain=0,
                proximity_epsilon=0.04,
                momentum=0.95,
                batch_size=None,  # Evaluate all N+1 poses in one call.
                save_every=1,
                check_substeps=10,
                gradient_method="autograd",
                fd_steps=(1e-3, 1e-3),
                echo=False
            )

            elapsed_time = perf_counter() - start_time

            results_info.append([info['success'], elapsed_time])
            print(results_info[-1])
            if results_info[-1][0]:
                no_success += 1
                time_success += results_info[-1][1]
                path_success = paths
                break

        except Exception:
            pass

    print(no_success / 30)

    if not path_success:
        print("No successful path found; skipping visualization.")
        return
    print("iterations =", info['iterations'])

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

    tot_j = len(path_success[0])
    for k in range(len(path_success)):
        for j in range(TRAIL_SIZE):
            box_trail[j].add_ani_frame(k, path_success[k][round(j * (tot_j - 4) / 14) + 3])

    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    sim.save(str(output), "result")
    print("Saved", output / "result.html")


if __name__ == "__main__":
    main()
