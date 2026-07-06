"""
Experiment 3: Empirical Oracle-Style Comparison on Real Image Patches from Set12

The SURE-based rank selection rule should perform close to the oracle
across a range of noise levels and image content.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils")); from patch_utils import build_patch_collection, IMAGE_NAMES

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def limiting_scores(sv, m, n, tau):
    k = len(sv)
    sv2 = sv ** 2
    S0 = np.sum(sv2) - m * n * tau**2
    diff = sv2[:, None] - sv2[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(diff) > 1e-15, sv2[:, None] / diff, 0.0)
    np.fill_diagonal(ratio, 0.0)
    cum_double = np.cumsum(ratio.sum(axis=1))
    scores = np.empty(k + 1)
    scores[0] = S0
    for h in range(1, k + 1):
        residual = np.sum(sv2[h:])
        dof = abs(m - n) * h + h + 2.0 * cum_double[h - 1]
        scores[h] = -m * n * tau**2 + residual + 2.0 * tau**2 * dof
    return scores


def true_mse_all_ranks(Y, X):
    U, sv, Vt = np.linalg.svd(Y, full_matrices=False)
    k = len(sv)
    mse = np.empty(k + 1)
    mse[0] = np.sum(X ** 2)
    X_hat = np.zeros_like(X)
    for h in range(k):
        X_hat = X_hat + sv[h] * np.outer(U[:, h], Vt[h, :])
        mse[h + 1] = np.sum((X_hat - X) ** 2)
    return mse


def mp_rank(sv, m, n, tau):
    """Energy-matching rank selection (Guo et al. motivated)."""
    sv2 = sv ** 2
    noise_energy = m * n * tau ** 2
    target = np.sum(sv2) - noise_energy
    if target <= 0:
        return 0
    cumulative = np.cumsum(sv2)
    h = int(np.searchsorted(cumulative, target)) + 1
    return min(h, len(sv))


def run_one_tau(X, tau, N_MC, rng):
    m, n = X.shape
    k = min(m, n)
    mse_oracle = np.empty(N_MC)
    mse_sure = np.empty(N_MC)
    mse_mp = np.empty(N_MC)
    h_hat_arr = np.empty(N_MC, dtype=int)
    h_orc_arr = np.empty(N_MC, dtype=int)
    h_mp_arr = np.empty(N_MC, dtype=int)
    for t in range(N_MC):
        W = tau * rng.standard_normal((m, n))
        Y = X + W
        _, sv, _ = np.linalg.svd(Y, full_matrices=False)
        mse_all = true_mse_all_ranks(Y, X)
        h_orc = int(np.argmin(mse_all))
        h_orc_arr[t] = h_orc
        mse_oracle[t] = mse_all[h_orc]
        scores = limiting_scores(sv, m, n, tau)
        h_hat = int(np.argmin(scores))
        h_hat_arr[t] = h_hat
        mse_sure[t] = mse_all[h_hat]
        h_mp = mp_rank(sv, m, n, tau)
        h_mp_arr[t] = h_mp
        mse_mp[t] = mse_all[h_mp]
    return dict(
        mean_oracle=float(np.mean(mse_oracle)),
        mean_sure=float(np.mean(mse_sure)),
        mean_mp=float(np.mean(mse_mp)),
        eff_sure=float(np.mean(mse_oracle) / max(np.mean(mse_sure), 1e-15)),
        eff_mp=float(np.mean(mse_oracle) / max(np.mean(mse_mp), 1e-15)),
        h_hat=h_hat_arr, h_orc=h_orc_arr, h_mp=h_mp_arr, k=k,
    )


def run_experiment3(collection, tau_grid, N_MC, seed=2024):
    rng = np.random.default_rng(seed)
    all_results = []
    for entry in collection:
        print(f"  {entry['image']} pos={entry['query_pos']}")
        query_by_tau = {}
        for tau in tau_grid:
            query_by_tau[tau] = run_one_tau(entry['X'], tau, N_MC, rng)
        all_results.append({
            'image': entry['image'], 'pos': entry['query_pos'],
            'query_by_tau': query_by_tau,
        })
    return all_results


def print_summary(all_results, tau_grid):
    print("\n" + "=" * 95)
    print("Experiment 3 — Oracle-Style Comparison Summary")
    print("=" * 95)
    print(f"{'Image':>8} {'Pos':>12} {'τ':>7} "
          f"{'oracle':>10} {'SURE':>10} {'EM(Guo)':>10} {'eff_SURE':>10} {'eff_EM':>10}")
    print("-" * 95)
    all_eff_sure, all_eff_mp = [], []
    for pr in all_results:
        for tau in tau_grid:
            r = pr['query_by_tau'][tau]
            all_eff_sure.append(r['eff_sure'])
            all_eff_mp.append(r['eff_mp'])
            print(f"{pr['image'].replace('.png',''):>8} "
                  f"{str(pr['pos']):>12} "
                  f"{tau:>7.4f} "
                  f"{r['mean_oracle']:>10.5f} "
                  f"{r['mean_sure']:>10.5f} "
                  f"{r['mean_mp']:>10.5f} "
                  f"{r['eff_sure']:>10.4f} "
                  f"{r['eff_mp']:>10.4f}")
    print("=" * 95)
    print(f"\nOverall mean oracle efficiency — SURE: {np.mean(all_eff_sure):.4f}")
    print(f"Overall mean oracle efficiency — Energy matching (Guo-style): {np.mean(all_eff_mp):.4f}")
    print(f"SURE > Energy matching in {sum(s>m for s,m in zip(all_eff_sure,all_eff_mp))}/{len(all_eff_sure)} cases")


if __name__ == "__main__":
    print("=" * 65)
    print("Experiment 3: Oracle-Style Comparison — Set12 Patches")
    print("=" * 65)
    PATCH_SIZE = 32; STRIDE = 4; N_QUERY = 3; K_SIMILAR = 50
    N_MC = 500; SEED = 2024
    PSNR_GRID = np.array([15, 18, 20, 22, 25, 28, 30])
    TAU_GRID = 10 ** (-PSNR_GRID / 20.0)
    IMAGES = ["01.png","02.png","03.png","04.png",
              "05.png","06.png","07.png","08.png"]
    print(f"\nPatch: {PATCH_SIZE}×{PATCH_SIZE}, stride: {STRIDE}")
    print(f"PSNR grid: {PSNR_GRID} dB → τ = {TAU_GRID.round(4)}")
    print(f"N_MC = {N_MC} per τ\n")
    print("Extracting patches...")
    collection = build_patch_collection(
        image_names=IMAGES, patch_size=PATCH_SIZE, stride=STRIDE,
        n_query_per_image=N_QUERY, k_similar=K_SIMILAR,
        rng=np.random.default_rng(SEED), verbose=True,
    )
    print(f"\nTotal patch groups: {len(collection)}\n")
    print("Running oracle comparison...")
    all_results = run_experiment3(collection, TAU_GRID, N_MC=N_MC, seed=SEED)
    print_summary(all_results, TAU_GRID)
    print("\nExperiment 3 complete.")
