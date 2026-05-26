"""
Experiment 2: Post-Selection Bias in the Plug-in Ranking Step
             on Real Image Patches from Set12

Theoretical basis (Section 4):
  - For FIXED deterministic lambda: E[SURE] = E[MSE] (unbiased).
  - After DATA-ADAPTIVE selection: E[SURE_selected] < E[MSE_selected].
  - But the selected rank h_hat should still be close to the oracle rank.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def run_one_patch(X, tau, N_MC, rng):
    m, n = X.shape
    k = min(m, n)
    sure_sel = np.empty(N_MC)
    mse_sel = np.empty(N_MC)
    mse_orac = np.empty(N_MC)
    h_hat_arr = np.empty(N_MC, dtype=int)
    h_orc_arr = np.empty(N_MC, dtype=int)
    all_scores = np.empty((N_MC, k + 1))
    all_true_mse = np.empty((N_MC, k + 1))
    for t in range(N_MC):
        W = tau * rng.standard_normal((m, n))
        Y = X + W
        _, sv, _ = np.linalg.svd(Y, full_matrices=False)
        scores = limiting_scores(sv, m, n, tau)
        mse_all = true_mse_all_ranks(Y, X)
        all_scores[t] = scores
        all_true_mse[t] = mse_all
        h_hat = int(np.argmin(scores))
        h_orc = int(np.argmin(mse_all))
        h_hat_arr[t] = h_hat
        h_orc_arr[t] = h_orc
        sure_sel[t] = scores[h_hat]
        mse_sel[t] = mse_all[h_hat]
        mse_orac[t] = mse_all[h_orc]
    return dict(
        sure_sel=sure_sel, mse_sel=mse_sel, mse_orac=mse_orac,
        h_hat=h_hat_arr, h_orc=h_orc_arr,
        all_scores=all_scores, all_true_mse=all_true_mse, k=k, m=m, n=n,
    )


def run_experiment2(collection, tau, N_MC, seed=2024):
    rng = np.random.default_rng(seed)
    all_results = []
    for entry in collection:
        res_query = run_one_patch(entry['X'], tau, N_MC, rng)
        n_alt = min(5, len(entry['similar']))
        alt_results = []
        for i in range(n_alt):
            alt_results.append(run_one_patch(entry['similar'][i], tau, N_MC // 2, rng))
        all_results.append({
            'image': entry['image'], 'pos': entry['query_pos'],
            'ncc_scores': entry['ncc_scores'][:n_alt],
            'query_res': res_query, 'alt_res': alt_results,
        })
    return all_results


def print_summary(all_results, tau, N_MC):
    print("\n" + "=" * 80)
    print("Experiment 2 — Post-Selection Bias Summary")
    print("=" * 80)
    print(f"{'Image':>8} {'Pos':>12} "
          f"{'E[SURE_sel]':>12} {'E[MSE_sel]':>12} {'E[oracle]':>11} "
          f"{'gap':>8} {'eff':>7}")
    print("-" * 80)
    gaps, effs = [], []
    for pr in all_results:
        res = pr['query_res']
        ms = np.mean(res['sure_sel'])
        mm = np.mean(res['mse_sel'])
        mo = np.mean(res['mse_orac'])
        gap = mm - ms
        eff = mo / max(mm, 1e-15)
        gaps.append(gap); effs.append(eff)
        print(f"{pr['image'].replace('.png',''):>8} "
              f"{str(pr['pos']):>12} "
              f"{ms:>12.5f} {mm:>12.5f} {mo:>11.5f} "
              f"{gap:>8.5f} {eff:>7.4f}")
    print("=" * 80)
    print(f"\nMean post-selection gap = {np.mean(gaps):.5f}  "
          f"({'positive = SURE optimistic ✓' if np.mean(gaps) > 0 else 'negative ✗'})")
    print(f"Mean oracle efficiency  = {np.mean(effs):.4f}")


if __name__ == "__main__":
    print("=" * 65)
    print("Experiment 2: Post-Selection Bias — Set12 Patches")
    print("=" * 65)
    PATCH_SIZE = 32; STRIDE = 4; N_QUERY = 3; K_SIMILAR = 50
    TAU = 25 / 255; N_MC = 1000; SEED = 2024
    IMAGES = ["01.png","02.png","03.png","04.png",
              "05.png","06.png","07.png","08.png"]
    print(f"\nτ = {TAU:.4f}, N_MC = {N_MC}, patch = {PATCH_SIZE}×{PATCH_SIZE}")
    print("Extracting patches...")
    collection = build_patch_collection(
        image_names=IMAGES, patch_size=PATCH_SIZE, stride=STRIDE,
        n_query_per_image=N_QUERY, k_similar=K_SIMILAR,
        rng=np.random.default_rng(SEED), verbose=True,
    )
    print(f"\nTotal patch groups: {len(collection)}\n")
    print("Running Monte Carlo...")
    all_results = run_experiment2(collection, tau=TAU, N_MC=N_MC, seed=SEED)
    print_summary(all_results, TAU, N_MC)
    print("\nExperiment 2 complete.")
