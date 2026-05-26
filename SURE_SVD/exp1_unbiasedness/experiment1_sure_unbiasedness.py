"""
Experiment 1: Monte Carlo Verification of Fixed-Threshold SURE Unbiasedness
             on Real Image Patches from Set12

Theoretical basis (Proposition 2.2):
    For fixed deterministic omega > 0 and lambda >= 0,
        E[SURE_{omega,lambda}(Y)] = E[||X_hat_{omega,lambda}(Y) - X||^2_F]
    where Y = X + W, W_{ij} ~ i.i.d. N(0, tau^2).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils")); from patch_utils import build_patch_collection, IMAGE_NAMES

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def smooth_threshold(x, omega, lam):
    z = np.clip(-omega * (x - lam), -500, 500)
    return x / (1.0 + np.exp(z))


def smooth_threshold_deriv(x, omega, lam):
    z = np.clip(omega * (x - lam), -500, 500)
    sig = 1.0 / (1.0 + np.exp(-z))
    return sig + omega * x * sig * (1.0 - sig)


def compute_sure(Y, tau, omega, lam):
    m, n = Y.shape
    U, sv, Vt = np.linalg.svd(Y, full_matrices=False)
    f = smooth_threshold(sv, omega, lam)
    fp = smooth_threshold_deriv(sv, omega, lam)
    residual = np.sum((sv - f) ** 2)
    div1 = abs(m - n) * np.sum(f / np.maximum(sv, 1e-15))
    div2 = np.sum(fp)
    sv2 = sv ** 2
    diff = sv2[:, None] - sv2[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(diff) > 1e-15, (sv * f)[:, None] / diff, 0.0)
    np.fill_diagonal(ratio, 0.0)
    div3 = 2.0 * np.sum(ratio)
    return float(-m * n * tau**2 + residual + 2 * tau**2 * (div1 + div2 + div3))


def compute_mse(Y, X, omega, lam):
    U, sv, Vt = np.linalg.svd(Y, full_matrices=False)
    f = smooth_threshold(sv, omega, lam)
    return float(np.sum(((U * f[None, :]) @ Vt - X) ** 2))


def run_one_patch(X, tau, lam_list, omega_list, N_MC, rng):
    m, n = X.shape
    results = {}
    for lam in lam_list:
        for omega in omega_list:
            sure_v = np.empty(N_MC)
            mse_v = np.empty(N_MC)
            for t in range(N_MC):
                W = tau * rng.standard_normal((m, n))
                Y = X + W
                sure_v[t] = compute_sure(Y, tau, omega, lam)
                mse_v[t] = compute_mse(Y, X, omega, lam)
            diff_v = sure_v - mse_v
            se_paired = float(np.std(diff_v) / np.sqrt(N_MC))
            results[(lam, omega)] = dict(
                mean_sure=float(np.mean(sure_v)),
                mean_mse=float(np.mean(mse_v)),
                se_paired=se_paired,
                sure_vals=sure_v,
                mse_vals=mse_v,
                bias=float(np.mean(diff_v)),
                rel_bias=float(np.mean(diff_v) / max(abs(np.mean(mse_v)), 1e-15)),
            )
    return results


def run_experiment1(collection, tau, omega_list, N_MC, seed=2024):
    rng = np.random.default_rng(seed)
    all_results = []
    for entry in collection:
        X = entry['X']
        sv_X = entry['sv_full']
        P = entry['patch_size']
        k = min(P, P)
        m, n = P, P
        noise_scale = tau * (np.sqrt(m) + np.sqrt(n))
        lam_list = []
        lam_list.append(float(noise_scale * 0.8))
        lam_list.append(float(noise_scale * 1.2))
        if k >= 2:
            lam_list.append(float((sv_X[0] + sv_X[1]) / 2.0))
        lam_list.append(float(sv_X[0] * 1.2))
        lam_list = sorted(set(round(l, 6) for l in lam_list))
        patch_res = run_one_patch(X, tau, lam_list, omega_list, N_MC, rng)
        all_results.append({
            'image': entry['image'],
            'pos': entry['query_pos'],
            'patch_size': P,
            'lam_list': lam_list,
            'results': patch_res,
        })
    return all_results


def print_summary(all_results, omega_list):
    print("\n" + "=" * 95)
    print("Experiment 1 — Fixed-Threshold SURE Unbiasedness on Set12 Patches")
    print("=" * 95)
    print(f"{'Image':>8} {'Pos':>12} {'λ':>8} {'ω':>7} "
          f"{'E[SURE]':>10} {'E[MSE]':>10} {'bias':>10} {'SE_paired':>10} {'|t|':>7} {'OK':>5}")
    print("-" * 95)
    n_pass = n_total = 0
    for pr in all_results:
        for lam in pr['lam_list']:
            for omega in omega_list:
                if (lam, omega) not in pr['results']:
                    continue
                r = pr['results'][(lam, omega)]
                t_stat = abs(r['bias']) / max(r['se_paired'], 1e-15)
                ok = t_stat < 1.96
                n_pass += int(ok)
                n_total += 1
                print(
                    f"{pr['image'].replace('.png',''):>8} "
                    f"{str(pr['pos']):>12} "
                    f"{lam:>8.4f} {omega:>7.1f} "
                    f"{r['mean_sure']:>10.4f} {r['mean_mse']:>10.4f} "
                    f"{r['bias']:>10.4f} {r['se_paired']:>10.4f} "
                    f"{t_stat:>7.2f} "
                    f"{'✓' if ok else '✗':>5}"
                )
    print("=" * 95)
    print(f"\nPASS rate: {n_pass}/{n_total} = {100*n_pass/max(n_total,1):.1f}%")
    print("(Two-sided test at 95% level: |bias| / SE_paired < 1.96)")
    print("(Expected PASS rate under H₀: ~95%)\n")


if __name__ == "__main__":
    print("=" * 65)
    print("Experiment 1: Fixed-Threshold SURE Unbiasedness — Set12 Patches")
    print("=" * 65)

    PATCH_SIZE = 32
    STRIDE = 4
    N_QUERY = 3
    K_SIMILAR = 50
    TAU = 25 / 255
    N_MC = 1000
    OMEGA_LIST = [1.0, 5.0, 20.0, 100.0]
    SEED = 2024
    IMAGES = ["01.png", "02.png", "03.png", "04.png",
              "05.png", "06.png", "07.png", "08.png"]

    print(f"\nPatch size: {PATCH_SIZE}×{PATCH_SIZE}, stride: {STRIDE}")
    print(f"Noise std τ = {TAU:.4f} (≈ {-20*np.log10(TAU):.1f} dB PSNR)")
    print(f"N_MC = {N_MC}, ω values = {OMEGA_LIST}")
    print(f"Images: {IMAGES}\n")

    print("Extracting patches...")
    collection = build_patch_collection(
        image_names=IMAGES, patch_size=PATCH_SIZE, stride=STRIDE,
        n_query_per_image=N_QUERY, k_similar=K_SIMILAR,
        rng=np.random.default_rng(SEED), verbose=True,
    )
    print(f"\nTotal patch groups: {len(collection)}\n")

    print("Running Monte Carlo...")
    all_results = run_experiment1(
        collection, tau=TAU, omega_list=OMEGA_LIST, N_MC=N_MC, seed=SEED
    )
    print_summary(all_results, OMEGA_LIST)
    print("Experiment 1 complete.")
