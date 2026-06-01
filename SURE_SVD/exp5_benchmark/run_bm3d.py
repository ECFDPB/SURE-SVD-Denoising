"""
run_bm3d.py
───────────
Run BM3D on saved noisy images from MATLAB and merge all results into
a single JSON file (results_set12.json).

Prerequisites:
  1. Run run_set12_benchmark.m in MATLAB first (generates noisy_images/*.mat
     and results_set12_matlab.mat).
  2. pip install bm3d scipy numpy

Usage:
  python run_bm3d.py
"""
import numpy as np
import scipy.io as sio
import bm3d as bm3d_pkg
from scipy.ndimage import convolve
import os
import json

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NOISY_DIR = os.path.join(OUT_DIR, 'noisy_images')

IMG_NAMES = [f'{i:02d}.png' for i in range(1, 13)]
SIGMAS = [10, 30, 50]


def gaussian_kernel(size, sigma):
    x = np.arange(-(size // 2), size // 2 + 1)
    X, Y = np.meshgrid(x, x)
    h = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return h / h.sum()


def compute_ssim(img1, img2):
    """SSIM matching MATLAB's fspecial('gaussian',11,1.5) + conv2(...,'same')."""
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    win = gaussian_kernel(11, 1.5)
    mu1 = convolve(img1, win, mode='reflect')
    mu2 = convolve(img2, win, mode='reflect')
    s1 = convolve(img1**2, win, mode='reflect') - mu1**2
    s2 = convolve(img2**2, win, mode='reflect') - mu2**2
    s12 = convolve(img1 * img2, win, mode='reflect') - mu1 * mu2
    ssim_map = ((2*mu1*mu2 + C1) * (2*s12 + C2)) / ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))
    return float(np.mean(ssim_map))


def compute_psnr(clean, denoised):
    mse = np.mean((clean - denoised)**2)
    return 10 * np.log10(255**2 / mse) if mse > 1e-10 else 100.0


def run_bm3d_all():
    """Run BM3D on all saved noisy images."""
    bm3d_results = {}
    print('Running BM3D on Set12 noisy images...\n')
    for img_name in IMG_NAMES:
        for sigma in SIGMAS:
            mat_path = os.path.join(NOISY_DIR, f'{img_name[:-4]}_sigma{sigma}.mat')
            if not os.path.exists(mat_path):
                print(f'  SKIP {img_name} sigma={sigma} (no .mat file)')
                continue
            data = sio.loadmat(mat_path)
            noisy = data['noisy'].astype(np.float64)
            clean = data['clean'].astype(np.float64)

            denoised = bm3d_pkg.bm3d(noisy / 255.0, sigma_psd=sigma / 255.0) * 255.0

            p = compute_psnr(clean, denoised)
            s = compute_ssim(clean, denoised)
            bm3d_results[f'{img_name}_sigma{sigma}'] = {'psnr': round(p, 2), 'ssim': round(s, 4)}
            print(f'  {img_name} sigma={sigma}: PSNR={p:.2f}, SSIM={s:.4f}')

    return bm3d_results


def merge_results(bm3d_results):
    """Merge MATLAB results + BM3D results into a single JSON."""
    mat_path = os.path.join(OUT_DIR, 'results_set12_matlab.mat')
    if not os.path.exists(mat_path):
        print(f'\nWARNING: {mat_path} not found. Saving BM3D-only results.')
        combined = {'bm3d': bm3d_results}
    else:
        mat = sio.loadmat(mat_path)
        combined = {}
        for i, img_name in enumerate(IMG_NAMES):
            for j, sigma in enumerate(SIGMAS):
                key = f'{img_name}_sigma{sigma}'
                entry = {}
                # MATLAB methods
                for method, psnr_key, ssim_key in [
                    ('sure', 'psnr_sure', 'ssim_sure'),
                    ('energy_matching', 'psnr_em', 'ssim_em'),
                    ('ksvd', 'psnr_ksvd', 'ssim_ksvd'),
                    ('lpgpca', 'psnr_lpg', 'ssim_lpg'),
                ]:
                    p = float(mat[psnr_key][i, j]) if not np.isnan(mat[psnr_key][i, j]) else None
                    s = float(mat[ssim_key][i, j]) if not np.isnan(mat[ssim_key][i, j]) else None
                    entry[method] = {'psnr': round(p, 2) if p is not None else None,
                                     'ssim': round(s, 4) if s is not None else None}
                # BM3D
                if key in bm3d_results:
                    entry['bm3d'] = bm3d_results[key]
                else:
                    entry['bm3d'] = {'psnr': None, 'ssim': None}
                combined[key] = entry

    return combined


def main():
    bm3d_results = run_bm3d_all()

    # Merge all results
    combined = merge_results(bm3d_results)

    # Save unified results
    out_path = os.path.join(OUT_DIR, 'results_set12.json')
    with open(out_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f'\nSaved unified results: {out_path}')

    # Print summary
    print('\n' + '=' * 70)
    print('SUMMARY (average PSNR over all images and noise levels)')
    print('=' * 70)
    methods = ['ksvd', 'lpgpca', 'energy_matching', 'sure', 'bm3d']
    for method in methods:
        vals = [combined[k][method]['psnr'] for k in combined if combined[k][method]['psnr'] is not None]
        if vals:
            print(f'  {method:>16s}: {np.mean(vals):.2f} dB (n={len(vals)})')
    print('=' * 70)


if __name__ == '__main__':
    main()
