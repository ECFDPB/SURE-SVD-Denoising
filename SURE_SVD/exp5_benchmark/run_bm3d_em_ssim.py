"""
Run BM3D on saved noisy images and compute PSNR/SSIM using the SAME
SSIM function as MATLAB (Gaussian 11x11, sigma=1.5, conv2 'same').
Also save denoised images for Energy matching SSIM computation in MATLAB.
"""
import numpy as np
import scipy.io as sio
import bm3d as bm3d_pkg
from scipy.ndimage import convolve
import os, json

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
noisy_dir = os.path.join(OUT_DIR, 'noisy_images')

img_names = [f'{i:02d}.png' for i in range(1, 13)]
sigmas = [10, 20, 30, 50]


def gaussian_kernel(size, sigma):
    x = np.arange(-(size//2), size//2 + 1)
    X, Y = np.meshgrid(x, x)
    h = np.exp(-(X**2 + Y**2) / (2*sigma**2))
    return h / h.sum()


def compute_ssim_matlab_style(img1, img2):
    """SSIM matching MATLAB's fspecial('gaussian',11,1.5) + conv2(...,'same')"""
    C1 = (0.01*255)**2
    C2 = (0.03*255)**2
    win = gaussian_kernel(11, 1.5)
    mu1 = convolve(img1, win, mode='reflect')
    mu2 = convolve(img2, win, mode='reflect')
    s1 = convolve(img1**2, win, mode='reflect') - mu1**2
    s2 = convolve(img2**2, win, mode='reflect') - mu2**2
    s12 = convolve(img1*img2, win, mode='reflect') - mu1*mu2
    ssim_map = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))
    return float(np.mean(ssim_map))


def compute_psnr(clean, denoised):
    mse = np.mean((clean - denoised)**2)
    return 10*np.log10(255**2/mse) if mse > 1e-10 else 100.0


# ── Run BM3D ──────────────────────────────────────────────────────────────
bm3d_results = {}
print('BM3D:')
for img_name in img_names:
    for sigma in sigmas:
        data = sio.loadmat(os.path.join(noisy_dir, f'{img_name[:-4]}_sigma{sigma}.mat'))
        noisy = data['noisy'].astype(np.float64)
        clean = data['clean'].astype(np.float64)

        denoised = bm3d_pkg.bm3d(noisy/255.0, sigma_psd=sigma/255.0) * 255.0

        p = compute_psnr(clean, denoised)
        s = compute_ssim_matlab_style(clean, denoised)
        bm3d_results[f'{img_name}_sigma{sigma}'] = {'psnr': round(p, 2), 'ssim': round(s, 4)}
        print(f'  {img_name} sigma={sigma}: PSNR={p:.2f}, SSIM={s:.4f}')

with open(os.path.join(OUT_DIR, 'bm3d_final.json'), 'w') as f:
    json.dump(bm3d_results, f, indent=2)
print('Saved: bm3d_final.json\n')
