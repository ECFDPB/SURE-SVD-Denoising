"""
Generate the final comparison table as a publication-quality PNG image,
similar to TABLE I in the reference paper (example3).

Methods: BM3D, LPG-PCA, Energy matching (Guo-style), SURE (proposed)
Images: Set12 (12 images)
Noise levels: sigma = 10, 30, 50
"""
import numpy as np
import json
import os
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load all results ──────────────────────────────────────────────────────
# MATLAB results: Energy matching + SURE
mat = sio.loadmat(os.path.join(OUT_DIR, 'results_table.mat'))
psnr_em   = mat['psnr_results'][:, :, 1]   # (12, 3)
ssim_em   = mat['ssim_results'][:, :, 1]
psnr_sure = mat['psnr_results'][:, :, 2]
ssim_sure = mat['ssim_results'][:, :, 2]

# MATLAB results: LPG-PCA
mat2 = sio.loadmat(os.path.join(OUT_DIR, 'results_ksvd_lpgpca.mat'))
psnr_lpg = mat2['psnr_lpg']   # (12, 3)
ssim_lpg = mat2['ssim_lpg']

# Python results: BM3D
with open(os.path.join(OUT_DIR, 'bm3d_results.json')) as f:
    bm3d_data = json.load(f)

# Python results: K-SVD
with open(os.path.join(OUT_DIR, 'ksvd_results.json')) as f:
    ksvd_data = json.load(f)

img_names = [f'{i:02d}.png' for i in range(1, 13)]
sigmas = [10, 30, 50]

psnr_bm3d = np.zeros((12, 3))
ssim_bm3d = np.zeros((12, 3))
psnr_ksvd = np.zeros((12, 3))
ssim_ksvd = np.zeros((12, 3))
for i, name in enumerate(img_names):
    for j, sigma in enumerate(sigmas):
        key = f'{name}_sigma{sigma}'
        psnr_bm3d[i, j] = bm3d_data[key]['psnr']
        ssim_bm3d[i, j] = bm3d_data[key]['ssim']
        psnr_ksvd[i, j] = ksvd_data[key]['psnr']
        ssim_ksvd[i, j] = ksvd_data[key]['ssim']

# ── Combine into one structure ────────────────────────────────────────────
methods = ['K-SVD', 'LPG-PCA', 'BM3D', 'Energy matching', 'SURE (proposed)']
psnr_all = np.stack([psnr_ksvd, psnr_lpg, psnr_bm3d, psnr_em, psnr_sure], axis=2)
ssim_all = np.stack([ssim_ksvd, ssim_lpg, ssim_bm3d, ssim_em, ssim_sure], axis=2)

n_methods = len(methods)

# ── Print text table ──────────────────────────────────────────────────────
print('\n' + '='*120)
print('TABLE: COMPARISON OF PSNR (dB) AND SSIM OF DIFFERENT DENOISING METHODS')
print('WITH DIFFERENT NOISE LEVELS. THE BEST RESULTS ARE HIGHLIGHTED WITH *')
print('='*120)
header = f'{"Image":<8} {"σ":>3}'
for m in methods:
    header += f' | {"PSNR":>6} {"SSIM":>6}'
print(header)
print('-'*120)

for i, name in enumerate(img_names):
    for j, sigma in enumerate(sigmas):
        row = f'{name[:-4]:<8} {sigma:>3}'
        best_p = np.argmax(psnr_all[i, j, :])
        for mi in range(n_methods):
            p = psnr_all[i, j, mi]
            s = ssim_all[i, j, mi]
            mark = '*' if mi == best_p else ' '
            row += f' | {p:6.2f}{mark} {s:.4f}'
        print(row)

print('-'*120)
row = f'{"Average":<8} {"":>3}'
for mi in range(n_methods):
    avg_p = np.mean(psnr_all[:, :, mi])
    avg_s = np.mean(ssim_all[:, :, mi])
    row += f' | {avg_p:6.2f}  {avg_s:.4f}'
print(row)
print('='*120)

# ── Generate table as image ───────────────────────────────────────────────
# Build data for matplotlib table
col_labels = ['Image', 'σ'] + [f'{m}\nPSNR / SSIM' for m in methods]
n_rows = 12 * 3 + 1  # 36 data rows + 1 average row

cell_text = []
cell_colors = []

for i, name in enumerate(img_names):
    for j, sigma in enumerate(sigmas):
        row_data = [name[:-4] if j == 0 else '', str(sigma)]
        best_p = np.argmax(psnr_all[i, j, :])
        for mi in range(n_methods):
            p = psnr_all[i, j, mi]
            s = ssim_all[i, j, mi]
            row_data.append(f'{p:.2f} / {s:.4f}')
        cell_text.append(row_data)
        # Highlight best
        colors = ['white', 'white']
        for mi in range(n_methods):
            colors.append('#d4edda' if mi == best_p else 'white')
        cell_colors.append(colors)

# Average row
avg_row = ['Average', '']
for mi in range(n_methods):
    avg_p = np.mean(psnr_all[:, :, mi])
    avg_s = np.mean(ssim_all[:, :, mi])
    avg_row.append(f'{avg_p:.2f} / {avg_s:.4f}')
cell_text.append(avg_row)
cell_colors.append(['#f0f0f0'] * (2 + n_methods))

# Create figure
fig_height = 0.4 * (n_rows + 2)
fig, ax = plt.subplots(figsize=(14, fig_height))
ax.axis('off')
ax.set_title(
    'COMPARISON OF THE PSNR (dB) AND SSIM RESULTS OF DIFFERENT DENOISING METHODS\n'
    'WITH DIFFERENT NOISE LEVELS ON SET12 TEST IMAGES.\n'
    'THE BEST RESULTS ARE HIGHLIGHTED IN GREEN.',
    fontsize=11, fontweight='bold', pad=20
)

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellColours=cell_colors,
    colColours=['#e8e8e8'] * (2 + n_methods),
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.2)

# Bold the header
for key, cell in table.get_celld().items():
    if key[0] == 0:  # header row
        cell.set_text_props(fontweight='bold', fontsize=9)
    cell.set_edgecolor('#cccccc')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'table_psnr_ssim_comparison.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
print(f'\nSaved: table_psnr_ssim_comparison.png')
plt.close()

# ── Also save as CSV for LaTeX ────────────────────────────────────────────
with open(os.path.join(OUT_DIR, 'table_psnr_ssim.csv'), 'w') as f:
    f.write('Image,Noise level,' + ','.join([f'{m} PSNR,{m} SSIM' for m in methods]) + '\n')
    for i, name in enumerate(img_names):
        for j, sigma in enumerate(sigmas):
            row = f'{name[:-4]},{sigma}'
            for mi in range(n_methods):
                row += f',{psnr_all[i,j,mi]:.2f},{ssim_all[i,j,mi]:.4f}'
            f.write(row + '\n')
    # Average
    row = 'Average,'
    for mi in range(n_methods):
        row += f',{np.mean(psnr_all[:,:,mi]):.2f},{np.mean(ssim_all[:,:,mi]):.4f}'
    f.write(row + '\n')
print('Saved: table_psnr_ssim.csv')
