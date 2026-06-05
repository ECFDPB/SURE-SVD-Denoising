"""
fig_data_construction.py
────────────────────────
Section 5 data-construction schematic.
Purpose: Show the reader how the test matrices X are built from Set12 images.
X = a single 32×32 query patch (full rank), NOT the grouped matrix.
Layout: horizontal pipeline, large figure.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_utils import load_image, extract_patches, find_top_k_similar

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_schematic(image_name='01.png', patch_size=32, stride=4,
                       n_similar=50, seed=2024, save_path=None):
    rng = np.random.default_rng(seed)
    P = patch_size

    # ── Load image and select a visually interesting query ─────────────────
    img = load_image(image_name)
    patches, positions = extract_patches(img, P, stride)
    variances = np.var(patches.reshape(len(patches), -1), axis=1)
    top_idx = np.argsort(variances)[-len(variances) // 5:]
    qi = rng.choice(top_idx)
    query = patches[qi]
    qr, qc = positions[qi]

    # ── Find top-50 similar patches ───────────────────────────────────────
    similar, ncc_vals = find_top_k_similar(query, patches, k=n_similar, exclude_idx=qi)

    # ── SVD of query patch ────────────────────────────────────────────────
    sv_query = np.linalg.svd(query, full_matrices=False)[1]

    # ══════════════════════════════════════════════════════════════════════
    # Figure: 5 logical columns
    #   [Source image] [Arrow] [Query patch] [Arrow] [Neighbours 3×2 + SV plot]
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(22, 8), dpi=300, facecolor='white')

    gs_main = GridSpec(1, 5, figure=fig,
                       width_ratios=[3.0, 0.6, 1.8, 0.6, 7.0],
                       wspace=0.06,
                       left=0.02, right=0.98, top=0.88, bottom=0.10)

    # ── Column 0: Set12 image with red box ────────────────────────────────
    ax_img = fig.add_subplot(gs_main[0])
    ax_img.imshow(img, cmap='gray', vmin=0, vmax=1)
    rect = Rectangle((qc - 0.5, qr - 0.5), P, P,
                     linewidth=3, edgecolor='red', facecolor='none', zorder=5)
    ax_img.add_patch(rect)
    ax_img.set_title('Set12 image\n(grayscale, [0, 1])', fontsize=16, fontweight='bold', pad=12)
    ax_img.axis('off')
    ax_img.text(qc + P/2, qr + P + 12, 'query', color='red', fontsize=14,
                ha='center', va='top', fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # ── Column 1: Arrow "extract" ─────────────────────────────────────────
    ax_a1 = fig.add_subplot(gs_main[1])
    ax_a1.axis('off')
    ax_a1.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                   xycoords='axes fraction', textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', lw=3, color='#2c3e50'))
    ax_a1.text(0.5, 0.56, 'extract', ha='center', va='bottom',
               fontsize=14, color='#2c3e50', transform=ax_a1.transAxes)

    # ── Column 2: Query patch enlarged ───────────────────────────────────
    ax_q = fig.add_subplot(gs_main[2])
    ax_q.imshow(query, cmap='gray', vmin=0, vmax=1)
    for spine in ax_q.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(3.5)
    ax_q.set_xticks([]); ax_q.set_yticks([])
    ax_q.set_title(r'Query patch $X$' + '\n' + r'$X \in \mathbb{R}^{32 \times 32}$',
                   fontsize=16, fontweight='bold', color='red', pad=12)
    ax_q.text(0.5, -0.08, '(signal matrix for\nMonte Carlo experiments)',
              ha='center', va='top', fontsize=13, color='#555555',
              transform=ax_q.transAxes)

    # ── Column 3: Arrow "NCC search" ──────────────────────────────────────
    ax_a2 = fig.add_subplot(gs_main[3])
    ax_a2.axis('off')
    ax_a2.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                   xycoords='axes fraction', textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', lw=3, color='#2c3e50'))
    ax_a2.text(0.5, 0.56, 'NCC\nsearch', ha='center', va='bottom',
               fontsize=14, color='#2c3e50', transform=ax_a2.transAxes)

    # ── Column 4: Neighbours (3×2 grid) + SV stem plot ───────────────────
    gs_right = GridSpec(1, 2, figure=fig,
                        width_ratios=[1.0, 0.9],
                        wspace=0.25,
                        left=gs_main[4].get_position(fig).x0,
                        right=gs_main[4].get_position(fig).x1,
                        top=gs_main[4].get_position(fig).y1,
                        bottom=gs_main[4].get_position(fig).y0)

    right_pos = gs_right[0].get_position(fig)
    rx0, ry0, rx1, ry1 = right_pos.x0, right_pos.y0, right_pos.x1, right_pos.y1
    n_show = 6
    n_cols, n_rows = 3, 2
    gap_x = 0.012
    gap_y = 0.06
    patch_w = (rx1 - rx0 - (n_cols - 1) * gap_x) / n_cols
    patch_h = patch_w * (fig.get_figwidth() / fig.get_figheight())
    total_h = n_rows * patch_h + (n_rows - 1) * gap_y
    y_start = (ry0 + ry1) / 2 + total_h / 2

    for idx in range(n_show):
        row = idx // n_cols
        col = idx % n_cols
        x = rx0 + col * (patch_w + gap_x)
        y = y_start - (row + 1) * patch_h - row * gap_y
        ax_p = fig.add_axes([x, y, patch_w, patch_h])
        ax_p.imshow(similar[idx], cmap='gray', vmin=0, vmax=1)
        for spine in ax_p.spines.values():
            spine.set_edgecolor('#2980b9')
            spine.set_linewidth(2.5)
        ax_p.set_xticks([]); ax_p.set_yticks([])
        ax_p.text(0.5, -0.06, f'NCC = {ncc_vals[idx]:.2f}',
                  ha='center', va='top', fontsize=12, color='#2980b9',
                  transform=ax_p.transAxes)

    fig.text((rx0 + rx1) / 2, y_start + 0.015,
             'Top-50 NCC neighbours\n(6 shown, from same image)',
             ha='center', va='bottom', fontsize=15, fontweight='bold', color='#2980b9')

    # ── SV stem plot ──────────────────────────────────────────────────────
    ax_sv = fig.add_subplot(gs_right[1])
    indices = np.arange(1, len(sv_query) + 1)
    markerline, stemlines, baseline = ax_sv.stem(indices, sv_query,
                                                  linefmt='k-', markerfmt='ko',
                                                  basefmt=' ')
    markerline.set_markersize(4)
    stemlines.set_linewidth(1.2)
    ax_sv.set_yscale('log')
    ax_sv.set_xlabel('Index $i$', fontsize=14)
    ax_sv.set_ylabel(r'$\sigma_i(X)$', fontsize=14)
    ax_sv.set_title('Singular values of $X$\n(diagnostic; full rank)',
                    fontsize=15, fontweight='bold', color='#555555', pad=12)
    ax_sv.set_xlim(0.5, 32.5)
    ax_sv.grid(True, alpha=0.3, axis='y')
    ax_sv.tick_params(labelsize=8)

    # ── Bottom annotation ─────────────────────────────────────────────────
    param_text = ('Set12 dataset  |  grayscale [0, 1]  |  '
                  'patch size 32\u00d732  |  stride 4  |  '
                  '3 queries/image  |  24 queries total  |  '
                  'top-50 NCC neighbours per query')
    fig.text(0.5, 0.02, param_text, ha='center', va='bottom',
             fontsize=14, color='#333333', style='italic')

    # ── Save ──────────────────────────────────────────────────────────────
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
        from PIL import Image as PILImage
        im = PILImage.open(save_path)
        print(f"  Pixel size: {im.size[0]} x {im.size[1]}")
    plt.close()


if __name__ == "__main__":
    print("Generating Section 5 data-construction schematic (Starfish)...")
    out = os.path.join(os.path.dirname(__file__), '..', '..', 'figures',
                       'fig_diagnostic_patch_group_schematic.png')
    generate_schematic(
        image_name='04.png',   # Starfish (top NCC similarity in Set12, mean=0.882)
        patch_size=32,
        stride=4,
        n_similar=50,
        seed=2024,
        save_path=out,
    )
    print("Done.")
