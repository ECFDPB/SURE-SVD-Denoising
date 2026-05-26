"""
Generate pipeline block diagram matching example1 style.
"SVD-Based Denoising" → "SURE-Ranked SVD Denoising" (highlighted).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def draw_box(ax, x, y, w, h, text, fontsize=11, bold=False, color='white', edgecolor='black'):
    """Draw a rounded box centred at (x, y)."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.03",
                          facecolor=color, edgecolor=edgecolor, linewidth=1.8)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight=weight)
    return (x - w/2, x + w/2, y - h/2, y + h/2)  # left, right, bottom, top


def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.8, color='black',
                                connectionstyle='arc3,rad=0'))


def draw_label_above(ax, x1, y1, x2, y2, text, fontsize=10, offset=0.15):
    """Place label just above the midpoint of an arrow."""
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2 + offset
    ax.text(mx, my, text, ha='center', va='bottom', fontsize=fontsize, style='italic')


def generate_pipeline(save_path=None):
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.set_xlim(-0.5, 15)
    ax.set_ylim(-0.2, 6.8)
    ax.axis('off')
    ax.set_aspect('equal')

    bw = 1.9   # box width
    bh = 0.85  # box height
    hw = bw / 2
    hh = bh / 2

    # ═══════════════════════════════════════════════════════════════════
    # Row 1 (top): y → Patch Grouping → SURE-Ranked SVD → Aggregation → x0
    # ═══════════════════════════════════════════════════════════════════
    r1 = 5.2  # row 1 y-centre

    # y (input label)
    y_x = 0.5
    ax.text(y_x, r1, r'$\mathbf{y}$', ha='center', va='center', fontsize=15, fontweight='bold')
    ax.text(y_x, r1 - 0.55, 'Noisy\nImage', ha='center', va='top', fontsize=9)

    # Patch Grouping
    pg1_x = 2.8
    L, R, B, T = draw_box(ax, pg1_x, r1, bw, bh, 'Patch\nGrouping')
    draw_arrow(ax, y_x + 0.25, r1, L, r1)

    # SURE-Ranked SVD
    svd1_x = 6.0
    L2, R2, B2, T2 = draw_box(ax, svd1_x, r1, 2.3, bh, 'SURE-Ranked\nSVD Denoising',
                               fontsize=10, bold=True, color='#dbeafe', edgecolor='#2563eb')
    draw_arrow(ax, R, r1, L2, r1)
    draw_label_above(ax, R, r1, L2, r1, r'$\{\mathbf{P}_j\}_{j=1}^C$', offset=0.12)

    # Aggregation
    agg1_x = 9.2
    L3, R3, B3, T3 = draw_box(ax, agg1_x, r1, bw, bh, 'Aggregation')
    draw_arrow(ax, R2, r1, L3, r1)
    draw_label_above(ax, R2, r1, L3, r1, r'$\{\widehat{\mathbf{Q}}_j\}_{j=1}^C$', offset=0.12)

    # x0
    x0_x = 11.5
    ax.text(x0_x, r1, r'$\widehat{\mathbf{x}}_0$', ha='center', va='center',
            fontsize=15, fontweight='bold')
    draw_arrow(ax, R3, r1, x0_x - 0.3, r1)

    # ═══════════════════════════════════════════════════════════════════
    # Back Projection (right side)
    # ═══════════════════════════════════════════════════════════════════
    bp_x = 13.0
    bp_y = 3.6
    L_bp, R_bp, B_bp, T_bp = draw_box(ax, bp_x, bp_y, bw, bh, 'Back\nProjection')

    # Arrow from x0 down-right to BP top
    draw_arrow(ax, x0_x + 0.3, r1, bp_x, T_bp)

    # Arrow from Noisy Image (y) down to BP left
    # Draw a line going down from y, then right to BP
    mid_y = bp_y
    ax.plot([y_x, y_x], [r1 - 0.55 - 0.4, mid_y], 'k-', lw=1.5)  # vertical down
    ax.plot([y_x, L_bp], [mid_y, mid_y], 'k-', lw=1.5)  # horizontal right
    ax.annotate('', xy=(L_bp, mid_y), xytext=(L_bp - 0.01, mid_y),
                arrowprops=dict(arrowstyle='->', lw=1.8, color='black'))

    # ═══════════════════════════════════════════════════════════════════
    # Row 2 (bottom): x_hat ← Aggregation ← SURE-SVD ← Patch Grouping ← y_tilde
    # ═══════════════════════════════════════════════════════════════════
    r2 = 1.8  # row 2 y-centre

    # y_tilde
    yt_x = 13.0
    ax.text(yt_x, r2, r'$\widetilde{\mathbf{y}}$', ha='center', va='center',
            fontsize=15, fontweight='bold')

    # Arrow from BP down to y_tilde
    draw_arrow(ax, bp_x, B_bp, yt_x, r2 + 0.3)

    # Patch Grouping 2
    pg2_x = 10.8
    L4, R4, B4, T4 = draw_box(ax, pg2_x, r2, bw, bh, 'Patch\nGrouping')
    draw_arrow(ax, yt_x - 0.3, r2, R4, r2)

    # SURE-Ranked SVD 2
    svd2_x = 7.6
    L5, R5, B5, T5 = draw_box(ax, svd2_x, r2, 2.3, bh, 'SURE-Ranked\nSVD Denoising',
                               fontsize=10, bold=True, color='#dbeafe', edgecolor='#2563eb')
    draw_arrow(ax, L4, r2, R5, r2)
    draw_label_above(ax, R5, r2, L4, r2, r'$\{\widetilde{\mathbf{P}}_j\}_{j=1}^C$', offset=0.12)

    # Aggregation 2
    agg2_x = 4.4
    L6, R6, B6, T6 = draw_box(ax, agg2_x, r2, bw, bh, 'Aggregation')
    draw_arrow(ax, L5, r2, R6, r2)
    draw_label_above(ax, L5, r2, R6, r2, r'$\{\widehat{\mathbf{Q}}_j\}_{j=1}^C$', offset=0.12)

    # x_hat (output)
    xhat_x = 1.8
    ax.text(xhat_x, r2, r'$\widehat{\mathbf{x}}$', ha='center', va='center',
            fontsize=15, fontweight='bold')
    ax.text(xhat_x, r2 - 0.55, 'Final\nEstimate', ha='center', va='top', fontsize=9)
    draw_arrow(ax, L6, r2, xhat_x + 0.3, r2)

    # ═══════════════════════════════════════════════════════════════════
    # Output arrow from x_hat to the left
    draw_arrow(ax, xhat_x - 0.25, r2, 0.5, r2)

    # ═══════════════════════════════════════════════════════════════════
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f'Saved: {save_path}')
    plt.close()


if __name__ == '__main__':
    generate_pipeline(save_path=os.path.join(OUT_DIR, 'fig_pipeline.png'))
