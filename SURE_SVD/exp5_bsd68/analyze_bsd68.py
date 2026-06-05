"""
analyze_bsd68.py
────────────────
Statistical analysis of BSD68 Energy matching vs SURE experiment.
68 images × 3 noise levels = 204 fully independent paired trials.
Wilcoxon signed-rank test for paired comparison.

Usage: python analyze_bsd68.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import sys
import platform
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def wilcoxon_safe(x, alternative='two-sided'):
    """Wilcoxon signed-rank test with safety checks."""
    x = x[~np.isnan(x)]
    x_nonzero = x[x != 0]
    if len(x_nonzero) < 10:
        return np.nan, 'insufficient data'
    try:
        stat, p = stats.wilcoxon(x_nonzero, alternative=alternative)
        return p, f'{p:.2e}' if p < 0.001 else f'{p:.4f}'
    except Exception as e:
        return np.nan, str(e)


def analyze_group(df, label):
    """Compute statistics for a group of paired trials."""
    n = len(df)
    delta_psnr = df['psnr_sure'].values - df['psnr_energy'].values
    delta_ssim = df['ssim_sure'].values - df['ssim_energy'].values
    time_ratio = df['time_sure'].values / np.maximum(df['time_energy'].values, 1e-10)

    result = {
        'label': label,
        'N': n,
        'psnr_energy_mean': np.nanmean(df['psnr_energy']),
        'psnr_sure_mean': np.nanmean(df['psnr_sure']),
        'delta_psnr_mean': np.nanmean(delta_psnr),
        'delta_psnr_std': np.nanstd(delta_psnr),
        'delta_psnr_median': np.nanmedian(delta_psnr),
        'win_rate_psnr': np.nanmean(delta_psnr > 0),
        'ssim_energy_mean': np.nanmean(df['ssim_energy']),
        'ssim_sure_mean': np.nanmean(df['ssim_sure']),
        'delta_ssim_mean': np.nanmean(delta_ssim),
        'delta_ssim_std': np.nanstd(delta_ssim),
        'win_rate_ssim': np.nanmean(delta_ssim > 0),
        'time_energy_mean': np.nanmean(df['time_energy']),
        'time_sure_mean': np.nanmean(df['time_sure']),
        'time_ratio_mean': np.nanmean(time_ratio),
        'time_ratio_std': np.nanstd(time_ratio),
    }

    # Wilcoxon tests (two-sided)
    p_psnr, p_psnr_str = wilcoxon_safe(delta_psnr)
    p_ssim, p_ssim_str = wilcoxon_safe(delta_ssim)
    result['p_psnr'] = p_psnr
    result['p_psnr_str'] = p_psnr_str
    result['p_ssim'] = p_ssim
    result['p_ssim_str'] = p_ssim_str

    return result


def main():
    csv_path = os.path.join(OUT_DIR, 'results_bsd68.csv')
    if not os.path.exists(csv_path):
        print(f'ERROR: {csv_path} not found.')
        print('Run run_bsd68_experiment.m in MATLAB first.')
        sys.exit(1)

    df = pd.read_csv(csv_path).dropna()
    print(f'Loaded {len(df)} paired trials from {csv_path}')

    version_info = (
        f'Python {sys.version.split()[0]}, '
        f'NumPy {np.__version__}, '
        f'SciPy {stats.scipy.__version__}, '
        f'Pandas {pd.__version__}, '
        f'Platform: {platform.platform()}'
    )

    # Analyze by sigma
    results = []
    for sigma in sorted(df['sigma'].unique()):
        group = df[df['sigma'] == sigma]
        results.append(analyze_group(group, f'σ={sigma}'))

    # Analyze pooled
    results.append(analyze_group(df, 'All'))

    # Print table
    print(f'\n{version_info}\n')
    print('=' * 130)
    print('STATISTICAL COMPARISON: SURE vs Energy matching on BSD68 (Wilcoxon Signed-Rank Test)')
    print('=' * 130)
    print(f'{"Setting":<8} {"N":>4} | '
          f'{"PSNR_E":>7} {"PSNR_S":>7} {"ΔPSNR±std":>12} {"med":>6} {"Win%":>5} {"p-value":>10} | '
          f'{"SSIM_E":>7} {"SSIM_S":>7} {"ΔSSIM±std":>14} {"Win%":>5} {"p-value":>10}')
    print('-' * 130)

    for r in results:
        print(
            f'{r["label"]:<8} {r["N"]:>4} | '
            f'{r["psnr_energy_mean"]:>7.2f} {r["psnr_sure_mean"]:>7.2f} '
            f'{r["delta_psnr_mean"]:>+5.2f}±{r["delta_psnr_std"]:<5.2f} '
            f'{r["delta_psnr_median"]:>+6.3f} {r["win_rate_psnr"]*100:>5.1f} {r["p_psnr_str"]:>10} | '
            f'{r["ssim_energy_mean"]:>7.4f} {r["ssim_sure_mean"]:>7.4f} '
            f'{r["delta_ssim_mean"]:>+7.4f}±{r["delta_ssim_std"]:<6.4f} '
            f'{r["win_rate_ssim"]*100:>5.1f} {r["p_ssim_str"]:>10}'
        )

    print('=' * 130)

    # Runtime table
    print('\n' + '=' * 80)
    print('RUNTIME (implementation-dependent indicative runtime)')
    print('=' * 80)
    print(f'{"Setting":<8} {"N":>4} | {"T_Energy (s)":>14} | {"T_SURE (s)":>14} | {"Ratio ± std":>16}')
    print('-' * 80)
    for r in results:
        print(f'{r["label"]:<8} {r["N"]:>4} | {r["time_energy_mean"]:>14.1f} | '
              f'{r["time_sure_mean"]:>14.1f} | {r["time_ratio_mean"]:>6.3f} ± {r["time_ratio_std"]:<6.3f}')
    print('=' * 80)

    # Save
    out_path = os.path.join(OUT_DIR, 'table_bsd68_statistical.txt')
    # Re-run print to file
    import io
    buf = io.StringIO()
    buf.write(version_info + '\n\n')
    buf.write('STATISTICAL COMPARISON: SURE vs Energy matching on BSD68\n')
    buf.write(f'Wilcoxon Signed-Rank Test (two-sided), 68 independent images, seed=rng(img_idx*100+sigma)\n')
    buf.write(f'Noise levels: sigma = {sorted(df["sigma"].unique().tolist())}\n\n')
    for r in results:
        buf.write(f'{r["label"]}: N={r["N"]}, ΔPSNR={r["delta_psnr_mean"]:+.3f}±{r["delta_psnr_std"]:.3f}, '
                  f'win={r["win_rate_psnr"]*100:.1f}%, p={r["p_psnr_str"]}, '
                  f'ΔSSIM={r["delta_ssim_mean"]:+.4f}±{r["delta_ssim_std"]:.4f}, '
                  f'win={r["win_rate_ssim"]*100:.1f}%, p={r["p_ssim_str"]}\n')
    with open(out_path, 'w') as f:
        f.write(buf.getvalue())
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
