# Smooth Hard-Thresholding for Singular Values with SURE

Code repository for the paper:  
**"Smooth Hard-Thresholding for Singular Values with Stein's Unbiased Risk Estimate"**

## Repository Structure

```
├── main.tex                    # Paper source (LaTeX)
├── figures/                    # All figures and tables referenced by the paper
├── set12/                      # Set12 standard grayscale test images (01–12.png)
├── SURE_SVD/                   # All experiment code
│   ├── utils/                  # Shared utilities
│   │   └── patch_utils.py     # Patch extraction, NCC search, collection builder
│   ├── exp1_unbiasedness/      # Experiment 1: Fixed-threshold SURE unbiasedness
│   │   └── experiment1_sure_unbiasedness.py
│   ├── exp2_postselection/     # Experiment 2: Post-selection bias
│   │   └── experiment2_postselection_bias.py
│   ├── exp3_oracle/            # Experiment 3: Oracle-style comparison
│   │   └── experiment3_oracle_comparison.py
│   ├── exp4_pipeline/          # Experiment 4: Complete denoising pipeline
│   │   ├── sure_svd_denoising.m           # Proposed SURE-SVD method
│   │   ├── energy_matching_svd_denoising.m # Energy matching baseline (same pipeline)
│   │   └── fig_pipeline.py                # Pipeline diagram generator
│   └── exp5_benchmark/         # Experiment 5: Full benchmark comparison
│       ├── run_all_remaining.m            # Run Guo/SURE/K-SVD/LPG-PCA (MATLAB)
│       ├── run_em_ssim.m                  # Run Energy matching with SSIM
│       ├── run_bm3d_em_ssim.py            # Run BM3D (Python official package)
│       ├── save_visual_house.m            # Generate visual comparison images
│       ├── generate_final_table.py        # Generate comparison table
│       ├── noisy_images/                  # Saved noisy images (.mat) for reproducibility
│       ├── results_all_matlab.mat         # MATLAB benchmark results
│       ├── results_em_full.mat            # Energy matching results
│       ├── bm3d_final.json               # BM3D results
│       └── table_final_comparison.txt     # Final text table
├── ksvdbox13/                  # K-SVD official toolbox (Ron Rubinstein)
├── ompbox10/                   # OMP official toolbox (Ron Rubinstein)
└── Program_lpgpca/             # LPG-PCA official code (Zhang et al.)
```

## Requirements

### Python (Experiments 1–3, BM3D)
- Python 3.10+
- NumPy, SciPy, Matplotlib, Pillow
- `bm3d` (official package): `pip install bm3d`

### MATLAB (Experiments 4–5)
- MATLAB R2023b+ (tested on R2026a)
- Image Processing Toolbox
- Wavelet Toolbox (optional, for some baselines)

## Running the Experiments

### Experiment 1: Fixed-Threshold SURE Unbiasedness
```bash
cd SURE_SVD/exp1_unbiasedness
python experiment1_sure_unbiasedness.py
```
Verifies Proposition 2.2: E[SURE] = E[MSE] for fixed deterministic (λ, ω).

### Experiment 2: Post-Selection Bias
```bash
cd SURE_SVD/exp2_postselection
python experiment2_postselection_bias.py
```
Demonstrates that SURE is optimistic after rank selection, but the selected rank is near-oracle.

### Experiment 3: Oracle-Style Comparison
```bash
cd SURE_SVD/exp3_oracle
python experiment3_oracle_comparison.py
```
Compares SURE vs energy matching vs oracle across noise levels.

### Experiment 4: Complete Denoising Pipeline
In MATLAB:
```matlab
cd SURE_SVD/exp4_pipeline
[denoised, psnr, ssim] = sure_svd_denoising(noisy, sigma, clean);
```

### Experiment 5: Full Benchmark
In MATLAB:
```matlab
cd SURE_SVD/exp5_benchmark
run_all_remaining   % Runs K-SVD, LPG-PCA, SURE on Set12
run_em_ssim         % Runs Energy matching
```
Then in Python:
```bash
cd SURE_SVD/exp5_benchmark
python run_bm3d_em_ssim.py   # BM3D on saved noisy images
python generate_final_table.py  # Generate comparison table
```

## Reproducibility

All methods are tested on the **same noisy images**, generated with fixed random seeds:
```matlab
rng(img_idx * 100 + sigma);
noisy = clean + sigma * randn(H, W);
```
The noisy images are saved in `exp5_benchmark/noisy_images/` as `.mat` files for exact reproducibility.

## Key Result

On Set12 (average over 12 images × 4 noise levels):

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| K-SVD | 29.65 | 0.8328 |
| LPG-PCA | 29.74 | 0.8169 |
| Energy matching | 29.99 | 0.8398 |
| **SURE (proposed)** | **30.07** | **0.8438** |
| BM3D | 30.32 | 0.8486 |

SURE outperforms energy matching by +0.08 dB on average, with the advantage increasing at higher noise levels (up to +0.94 dB at σ=50 on individual images).

## Citation

```bibtex
@article{yang2025sure,
  title={Smooth Hard-Thresholding for Singular Values with Stein's Unbiased Risk Estimate},
  author={Yang, Guanzhong},
  year={2025}
}
```
