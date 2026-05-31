# Smooth Hard-Thresholding for Singular Values with SURE

Code repository for the paper:  
**"Smooth Hard-Thresholding for Singular Values with Stein's Unbiased Risk Estimate"**

## Repository Structure

```
├── set12/                      # Set12 standard grayscale test images (01–12.png)
├── BSD68/                      # BSD68 grayscale test images (0000–0067.png)
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
│   │   └── energy_matching_svd_denoising.m # Energy matching baseline (same pipeline)
│   ├── exp5_benchmark/         # Experiment 5: Set12 benchmark comparison
│   │   ├── run_all_remaining.m            # Run Guo/SURE/K-SVD/LPG-PCA (MATLAB)
│   │   ├── run_em_ssim.m                  # Run Energy matching with SSIM
│   │   ├── run_bm3d_em_ssim.py            # Run BM3D (Python official package)
│   │   ├── noisy_images/                  # Saved noisy images (.mat) for reproducibility
│   │   ├── results_all_matlab.mat         # MATLAB benchmark results
│   │   ├── results_em_full.mat            # Energy matching results
│   │   └── bm3d_final.json               # BM3D results
│   └── exp6_bsd68/             # Experiment 6: BSD68 statistical comparison
│       ├── run_bsd68_experiment.m         # Run Energy matching vs SURE on BSD68
│       └── analyze_bsd68.py               # Wilcoxon signed-rank test analysis
├── ksvdbox13/                  # K-SVD official toolbox (Ron Rubinstein)
├── ompbox10/                   # OMP official toolbox (Ron Rubinstein)
└── Program_lpgpca/             # LPG-PCA official code (Zhang et al.)
```

## Requirements

### Python (Experiments 1–3, BM3D, Statistical Analysis)
- Python 3.10+
- NumPy, SciPy, Matplotlib, Pillow, Pandas
- `bm3d` (official package): `pip install bm3d`

### MATLAB (Experiments 4–6)
- MATLAB R2023b+ (tested on R2026a, Apple Silicon)
- Image Processing Toolbox

## Running the Experiments

### Experiment 1: Fixed-Threshold SURE Unbiasedness
```bash
python SURE_SVD/exp1_unbiasedness/experiment1_sure_unbiasedness.py
```
Verifies Proposition 2.2: E[SURE] = E[MSE] for fixed deterministic (λ, ω).

### Experiment 2: Post-Selection Bias
```bash
python SURE_SVD/exp2_postselection/experiment2_postselection_bias.py
```
Demonstrates that SURE is optimistic after rank selection, but the selected rank is near-oracle.

### Experiment 3: Oracle-Style Comparison
```bash
python SURE_SVD/exp3_oracle/experiment3_oracle_comparison.py
```
Compares SURE vs energy matching vs oracle across noise levels.

### Experiment 4: Complete Denoising Pipeline
In MATLAB:
```matlab
addpath('SURE_SVD/exp4_pipeline');
[denoised, psnr, ssim] = sure_svd_denoising(noisy, sigma, clean);
```

### Experiment 5: Set12 Benchmark
In MATLAB:
```matlab
cd SURE_SVD/exp5_benchmark
run_all_remaining   % Runs Guo/SURE/K-SVD/LPG-PCA on Set12
run_em_ssim         % Runs Energy matching
```
Then in Python:
```bash
python SURE_SVD/exp5_benchmark/run_bm3d_em_ssim.py
```

### Experiment 6: BSD68 Statistical Comparison
In MATLAB:
```matlab
cd SURE_SVD/exp6_bsd68
run_bsd68_experiment   % 68 images × 3 noise levels = 204 paired trials
```
Then in Python:
```bash
python SURE_SVD/exp6_bsd68/analyze_bsd68.py   # Wilcoxon signed-rank test
```

## Reproducibility

All methods are tested on the **same noisy images**, generated with deterministic seeds:
```matlab
rng(img_idx * 100 + sigma);
noisy = clean + sigma * randn(H, W);
```

Noise levels: σ ∈ {10, 30, 50}.

The noisy images for Set12 are saved in `exp5_benchmark/noisy_images/` as `.mat` files for exact reproducibility.

## Citation

```bibtex
@article{yang2026sure,
  title={Smooth Hard-Thresholding for Singular Values with Stein's Unbiased Risk Estimate},
  author={Yang, Guanzhong},
  year={2026}
}
```
