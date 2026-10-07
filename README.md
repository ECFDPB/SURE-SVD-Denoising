# Smooth Hard-Thresholding for Singular Values with SURE

Code repository for the paper:  
**"Smooth Hard-Thresholding for Singular Values with Stein's Unbiased Risk Estimate"**

## Repository Structure

```
├── SURE_SVD/                   # All experiment code
│   ├── utils/                  # Shared utilities
│   │   └── patch_utils.py     # Patch extraction, NCC search, collection builder
│   ├── exp1_unbiasedness/      # Experiment 1: Fixed-threshold SURE unbiasedness
│   │   └── experiment1_sure_unbiasedness.py
│   ├── exp2_postselection/     # Experiment 2: Post-selection bias
│   │   └── experiment2_postselection_bias.py
│   ├── exp3_oracle/            # Experiment 3: Oracle-style comparison
│   │   └── experiment3_oracle_comparison.py
│   ├── exp4_pipeline/          # Shared denoising implementations
│   │   ├── sure_svd_denoising.m           # Hard-limit SURE-SVD pipeline
│   │   └── smooth_svd_denoising.m         # Finite-omega smooth reconstruction
│   ├── exp5_bsd68/             # Experiment 5: BSD68 statistical comparison
│   │   ├── run_bsd68_experiment.m         # Run Energy matching vs SURE on BSD68
│   │   └── results_bsd68.json            # Raw results (68 images × 3 sigmas)
│   ├── exp6_benchmark/         # Experiment 6: Set12 benchmark comparison
│   │   ├── run_set12_benchmark.m          # Run K-SVD/LPG-PCA/EM/SURE (MATLAB)
│   │   └── results_set12.json            # Raw results (12 images × 3 sigmas × 5 methods)
│   ├── exp7_smooth_denoising/  # Finite-omega denoising requested in review
│   │   └── run_smooth_denoising_experiment.m
│   └── exp8_noise_sensitivity/ # Sensitivity to noise-level misspecification
│       └── run_noise_level_sensitivity.m
├── ksvdbox13/                  # K-SVD official toolbox (Ron Rubinstein)
├── ompbox10/                   # OMP official toolbox (Ron Rubinstein)
└── Program_lpgpca/             # LPG-PCA official code (Zhang et al.)
```

## Requirements

### Python (Experiments 1–3, Statistical Analysis)
- Python 3.10+
- NumPy, SciPy, Matplotlib, Pillow, Pandas

### MATLAB (Experiments 4–8)
- MATLAB R2023b+ (tested on R2026a, Apple Silicon)
- Experiments 7–8 and their two shared denoisers require no additional toolbox.

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
[denoised, psnr_val, ssim_val] = sure_svd_denoising(noisy, sigma, clean);
```

### Experiment 5: BSD68 Statistical Comparison
In MATLAB:
```matlab
cd SURE_SVD/exp5_bsd68
run_bsd68_experiment   % 68 images × 3 noise levels = 204 paired trials
```
Results saved to `results_bsd68.json`.

### Experiment 6: Set12 Benchmark
In MATLAB:
```matlab
cd SURE_SVD/exp6_benchmark
run_set12_benchmark   % Runs K-SVD, LPG-PCA, Energy matching, SURE on Set12
```

### Experiment 7: Finite-Omega Smooth Denoising
In MATLAB:
```matlab
cd SURE_SVD/exp7_smooth_denoising
run_smooth_denoising_experiment
```
This experiment compares finite-omega smooth reconstruction with the hard-limit
output on identical noisy observations. It evaluates normalized values
`omega = {1, 5, 20, 100, 1000}` at noise levels `sigma = {10, 30, 50}` and
writes the raw results, summary, and figure to `results/`.

### Experiment 8: Noise-Level Sensitivity
In MATLAB:
```matlab
cd SURE_SVD/exp8_noise_sensitivity
run_noise_level_sensitivity
```
This experiment holds each noisy observation fixed while the denoiser receives
`sigma_est/sigma_true = {0.8, 0.9, 1.0, 1.1, 1.2}`. The default configuration
uses three independent noise realizations for every Set12 image and true noise
level and writes paired results, summaries, and figures to `results/`.

Experiments 7–8 expect local Set12 inputs under
`SURE_SVD/exp6_benchmark/noisy_images/`. Files are named
`01_sigma10.mat`, ..., `12_sigma50.mat` and contain `clean` and `noisy`
arrays. This generated-data directory is intentionally excluded from Git.

## Minimal Reviewer-Requested Addition

The reviewer-requested experiments are provided with only four MATLAB files:

1. `SURE_SVD/exp4_pipeline/sure_svd_denoising.m`
2. `SURE_SVD/exp4_pipeline/smooth_svd_denoising.m`
3. `SURE_SVD/exp7_smooth_denoising/run_smooth_denoising_experiment.m`
4. `SURE_SVD/exp8_noise_sensitivity/run_noise_level_sensitivity.m`

Shard runners, monitoring scripts, smoke tests, generated figures, and result
files are not part of this minimal addition.

## Reproducibility

All methods are tested on the **same noisy images**, generated with deterministic seeds:
```matlab
rng(img_idx * 100 + sigma, 'twister');
noisy = clean + sigma * randn(H, W);
```

Noise levels: σ ∈ {10, 30, 50}.

Pre-computed results are provided in `results_set12.json` (Set12, 4 methods) and `results_bsd68.json` (BSD68, Energy matching vs SURE).

## Citation

```bibtex
@article{yang2026sure,
  title={Smooth Hard-Thresholding for Singular Values with Stein's Unbiased Risk Estimate},
  author={Yang, Guanzhong},
  year={2026}
}
```
