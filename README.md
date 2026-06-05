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
│   │   ├── run_set12_benchmark.m          # Run K-SVD/LPG-PCA/EM/SURE (MATLAB)
│   │   └── results_set12.json            # Raw results (12 images × 3 sigmas × 4 methods)
│   └── exp6_bsd68/             # Experiment 6: BSD68 statistical comparison
│       ├── run_bsd68_experiment.m         # Run Energy matching vs SURE on BSD68
│       └── results_bsd68.json            # Raw results (68 images × 3 sigmas)
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
run_set12_benchmark   % Runs K-SVD, LPG-PCA, Energy matching, SURE on Set12
```
Then in Python (runs BM3D and merges all results into results_set12.json):
```bash
python SURE_SVD/exp5_benchmark/run_bm3d.py
```

### Experiment 6: BSD68 Statistical Comparison
In MATLAB:
```matlab
cd SURE_SVD/exp6_bsd68
run_bsd68_experiment   % 68 images × 3 noise levels = 204 paired trials
```
Results saved to `results_bsd68.json`.

## Reproducibility

All methods are tested on the **same noisy images**, generated with deterministic seeds:
```matlab
rng(img_idx * 100 + sigma);
noisy = clean + sigma * randn(H, W);
```

Noise levels: σ ∈ {10, 30, 50}.

Pre-computed results are provided in `results_set12.json` (Set12, 5 methods) and `results_bsd68.json` (BSD68, Energy matching vs SURE).

## Citation

```bibtex
@article{yang2026sure,
  title={Smooth Hard-Thresholding for Singular Values with Stein's Unbiased Risk Estimate},
  author={Yang, Guanzhong},
  year={2026}
}
```
