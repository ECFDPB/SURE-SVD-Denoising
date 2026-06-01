%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run_set12_benchmark.m
% Set12 benchmark: K-SVD, LPG-PCA, Energy matching, SURE
% 12 images × 3 noise levels = 36 paired trials
%
% Seed = rng(img_idx * 100 + sigma)
% All methods use the SAME noisy image per (image, sigma) pair.
% BM3D must be run separately in Python (run_bm3d.py) due to Apple Silicon.
% Noisy images are saved as .mat for Python to load.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clc;

%% ── Configuration ────────────────────────────────────────────────────────
img_dir = '../../set12/';
img_names = {'01.png','02.png','03.png','04.png','05.png','06.png',...
             '07.png','08.png','09.png','10.png','11.png','12.png'};
sigmas = [10, 30, 50];

addpath('../exp4_pipeline');
addpath('../../ksvdbox13');
addpath('../../ompbox10');
addpath('../../Program_lpgpca/Code');

n_imgs = length(img_names);
n_sigmas = length(sigmas);

%% ── Preallocate ─────────────────────────────────────────────────────────
psnr_sure   = NaN(n_imgs, n_sigmas);  ssim_sure   = NaN(n_imgs, n_sigmas);
psnr_em     = NaN(n_imgs, n_sigmas);  ssim_em     = NaN(n_imgs, n_sigmas);
psnr_ksvd   = NaN(n_imgs, n_sigmas);  ssim_ksvd   = NaN(n_imgs, n_sigmas);
psnr_lpg    = NaN(n_imgs, n_sigmas);  ssim_lpg    = NaN(n_imgs, n_sigmas);

%% ── Create directory for noisy images (for Python BM3D) ─────────────────
noisy_dir = 'noisy_images';
if ~exist(noisy_dir, 'dir'), mkdir(noisy_dir); end

%% ── Print experiment info ────────────────────────────────────────────────
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('Set12 Benchmark: K-SVD, LPG-PCA, Energy matching, SURE\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('Images: %d (Set12, grayscale)\n', n_imgs);
fprintf('Noise levels: %s\n', mat2str(sigmas));
fprintf('Seed: rng(img_idx * 100 + sigma)\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

%% ── Main loop ────────────────────────────────────────────────────────────
for img_idx = 1:n_imgs
    clean = double(imread(fullfile(img_dir, img_names{img_idx})));
    if size(clean,3)==3, clean = double(rgb2gray(uint8(clean))); end
    [H, W] = size(clean);
    fprintf('\n=== Image %d/%d: %s (%dx%d) ===\n', img_idx, n_imgs, img_names{img_idx}, H, W);

    for sig_idx = 1:n_sigmas
        sigma = sigmas(sig_idx);
        rng(img_idx * 100 + sigma);
        noisy = clean + sigma * randn(H, W);
        fprintf('  sigma=%2d: ', sigma);

        % Save noisy image as .mat for Python BM3D
        noisy_filename = fullfile(noisy_dir, sprintf('%s_sigma%d.mat', img_names{img_idx}(1:end-4), sigma));
        save(noisy_filename, 'noisy', 'clean', 'sigma');

        % ── Energy matching ──
        try
            [~, p, s] = energy_matching_svd_denoising(noisy, sigma, clean);
            psnr_em(img_idx, sig_idx) = p;
            ssim_em(img_idx, sig_idx) = s;
            fprintf('EM=%.2f ', p);
        catch
            fprintf('EM:ERR ');
        end

        % ── SURE proposed ──
        try
            [~, p, s] = sure_svd_denoising(noisy, sigma, clean);
            psnr_sure(img_idx, sig_idx) = p;
            ssim_sure(img_idx, sig_idx) = s;
            fprintf('SURE=%.2f ', p);
        catch
            fprintf('SURE:ERR ');
        end

        % ── K-SVD ──
        try
            params.x = noisy;
            params.blocksize = 8;
            params.dictsize = 256;
            params.sigma = sigma;
            params.maxval = 255;
            params.trainnum = 40000;
            params.iternum = 10;
            params.memusage = 'high';
            [est, ~] = ksvddenoise(params);
            psnr_ksvd(img_idx, sig_idx) = compute_psnr(clean, est);
            ssim_ksvd(img_idx, sig_idx) = compute_ssim(clean, est);
            fprintf('KSVD=%.2f ', psnr_ksvd(img_idx, sig_idx));
        catch
            fprintf('KSVD:ERR ');
        end

        % ── LPG-PCA ──
        try
            [~, ~, ~, p2, s2] = LPGPCA_denoising(noisy, clean, sigma, 'fast', 0);
            psnr_lpg(img_idx, sig_idx) = p2;
            ssim_lpg(img_idx, sig_idx) = s2;
            fprintf('LPG=%.2f ', p2);
        catch
            fprintf('LPG:ERR ');
        end

        fprintf('\n');
    end
end

%% ── Save results ─────────────────────────────────────────────────────────
save('results_set12_matlab.mat', ...
     'psnr_sure', 'ssim_sure', ...
     'psnr_em', 'ssim_em', ...
     'psnr_ksvd', 'ssim_ksvd', ...
     'psnr_lpg', 'ssim_lpg', ...
     'img_names', 'sigmas');
fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf('MATLAB results saved to results_set12_matlab.mat\n');
fprintf('Noisy images saved to %s/ for Python BM3D\n', noisy_dir);
fprintf('Next: run "python run_bm3d.py" to complete BM3D results.\n');
fprintf('═══════════════════════════════════════════════════════════════\n');

%% ── Quick summary ────────────────────────────────────────────────────────
fprintf('\nQuick summary (average PSNR over all images and noise levels):\n');
fprintf('  K-SVD:            %.2f dB\n', mean(psnr_ksvd(:), 'omitnan'));
fprintf('  LPG-PCA:          %.2f dB\n', mean(psnr_lpg(:), 'omitnan'));
fprintf('  Energy matching:  %.2f dB\n', mean(psnr_em(:), 'omitnan'));
fprintf('  SURE (proposed):  %.2f dB\n', mean(psnr_sure(:), 'omitnan'));

%% ── Helper functions ─────────────────────────────────────────────────────
function p = compute_psnr(clean, denoised)
    mse = mean((clean(:)-denoised(:)).^2);
    if mse < 1e-10, p = 100; else, p = 10*log10(255^2/mse); end
end

function s = compute_ssim(img1, img2)
    C1 = (0.01*255)^2; C2 = (0.03*255)^2;
    img1 = double(img1); img2 = double(img2);
    win = fspecial('gaussian', 11, 1.5);
    mu1 = conv2(img1, win, 'same'); mu2 = conv2(img2, win, 'same');
    s1 = conv2(img1.^2, win, 'same') - mu1.^2;
    s2 = conv2(img2.^2, win, 'same') - mu2.^2;
    s12 = conv2(img1.*img2, win, 'same') - mu1.*mu2;
    ssim_map = ((2*mu1.*mu2+C1).*(2*s12+C2))./((mu1.^2+mu2.^2+C1).*(s1+s2+C2));
    s = mean(ssim_map(:));
end
