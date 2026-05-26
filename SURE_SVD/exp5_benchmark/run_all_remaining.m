%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run Guo official, SURE proposed, K-SVD, and LPG-PCA on Set12
% Also saves noisy images for BM3D (Python) to use the same noise
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clc;

img_dir = '../set12/';
img_names = {'01.png','02.png','03.png','04.png','05.png','06.png',...
             '07.png','08.png','09.png','10.png','11.png','12.png'};
sigmas = [10, 20, 30, 50];

addpath('../lra_svd/LRA_SVD');
addpath('../ksvdbox13');
addpath('../ompbox10');
addpath('../Program_lpgpca/Code');

n_imgs = length(img_names);
n_sigmas = length(sigmas);

psnr_guo = NaN(n_imgs, n_sigmas);
ssim_guo = NaN(n_imgs, n_sigmas);
psnr_sure = NaN(n_imgs, n_sigmas);
ssim_sure = NaN(n_imgs, n_sigmas);
psnr_ksvd = NaN(n_imgs, n_sigmas);
ssim_ksvd = NaN(n_imgs, n_sigmas);
psnr_lpg = NaN(n_imgs, n_sigmas);
ssim_lpg = NaN(n_imgs, n_sigmas);

% Create directory for noisy images
noisy_dir = 'noisy_images';
if ~exist(noisy_dir, 'dir'), mkdir(noisy_dir); end

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

        % Guo official
        try
            est = lra_svd_denoising(noisy, sigma);
            psnr_guo(img_idx, sig_idx) = compute_psnr(clean, est);
            ssim_guo(img_idx, sig_idx) = compute_ssim(clean, est);
            fprintf('Guo=%.2f ', psnr_guo(img_idx, sig_idx));
        catch e
            fprintf('Guo:ERR ');
        end

        % SURE proposed
        try
            [est, p, s] = sure_svd_denoising(noisy, sigma, clean);
            psnr_sure(img_idx, sig_idx) = p;
            ssim_sure(img_idx, sig_idx) = s;
            fprintf('SURE=%.2f ', p);
        catch e
            fprintf('SURE:ERR ');
        end

        % K-SVD (run from ksvdbox13 directory for private/ access)
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
        catch e
            fprintf('KSVD:ERR ');
        end

        % LPG-PCA
        try
            [~, ~, ~, p2, s2] = LPGPCA_denoising(noisy, clean, sigma, 'fast', 0);
            psnr_lpg(img_idx, sig_idx) = p2;
            ssim_lpg(img_idx, sig_idx) = s2;
            fprintf('LPG=%.2f ', p2);
        catch e
            fprintf('LPG:ERR ');
        end

        fprintf('\n');
    end
end

save('results_all_matlab.mat', ...
     'psnr_guo', 'ssim_guo', ...
     'psnr_sure', 'ssim_sure', ...
     'psnr_ksvd', 'ssim_ksvd', ...
     'psnr_lpg', 'ssim_lpg', ...
     'img_names', 'sigmas');
fprintf('\nAll MATLAB results saved to results_all_matlab.mat\n');
fprintf('Noisy images saved to %s/ for Python BM3D\n', noisy_dir);
