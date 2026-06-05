% Save denoised images for House (02.png) at sigma=50 for visual comparison
addpath('../ksvdbox13'); addpath('../ompbox10'); addpath('../Program_lpgpca/Code');

data = load('noisy_images/02_sigma50.mat');
noisy = data.noisy; clean = data.clean; sigma = 50;

% K-SVD
params.x = noisy; params.blocksize = 8; params.dictsize = 256;
params.sigma = sigma; params.maxval = 255; params.trainnum = 40000;
params.iternum = 10; params.memusage = 'high';
[est_ksvd, ~] = ksvddenoise(params);
imwrite(uint8(max(0,min(255,est_ksvd))), 'visual_house_ksvd.png');
fprintf('K-SVD: %.2f dB\n', 10*log10(255^2/mean((clean(:)-est_ksvd(:)).^2)));

% LPG-PCA
[~,~,~,p_lpg,~] = LPGPCA_denoising(noisy, clean, sigma, 'fast', 0);
% LPG-PCA returns cropped image, need to get full size denoised
% Actually re-run and capture output
[d_lpg,~,~,~,~] = LPGPCA_denoising(noisy, clean, sigma, 'fast', 0);
% d_lpg is cropped; save noisy-sized version by padding
% Actually LPGPCA_denoising with K=0 returns full size
imwrite(uint8(max(0,min(255,d_lpg))), 'visual_house_lpgpca.png');
fprintf('LPG-PCA done\n');

% Energy matching
[est_em, p_em, ~] = energy_matching_svd_denoising(noisy, sigma, clean);
imwrite(uint8(max(0,min(255,est_em))), 'visual_house_em.png');
fprintf('EM: %.2f dB\n', p_em);

% SURE
[est_sure, p_sure, ~] = sure_svd_denoising(noisy, sigma, clean);
imwrite(uint8(max(0,min(255,est_sure))), 'visual_house_sure.png');
fprintf('SURE: %.2f dB\n', p_sure);

% Also save clean and noisy
imwrite(uint8(clean), 'visual_house_clean.png');
imwrite(uint8(max(0,min(255,noisy))), 'visual_house_noisy.png');

fprintf('All saved.\n');
