% Run Energy Matching and record SSIM on saved noisy images
img_names = {'01.png','02.png','03.png','04.png','05.png','06.png',...
             '07.png','08.png','09.png','10.png','11.png','12.png'};
sigmas = [10, 20, 30, 50];
n_imgs = length(img_names); n_sigmas = length(sigmas);
psnr_em = NaN(n_imgs, n_sigmas);
ssim_em = NaN(n_imgs, n_sigmas);

for img_idx = 1:n_imgs
    fprintf('=== %s ===\n', img_names{img_idx});
    for sig_idx = 1:n_sigmas
        sigma = sigmas(sig_idx);
        data = load(fullfile('noisy_images', sprintf('%s_sigma%d.mat', img_names{img_idx}(1:end-4), sigma)));
        noisy = data.noisy;
        clean = data.clean;
        [~, p, s] = energy_matching_svd_denoising(noisy, sigma, clean);
        psnr_em(img_idx, sig_idx) = p;
        ssim_em(img_idx, sig_idx) = s;
        fprintf('  sigma=%d: PSNR=%.2f, SSIM=%.4f\n', sigma, p, s);
    end
end
save('results_em_full.mat', 'psnr_em', 'ssim_em', 'img_names', 'sigmas');
fprintf('\nSaved to results_em_full.mat\n');
