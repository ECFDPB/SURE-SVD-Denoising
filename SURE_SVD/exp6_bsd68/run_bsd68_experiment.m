%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run_bsd68_experiment.m
% BSD68 benchmark: Energy matching vs SURE-modified SVD
% 68 images × 3 noise levels × 1 seed = 204 fully independent paired trials
%
% Seed = rng(img_idx * 100 + sigma) (unique per image-sigma pair)
% Data saved incrementally to CSV (crash-safe)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clc;

%% ── Configuration ────────────────────────────────────────────────────────
sigmas = [10, 30, 50];
img_dir = '../../BSD68/';

addpath('../exp4_pipeline');

% Get all image files
img_files = dir(fullfile(img_dir, '*.png'));
img_names = {img_files.name};
n_imgs = length(img_names);
n_sigmas = length(sigmas);
n_total = n_imgs * n_sigmas;

%% ── Preallocate ─────────────────────────────────────────────────────────
psnr_sure   = NaN(n_imgs, n_sigmas);
ssim_sure   = NaN(n_imgs, n_sigmas);
time_sure   = NaN(n_imgs, n_sigmas);
psnr_energy = NaN(n_imgs, n_sigmas);
ssim_energy = NaN(n_imgs, n_sigmas);
time_energy = NaN(n_imgs, n_sigmas);

%% ── Print experiment info ────────────────────────────────────────────────
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('BSD68 Benchmark: Energy matching vs SURE\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('Images: %d (BSD68, grayscale)\n', n_imgs);
fprintf('Noise levels: %s\n', mat2str(sigmas));
fprintf('Seed: rng(img_idx * 100 + sigma) (unique per image-sigma pair)\n');
fprintf('Total paired trials: %d\n', n_total);
fprintf('═══════════════════════════════════════════════════════════════\n\n');

%% ── Open CSV for incremental writing (append mode, skip header if file exists) ──
csv_path = 'results_bsd68.csv';
if ~exist(csv_path, 'file')
    fid = fopen(csv_path, 'w');
    fprintf(fid, 'image_index,image_name,sigma,psnr_sure,ssim_sure,time_sure,psnr_energy,ssim_energy,time_energy\n');
    fclose(fid);
    start_img = 1;
    start_sig = 1;
else
    % Count existing data rows to determine where to resume
    fid_check = fopen(csv_path, 'r');
    n_lines = 0;
    while ~feof(fid_check), fgetl(fid_check); n_lines = n_lines + 1; end
    fclose(fid_check);
    n_done = n_lines - 1;  % subtract header
    start_img = floor(n_done / n_sigmas) + 1;
    start_sig = mod(n_done, n_sigmas) + 1;
    fprintf('Resuming from image %d, sigma index %d (found %d completed trials in CSV)\n', start_img, start_sig, n_done);
end

trial_count = 0;
tic_total = tic;

%% ── Main loop ────────────────────────────────────────────────────────────
for img_idx = start_img:n_imgs
    clean = double(imread(fullfile(img_dir, img_names{img_idx})));
    if size(clean,3)==3, clean = double(rgb2gray(uint8(clean))); end
    [H, W] = size(clean);

    if mod(img_idx, 10) == 1 || img_idx == start_img
        fprintf('\n=== Image %d/%d: %s (%dx%d) ===\n', img_idx, n_imgs, img_names{img_idx}, H, W);
    end

    % Determine starting sigma index for this image
    if img_idx == start_img
        sig_start = start_sig;
    else
        sig_start = 1;
    end

    for sig_idx = sig_start:n_sigmas
        sigma = sigmas(sig_idx);
        trial_count = trial_count + 1;

        % Generate noisy image with unique seed per image-sigma pair
        rng(img_idx * 100 + sigma);
        noisy = clean + sigma * randn(H, W);

        % Energy matching
        pe = NaN; se = NaN; te = NaN;
        try
            tic;
            [~, pe, se] = energy_matching_svd_denoising(noisy, sigma, clean);
            te = toc;
            psnr_energy(img_idx, sig_idx) = pe;
            ssim_energy(img_idx, sig_idx) = se;
            time_energy(img_idx, sig_idx) = te;
        catch
            te = toc;
        end

        % SURE proposed
        ps = NaN; ss = NaN; ts = NaN;
        try
            tic;
            [~, ps, ss] = sure_svd_denoising(noisy, sigma, clean);
            ts = toc;
            psnr_sure(img_idx, sig_idx) = ps;
            ssim_sure(img_idx, sig_idx) = ss;
            time_sure(img_idx, sig_idx) = ts;
        catch
            ts = toc;
        end

        % Append to CSV immediately
        fid = fopen(csv_path, 'a');
        fprintf(fid, '%d,%s,%d,%.4f,%.6f,%.2f,%.4f,%.6f,%.2f\n', ...
            img_idx, img_names{img_idx}, sigma, ...
            ps, ss, ts, pe, se, te);
        fclose(fid);

        % Progress
        fprintf('  [%3d/%d] %s σ=%2d: EM=%.2f SURE=%.2f Δ=%+.2f (%.0fs+%.0fs)\n', ...
            trial_count, n_total, img_names{img_idx}, sigma, pe, ps, ps-pe, te, ts);
    end
end

elapsed_total = toc(tic_total);
fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf('BSD68 experiment complete. Total time: %.1f hours\n', elapsed_total/3600);
fprintf('═══════════════════════════════════════════════════════════════\n');

%% ── Save MAT file ────────────────────────────────────────────────────────
matlab_version = version;
platform_info = computer;
save('results_bsd68.mat', ...
     'psnr_sure', 'ssim_sure', 'time_sure', ...
     'psnr_energy', 'ssim_energy', 'time_energy', ...
     'img_names', 'sigmas', ...
     'matlab_version', 'platform_info');
fprintf('Saved: results_bsd68.mat\n');
fprintf('CSV saved incrementally: results_bsd68.csv\n');

%% ── Quick summary ────────────────────────────────────────────────────────
delta_all = psnr_sure(:) - psnr_energy(:);
fprintf('\nQuick summary:\n');
fprintf('  Mean ΔPSNR = %+.3f dB\n', mean(delta_all, 'omitnan'));
fprintf('  Win rate = %.1f%%\n', 100*mean(delta_all > 0, 'omitnan'));
for sig_idx = 1:n_sigmas
    d = psnr_sure(:, sig_idx) - psnr_energy(:, sig_idx);
    fprintf('  σ=%2d: mean Δ = %+.3f, win = %.1f%%\n', sigmas(sig_idx), mean(d,'omitnan'), 100*mean(d>0,'omitnan'));
end
