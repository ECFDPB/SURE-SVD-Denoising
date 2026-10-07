%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run_noise_level_sensitivity.m
%
% End-to-end sensitivity of SURE-SVD denoising to noise-level estimation
% error.  Images are corrupted with a true sigma, while the denoiser is
% supplied with sigma_est = ratio * sigma_true.
%
% The default ratio grid [0.8,0.9,1,1.1,1.2] is deliberately chosen so the
% patch-size regime does not change at true sigma 10, 30 or 50.  Therefore
% the experiment primarily measures sensitivity of the SURE score and the
% second-stage noise update, rather than a discontinuous parameter-profile
% switch.  Set INCLUDE_SMOOTH=true to test the finite-omega estimator too.
%
% Results are appended after every run and the script is safe to resume.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clc;

this_dir = fileparts(mfilename('fullpath'));
pipeline_dir = fullfile(this_dir, '..', 'exp4_pipeline');
clean_source_dir = fullfile(this_dir, '..', 'exp6_benchmark', 'noisy_images');
output_dir = fullfile(this_dir, 'results');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
addpath(pipeline_dir);

all_image_names = arrayfun(@(i) sprintf('%02d.png', i), 1:12, ...
                           'UniformOutput', false);
image_indices_text = strtrim(getenv('SURE_SVD_IMAGE_INDICES'));
shard_id = strtrim(getenv('SURE_SVD_SHARD_ID'));
if isempty(image_indices_text)
    selected_image_indices = 1:numel(all_image_names);
else
    selected_image_indices = sscanf(image_indices_text, '%d').';
    if isempty(selected_image_indices) || ...
            any(~ismember(selected_image_indices, 1:numel(all_image_names)))
        error('Invalid SURE_SVD_IMAGE_INDICES: %s', image_indices_text);
    end
end
image_names = all_image_names(selected_image_indices);
true_sigmas = [10, 30, 50];
sigma_ratios = [0.8, 0.9, 1.0, 1.1, 1.2];
num_noise_seeds = 3;
base_seed = 20260927;

% The manuscript's proposed output is the hard-limit method.  Smooth
% sensitivity is optional because it approximately doubles the runtime.
include_smooth = false;
smooth_omega_normalized = 20;

if isempty(shard_id)
    run_output_dir = output_dir;
else
    run_output_dir = fullfile(output_dir, ['shard_', shard_id]);
    if ~exist(run_output_dir, 'dir'), mkdir(run_output_dir); end
end
results_csv = fullfile(run_output_dir, ...
                       'noise_level_sensitivity_results.csv');

fprintf('===============================================================\n');
fprintf('Noise-level misspecification sensitivity experiment\n');
fprintf('Images: %d, true sigmas: %s\n', numel(image_names), mat2str(true_sigmas));
fprintf('sigma_est/sigma_true: %s, noise seeds: %d\n', ...
        mat2str(sigma_ratios), num_noise_seeds);
if ~isempty(shard_id)
    fprintf('Shard %s, original image indices: %s\n', ...
            shard_id, mat2str(selected_image_indices));
end
fprintf('Methods: hard limit%s\n', ...
        conditional_text(include_smooth, ', finite-omega smooth', ''));
fprintf('Results: %s\n', results_csv);
fprintf('===============================================================\n\n');

for local_image_index = 1:numel(image_names)
    image_index = selected_image_indices(local_image_index);
    image_name = image_names{local_image_index};
    image_stem = erase(image_name, '.png');

    % Any recorded sigma file contains the same clean image.  sigma=10 is
    % used only as a convenient bundled source for that clean image.
    clean_file = fullfile(clean_source_dir, ...
                          sprintf('%s_sigma10.mat', image_stem));
    if ~exist(clean_file, 'file')
        error('Missing clean-image source: %s', clean_file);
    end
    data = load(clean_file);
    if ~isfield(data, 'clean')
        error('%s must contain variable clean.', clean_file);
    end
    clean = double(data.clean);
    [H, W] = size(clean);

    for sigma_true = true_sigmas
        for seed_index = 1:num_noise_seeds
            rng_seed = base_seed + 100000 * image_index + ...
                       100 * sigma_true + seed_index;
            rng(rng_seed, 'twister');
            noisy = clean + sigma_true * randn(H, W);

            fprintf('[%02d/%02d] %s, true sigma=%d, seed=%d\n', ...
                    local_image_index, numel(image_names), image_name, ...
                    sigma_true, seed_index);

            for sigma_ratio = sigma_ratios
                sigma_est = sigma_ratio * sigma_true;

                run_method(results_csv, image_index, image_name, ...
                           sigma_true, seed_index, rng_seed, sigma_ratio, ...
                           sigma_est, "hard_limit", NaN, noisy, clean);

                if include_smooth
                    run_method(results_csv, image_index, image_name, ...
                               sigma_true, seed_index, rng_seed, ...
                               sigma_ratio, sigma_est, "smooth", ...
                               smooth_omega_normalized, noisy, clean);
                end
            end
        end
    end
end

summarize_results(results_csv, run_output_dir, true_sigmas, sigma_ratios);
fprintf('\nExperiment complete.\n');


function text = conditional_text(condition, true_text, false_text)
    if condition, text = true_text; else, text = false_text; end
end


function run_method(csv_path, image_index, image_name, sigma_true, ...
                    seed_index, rng_seed, sigma_ratio, sigma_est, ...
                    method, omega_normalized, noisy, clean)
    if is_completed(csv_path, image_name, sigma_true, seed_index, ...
                    sigma_ratio, method, omega_normalized)
        fprintf('  %-10s ratio=%.2f: already complete\n', ...
                method, sigma_ratio);
        return;
    end

    timer = tic;
    if method == "hard_limit"
        [~, psnr_value, ssim_value] = ...
            sure_svd_denoising(noisy, sigma_est, clean);
    elseif method == "smooth"
        [~, psnr_value, ssim_value] = ...
            smooth_svd_denoising(noisy, sigma_est, clean, ...
                                 omega_normalized);
    else
        error('Unknown method: %s', method);
    end
    elapsed = toc(timer);

    append_result(csv_path, image_index, image_name, sigma_true, ...
                  seed_index, rng_seed, sigma_ratio, sigma_est, method, ...
                  omega_normalized, psnr_value, ssim_value, elapsed);
    fprintf('  %-10s ratio=%.2f: %.3f dB / %.5f (%.1f s)\n', ...
            method, sigma_ratio, psnr_value, ssim_value, elapsed);
end


function tf = is_completed(csv_path, image_name, sigma_true, seed_index, ...
                           sigma_ratio, method, omega_normalized)
    tf = false;
    if ~exist(csv_path, 'file'), return; end
    previous = readtable(csv_path, 'TextType', 'string');
    mask = previous.image_name == string(image_name) & ...
           previous.sigma_true == sigma_true & ...
           previous.seed_index == seed_index & ...
           abs(previous.sigma_ratio - sigma_ratio) < 1e-12 & ...
           previous.method == string(method);
    if isnan(omega_normalized)
        mask = mask & isnan(previous.omega_normalized);
    else
        mask = mask & ...
            abs(previous.omega_normalized - omega_normalized) < 1e-12;
    end
    tf = any(mask);
end


function append_result(csv_path, image_index, image_name, sigma_true, ...
                       seed_index, rng_seed, sigma_ratio, sigma_est, ...
                       method, omega_normalized, psnr_value, ssim_value, elapsed)
    row = table(image_index, string(image_name), sigma_true, seed_index, ...
                rng_seed, sigma_ratio, sigma_est, string(method), ...
                omega_normalized, ...
                psnr_value, ssim_value, elapsed, ...
        'VariableNames', {'image_index','image_name','sigma_true', ...
                          'seed_index','rng_seed','sigma_ratio','sigma_est', ...
                          'method','omega_normalized','psnr','ssim', ...
                          'time_seconds'});
    if exist(csv_path, 'file')
        writetable(row, csv_path, 'WriteMode', 'append', ...
                   'WriteVariableNames', false);
    else
        writetable(row, csv_path);
    end
end


function summarize_results(csv_path, output_dir, sigmas, ratios)
    results = readtable(csv_path, 'TextType', 'string');
    methods = unique(results.method, 'stable');
    summary = table();

    for method_index = 1:numel(methods)
        method = methods(method_index);
        for sigma_true = sigmas
            baseline = results(results.method == method & ...
                               results.sigma_true == sigma_true & ...
                               abs(results.sigma_ratio - 1) < 1e-12, :);

            for sigma_ratio = ratios
                subset = results(results.method == method & ...
                                 results.sigma_true == sigma_true & ...
                                 abs(results.sigma_ratio - sigma_ratio) < 1e-12, :);
                if isempty(subset), continue; end

                delta_psnr = NaN(height(subset), 1);
                delta_ssim = NaN(height(subset), 1);
                for row_index = 1:height(subset)
                    match = baseline.image_index == subset.image_index(row_index) & ...
                            baseline.seed_index == subset.seed_index(row_index);
                    if any(match)
                        first = find(match, 1, 'first');
                        delta_psnr(row_index) = subset.psnr(row_index) - ...
                                                baseline.psnr(first);
                        delta_ssim(row_index) = subset.ssim(row_index) - ...
                                                baseline.ssim(first);
                    end
                end

                row = table(method, sigma_true, sigma_ratio, ...
                            100 * (sigma_ratio - 1), height(subset), ...
                            mean(subset.psnr, 'omitnan'), ...
                            mean(subset.ssim, 'omitnan'), ...
                            mean(delta_psnr, 'omitnan'), ...
                            std(delta_psnr, 0, 'omitnan'), ...
                            mean(delta_ssim, 'omitnan'), ...
                            std(delta_ssim, 0, 'omitnan'), ...
                    'VariableNames', {'method','sigma_true','sigma_ratio', ...
                                      'relative_error_percent','N', ...
                                      'mean_psnr','mean_ssim', ...
                                      'mean_delta_psnr','std_delta_psnr', ...
                                      'mean_delta_ssim','std_delta_ssim'});
                if width(summary) == 0
                    summary = row;
                else
                    summary = [summary; row]; %#ok<AGROW>
                end
            end
        end
    end

    writetable(summary, fullfile(output_dir, ...
               'noise_level_sensitivity_summary.csv'));

    colors = lines(max(1, numel(methods)));
    figure_handle = figure('Color', 'white', 'Position', [100 100 1200 620]);
    layout = tiledlayout(2, numel(sigmas), 'TileSpacing', 'compact', ...
                         'Padding', 'compact');
    title(layout, 'Sensitivity to noise-standard-deviation misspecification');

    for sigma_index = 1:numel(sigmas)
        sigma_true = sigmas(sigma_index);

        nexttile(sigma_index); hold on;
        for method_index = 1:numel(methods)
            subset = summary(summary.sigma_true == sigma_true & ...
                             summary.method == methods(method_index), :);
            standard_error = subset.std_delta_psnr ./ sqrt(subset.N);
            errorbar(subset.relative_error_percent, subset.mean_delta_psnr, ...
                     standard_error, '-o', 'LineWidth', 1.4, ...
                     'Color', colors(method_index,:), ...
                     'DisplayName', display_method(methods(method_index)));
        end
        xline(0, ':'); yline(0, '--'); grid on;
        xlabel('Noise-level error (%)'); ylabel('PSNR change from correct \sigma (dB)');
        title(sprintf('True \\sigma=%d', sigma_true));
        if sigma_index == 1, legend('Location', 'best'); end

        nexttile(numel(sigmas) + sigma_index); hold on;
        for method_index = 1:numel(methods)
            subset = summary(summary.sigma_true == sigma_true & ...
                             summary.method == methods(method_index), :);
            standard_error = subset.std_delta_ssim ./ sqrt(subset.N);
            errorbar(subset.relative_error_percent, subset.mean_delta_ssim, ...
                     standard_error, '-o', 'LineWidth', 1.4, ...
                     'Color', colors(method_index,:), ...
                     'DisplayName', display_method(methods(method_index)));
        end
        xline(0, ':'); yline(0, '--'); grid on;
        xlabel('Noise-level error (%)'); ylabel('SSIM change from correct \sigma');
        title(sprintf('True \\sigma=%d', sigma_true));
    end

    exportgraphics(figure_handle, fullfile(output_dir, ...
                   'noise_level_sensitivity.png'), 'Resolution', 200);
    savefig(figure_handle, fullfile(output_dir, ...
            'noise_level_sensitivity.fig'));
    close(figure_handle);
end


function label = display_method(method)
    if method == "hard_limit"
        label = 'Hard-limit SURE';
    elseif method == "smooth"
        label = 'Finite-omega smooth SURE';
    else
        label = char(method);
    end
end
