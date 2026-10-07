%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run_smooth_denoising_experiment.m
%
% Image-level comparison of the finite-omega smooth estimator with the
% limiting hard-truncation estimator used in the current manuscript.
%
% Inputs are the recorded Set12 noisy images from exp6_benchmark so every
% method receives exactly the same noisy realization.  Results are appended
% to CSV after every run and the script can be resumed safely.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clc;

this_dir = fileparts(mfilename('fullpath'));
pipeline_dir = fullfile(this_dir, '..', 'exp4_pipeline');
input_dir = fullfile(this_dir, '..', 'exp6_benchmark', 'noisy_images');
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

% Omega values are reported on the normalized [0,1] scale used in the
% manuscript's matrix experiments.  Because this pipeline works on
% [0,255], smooth_svd_denoising uses omega_code = omega_normalized/255.
% The value Inf is represented separately by the existing hard-limit code.
omega_grid = [1, 5, 20, 100, 1000];

if isempty(shard_id)
    run_output_dir = output_dir;
else
    run_output_dir = fullfile(output_dir, ['shard_', shard_id]);
    if ~exist(run_output_dir, 'dir'), mkdir(run_output_dir); end
end
results_csv = fullfile(run_output_dir, 'smooth_denoising_results.csv');

fprintf('===============================================================\n');
fprintf('Finite-omega smooth SVD denoising experiment\n');
fprintf('Images: %d, sigmas: %s, normalized omega: %s plus Inf\n', ...
        numel(image_names), mat2str(true_sigmas), mat2str(omega_grid));
if ~isempty(shard_id)
    fprintf('Shard %s, original image indices: %s\n', ...
            shard_id, mat2str(selected_image_indices));
end
fprintf('Results: %s\n', results_csv);
fprintf('===============================================================\n\n');

for local_image_index = 1:numel(image_names)
    image_index = selected_image_indices(local_image_index);
    image_name = image_names{local_image_index};
    image_stem = erase(image_name, '.png');

    for sigma_index = 1:numel(true_sigmas)
        sigma_true = true_sigmas(sigma_index);
        input_file = fullfile(input_dir, ...
            sprintf('%s_sigma%d.mat', image_stem, sigma_true));
        if ~exist(input_file, 'file')
            error('Missing recorded noisy image: %s', input_file);
        end

        data = load(input_file);
        if ~isfield(data, 'clean') || ~isfield(data, 'noisy')
            error('%s must contain variables clean and noisy.', input_file);
        end
        clean = double(data.clean);
        noisy = double(data.noisy);

        fprintf('[%02d/%02d] %s, sigma=%d\n', local_image_index, ...
                numel(image_names), image_name, sigma_true);

        if ~is_completed(results_csv, image_name, sigma_true, ...
                         "hard_limit", NaN)
            timer = tic;
            [~, psnr_value, ssim_value] = ...
                sure_svd_denoising(noisy, sigma_true, clean);
            elapsed = toc(timer);
            append_result(results_csv, image_index, image_name, sigma_true, ...
                          "hard_limit", NaN, psnr_value, ssim_value, ...
                          elapsed, NaN, NaN, NaN);
            fprintf('  hard limit:       %.3f dB / %.5f (%.1f s)\n', ...
                    psnr_value, ssim_value, elapsed);
        else
            fprintf('  hard limit:       already complete\n');
        end

        for omega_normalized = omega_grid
            if is_completed(results_csv, image_name, sigma_true, ...
                            "smooth", omega_normalized)
                fprintf('  smooth omega=%-4g: already complete\n', ...
                        omega_normalized);
                continue;
            end

            timer = tic;
            [~, psnr_value, ssim_value, diagnostics] = ...
                smooth_svd_denoising(noisy, sigma_true, clean, ...
                                     omega_normalized);
            elapsed = toc(timer);
            append_result(results_csv, image_index, image_name, sigma_true, ...
                          "smooth", omega_normalized, psnr_value, ssim_value, ...
                          elapsed, diagnostics.stage1.mean_candidate_rank, ...
                          diagnostics.stage2.mean_candidate_rank, ...
                          diagnostics.second_stage_tau);
            fprintf('  smooth omega=%-4g: %.3f dB / %.5f (%.1f s)\n', ...
                    omega_normalized, psnr_value, ssim_value, elapsed);
        end
    end
end

summarize_results(results_csv, run_output_dir, true_sigmas, omega_grid);
fprintf('\nExperiment complete.\n');


function tf = is_completed(csv_path, image_name, sigma_true, method, ...
                           omega_normalized)
    tf = false;
    if ~exist(csv_path, 'file'), return; end
    previous = readtable(csv_path, 'TextType', 'string');
    mask = previous.image_name == string(image_name) & ...
           previous.sigma_true == sigma_true & ...
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
                       method, omega_normalized, psnr_value, ssim_value, ...
                       elapsed_seconds, mean_rank_stage1, ...
                       mean_rank_stage2, second_stage_tau)
    row = table(image_index, string(image_name), sigma_true, string(method), ...
                omega_normalized, psnr_value, ssim_value, elapsed_seconds, ...
                mean_rank_stage1, mean_rank_stage2, second_stage_tau, ...
        'VariableNames', {'image_index','image_name','sigma_true','method', ...
                          'omega_normalized','psnr','ssim','time_seconds', ...
                          'mean_rank_stage1','mean_rank_stage2', ...
                          'second_stage_tau'});
    if exist(csv_path, 'file')
        writetable(row, csv_path, 'WriteMode', 'append', ...
                   'WriteVariableNames', false);
    else
        writetable(row, csv_path);
    end
end


function summarize_results(csv_path, output_dir, sigmas, omega_grid)
    results = readtable(csv_path, 'TextType', 'string');
    summary_rows = table();

    for sigma_true = sigmas
        hard = results(results.sigma_true == sigma_true & ...
                       results.method == "hard_limit", :);
        if ~isempty(hard)
            row = table(sigma_true, "hard_limit", NaN, height(hard), ...
                        mean(hard.psnr, 'omitnan'), ...
                        mean(hard.ssim, 'omitnan'), 0, 0, ...
                'VariableNames', {'sigma_true','method','omega_normalized', ...
                                  'N','mean_psnr','mean_ssim', ...
                                  'mean_delta_psnr_vs_hard', ...
                                  'mean_delta_ssim_vs_hard'});
            if width(summary_rows) == 0
                summary_rows = row;
            else
                summary_rows = [summary_rows; row]; %#ok<AGROW>
            end
        end

        for omega_normalized = omega_grid
            subset = results(results.sigma_true == sigma_true & ...
                             results.method == "smooth" & ...
                             abs(results.omega_normalized - ...
                                 omega_normalized) < 1e-12, :);
            if isempty(subset), continue; end
            [paired_smooth, paired_hard] = paired_rows(subset, hard);
            delta_psnr = mean(paired_smooth.psnr - paired_hard.psnr, ...
                              'omitnan');
            delta_ssim = mean(paired_smooth.ssim - paired_hard.ssim, ...
                              'omitnan');
            row = table(sigma_true, "smooth", omega_normalized, ...
                        height(subset), ...
                        mean(subset.psnr, 'omitnan'), ...
                        mean(subset.ssim, 'omitnan'), ...
                        delta_psnr, delta_ssim, ...
                'VariableNames', {'sigma_true','method','omega_normalized','N', ...
                                  'mean_psnr','mean_ssim', ...
                                  'mean_delta_psnr_vs_hard', ...
                                  'mean_delta_ssim_vs_hard'});
            if width(summary_rows) == 0
                summary_rows = row;
            else
                summary_rows = [summary_rows; row]; %#ok<AGROW>
            end
        end
    end

    writetable(summary_rows, fullfile(output_dir, ...
               'smooth_denoising_summary.csv'));

    figure_handle = figure('Color', 'white', 'Position', [100 100 1200 600]);
    layout = tiledlayout(2, numel(sigmas), 'TileSpacing', 'compact', ...
                         'Padding', 'compact');
    title(layout, 'Finite-omega smooth shrinkage versus hard-limit output');

    for sigma_index = 1:numel(sigmas)
        sigma_true = sigmas(sigma_index);
        smooth = summary_rows(summary_rows.sigma_true == sigma_true & ...
                              summary_rows.method == "smooth", :);
        hard = summary_rows(summary_rows.sigma_true == sigma_true & ...
                            summary_rows.method == "hard_limit", :);

        nexttile(sigma_index);
        semilogx(smooth.omega_normalized, smooth.mean_psnr, '-o', ...
                 'LineWidth', 1.5, 'DisplayName', 'Smooth'); hold on;
        if ~isempty(hard)
            yline(hard.mean_psnr(1), '--', 'Hard limit', 'LineWidth', 1.2);
        end
        grid on; xlabel('Normalized \omega'); ylabel('Mean PSNR (dB)');
        title(sprintf('\\sigma=%d', sigma_true));

        nexttile(numel(sigmas) + sigma_index);
        semilogx(smooth.omega_normalized, smooth.mean_ssim, '-o', ...
                 'LineWidth', 1.5, 'DisplayName', 'Smooth'); hold on;
        if ~isempty(hard)
            yline(hard.mean_ssim(1), '--', 'Hard limit', 'LineWidth', 1.2);
        end
        grid on; xlabel('Normalized \omega'); ylabel('Mean SSIM');
        title(sprintf('\\sigma=%d', sigma_true));
    end

    exportgraphics(figure_handle, fullfile(output_dir, ...
                   'smooth_denoising_summary.png'), 'Resolution', 200);
    savefig(figure_handle, fullfile(output_dir, ...
            'smooth_denoising_summary.fig'));
    close(figure_handle);
end


function [smooth_paired, hard_paired] = paired_rows(smooth_rows, hard_rows)
    [common_names, smooth_indices, hard_indices] = intersect( ...
        smooth_rows.image_name, hard_rows.image_name, 'stable');
    if isempty(common_names)
        smooth_paired = smooth_rows([],:);
        hard_paired = hard_rows([],:);
    else
        smooth_paired = smooth_rows(smooth_indices,:);
        hard_paired = hard_rows(hard_indices,:);
    end
end
