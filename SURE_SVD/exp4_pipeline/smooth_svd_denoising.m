function [denoised, psnr_val, ssim_val, diagnostics] = ...
    smooth_svd_denoising(noisy, sigma_est, clean, omega_normalized, options)
%SMOOTH_SVD_DENOISING Patch-based SVD denoising with finite-omega shrinkage.
%
%   DENOISED = SMOOTH_SVD_DENOISING(NOISY, SIGMA_EST, CLEAN, OMEGA_NORMALIZED)
%   uses the same two-pass patch grouping, aggregation and back-projection
%   structure as sure_svd_denoising.m.  The difference is that each grouped
%   patch matrix is reconstructed with the actual smooth shrinker
%
%       f_{omega,lambda}(s) = s / (1 + exp(-omega*(s-lambda)))
%
%   rather than with the limiting hard rank truncation.
%
%   OMEGA_NORMALIZED is the smoothing parameter on the [0,1] intensity
%   scale used in the manuscript's matrix experiments.  This image pipeline
%   operates on [0,255], so the actual slope used by the shrinker is
%
%       omega = OMEGA_NORMALIZED / 255.
%
%   This is an exact change-of-units rule: scaling singular values and lambda
%   by 255 while dividing omega by 255 leaves omega*(s-lambda) unchanged.
%   It is not an optimal-parameter formula.
%
%   Lambda is selected by minimizing the finite-omega smooth SURE formula
%   over one representative threshold for every possible retained rank.
%   Candidate h=1,...,k-1 uses the midpoint between adjacent singular
%   values, h=k uses sigma_k/2, and h=0 extrapolates a midpoint above
%   sigma_1.  Thus the candidate set is fixed when omega varies, as required
%   by the pointwise convergence comparison in the manuscript.
%
%   Optional fields in OPTIONS:
%       patch_size       [] (chosen from sigma_est), or a positive integer
%       similar_patches  85
%       delta             0.5
%       gamma             0.65
%       search_half_size 35
%       reference_step    3
%       min_second_tau    1
%       intensity_scale  255
%
%   CLEAN may be [] when PSNR and SSIM are not required.  DIAGNOSTICS
%   reports the average selected candidate rank and threshold in each pass.

    if nargin < 3 || isempty(clean), clean = []; end
    if nargin < 4 || isempty(omega_normalized), omega_normalized = 20; end
    if nargin < 5 || isempty(options), options = struct(); end

    validateattributes(noisy, {'numeric'}, {'real','2d','nonempty','finite'});
    validateattributes(sigma_est, {'numeric'}, {'real','scalar','positive','finite'});
    validateattributes(omega_normalized, {'numeric'}, ...
                       {'real','scalar','positive','finite'});
    if ~isempty(clean)
        validateattributes(clean, {'numeric'}, {'real','2d','size',size(noisy),'finite'});
    end

    opts = default_options(options, sigma_est);
    noisy = double(noisy);
    if ~isempty(clean), clean = double(clean); end

    [x0, stage1] = one_pass_smooth(noisy, sigma_est, omega_normalized, opts);

    % Back projection and noise update, matching the hard-limit pipeline.
    y_tilde = x0 + opts.delta * (noisy - x0);
    [H, W] = size(noisy);
    residual_mse = sum((y_tilde(:) - x0(:)).^2) / (H * W);
    tau_new = opts.gamma * sqrt(max(sigma_est^2 - residual_mse, ...
                                    opts.min_second_tau^2));

    [denoised, stage2] = one_pass_smooth(y_tilde, tau_new, ...
                                        omega_normalized, opts);
    denoised = max(0, min(255, denoised));

    if isempty(clean)
        psnr_val = NaN;
        ssim_val = NaN;
    else
        psnr_val = compute_psnr(clean, denoised);
        ssim_val = compute_ssim(clean, denoised);
    end

    diagnostics = struct( ...
        'sigma_est', sigma_est, ...
        'omega_normalized', omega_normalized, ...
        'omega_code', omega_normalized / opts.intensity_scale, ...
        'patch_size', opts.patch_size, ...
        'second_stage_tau', tau_new, ...
        'stage1', stage1, ...
        'stage2', stage2);
end


function opts = default_options(user, sigma_est)
    opts = struct( ...
        'patch_size', [], ...
        'similar_patches', 85, ...
        'delta', 0.5, ...
        'gamma', 0.65, ...
        'search_half_size', 35, ...
        'reference_step', 3, ...
        'min_second_tau', 1, ...
        'intensity_scale', 255);

    names = fieldnames(user);
    valid = fieldnames(opts);
    for i = 1:numel(names)
        if ~ismember(names{i}, valid)
            error('smooth_svd_denoising:UnknownOption', ...
                  'Unknown option "%s".', names{i});
        end
        opts.(names{i}) = user.(names{i});
    end

    if isempty(opts.patch_size)
        if sigma_est < 20
            opts.patch_size = 9;
        elseif sigma_est < 40
            opts.patch_size = 10;
        else
            opts.patch_size = 11;
        end
    end

    validateattributes(opts.patch_size, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(opts.similar_patches, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(opts.delta, {'numeric'}, {'scalar','real','nonnegative','finite'});
    validateattributes(opts.gamma, {'numeric'}, {'scalar','real','nonnegative','finite'});
    validateattributes(opts.search_half_size, {'numeric'}, {'scalar','integer','nonnegative'});
    validateattributes(opts.reference_step, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(opts.min_second_tau, {'numeric'}, {'scalar','real','positive','finite'});
    validateattributes(opts.intensity_scale, {'numeric'}, {'scalar','real','positive','finite'});
end


function [est, stats] = one_pass_smooth(img, tau, omega_normalized, opts)
    [H, W] = size(img);
    ps = opts.patch_size;
    if H < ps || W < ps
        error('smooth_svd_denoising:ImageTooSmall', ...
              'Image dimensions must be at least the patch size %d.', ps);
    end

    m = ps * ps;
    N_row = H - ps + 1;
    N_col = W - ps + 1;

    all_patches = zeros(m, N_row * N_col);
    idx = 0;
    for col = 1:N_col
        for row = 1:N_row
            idx = idx + 1;
            all_patches(:, idx) = reshape( ...
                img(row:row+ps-1, col:col+ps-1), [], 1);
        end
    end

    rows_ref = unique([1:opts.reference_step:N_row, N_row]);
    cols_ref = unique([1:opts.reference_step:N_col, N_col]);
    est_acc = zeros(H, W);
    weight_acc = zeros(H, W);

    group_count = 0;
    rank_sum = 0;
    lambda_sum = 0;
    omega_sum = 0;
    score_sum = 0;

    for ci = 1:numel(cols_ref)
        col = cols_ref(ci);
        for ri = 1:numel(rows_ref)
            row = rows_ref(ri);
            ref_idx = (col - 1) * N_row + row;
            ref = all_patches(:, ref_idx);

            rmin = max(row - opts.search_half_size, 1);
            rmax = min(row + opts.search_half_size, N_row);
            cmin = max(col - opts.search_half_size, 1);
            cmax = min(col + opts.search_half_size, N_col);
            [rr, cc] = ndgrid(rmin:rmax, cmin:cmax);
            candidates = (cc(:) - 1) * N_row + rr(:);

            differences = all_patches(:, candidates) - ref;
            distances = sum(differences.^2, 1);
            [~, order] = sort(distances, 'ascend');
            group_size = min(opts.similar_patches + 1, numel(candidates));
            indices = candidates(order(1:group_size));

            group = all_patches(:, indices);
            [mg, ng] = size(group);
            [U, S, V] = svd(group, 'econ');
            singular_values = diag(S);

            [shrunk_values, nominal_rank, lambda, omega, selected_score] = ...
                select_smooth_sure(singular_values, mg, ng, tau, ...
                                   omega_normalized, opts.intensity_scale);
            denoised_group = U * diag(shrunk_values) * V';

            % Keep the original aggregation rule controlled by the selected
            % candidate rank.  This isolates smooth-versus-hard shrinkage.
            if nominal_rank < ng
                group_weight = 1 - nominal_rank / ng;
            else
                group_weight = 1 / ng;
            end
            group_weight = max(group_weight, 1e-6);

            for p = 1:numel(indices)
                patch_index = indices(p);
                patch_row = mod(patch_index - 1, N_row) + 1;
                patch_col = floor((patch_index - 1) / N_row) + 1;
                patch = reshape(denoised_group(:, p), [ps, ps]);
                row_range = patch_row:patch_row+ps-1;
                col_range = patch_col:patch_col+ps-1;
                est_acc(row_range, col_range) = ...
                    est_acc(row_range, col_range) + group_weight * patch;
                weight_acc(row_range, col_range) = ...
                    weight_acc(row_range, col_range) + group_weight;
            end

            group_count = group_count + 1;
            rank_sum = rank_sum + nominal_rank;
            lambda_sum = lambda_sum + lambda;
            omega_sum = omega_sum + omega;
            score_sum = score_sum + selected_score;
        end
    end

    if any(weight_acc(:) == 0)
        error('smooth_svd_denoising:UncoveredPixels', ...
              'Patch aggregation left at least one pixel uncovered.');
    end
    est = est_acc ./ weight_acc;

    stats = struct( ...
        'group_count', group_count, ...
        'mean_candidate_rank', rank_sum / group_count, ...
        'mean_lambda', lambda_sum / group_count, ...
        'mean_omega', omega_sum / group_count, ...
        'mean_selected_sure', score_sum / group_count);
end


function [f_selected, h_selected, lambda_selected, omega, score_selected] = ...
    select_smooth_sure(sv, m, n, tau, omega_normalized, intensity_scale)

    k = numel(sv);
    omega = omega_normalized / intensity_scale;

    lambdas = zeros(k + 1, 1);
    if k >= 2 && sv(1) > sv(2)
        % Mirror the first inter-singular-value half-gap above sigma_1.
        lambdas(1) = sv(1) + (sv(1) - sv(2)) / 2;
    elseif sv(1) > 0
        lambdas(1) = 1.5 * sv(1);
    else
        lambdas(1) = eps(intensity_scale);
    end
    for h = 1:k-1
        lambdas(h + 1) = (sv(h) + sv(h + 1)) / 2;
    end
    lambdas(k + 1) = sv(k) / 2;

    [scores, shrunk] = smooth_sure_scores( ...
        sv, m, n, tau, omega, lambdas);

    % MATLAB min returns the first minimizer, hence ties favor lower rank.
    [score_selected, index] = min(scores);
    h_selected = index - 1;
    lambda_selected = lambdas(index);
    f_selected = shrunk(:, index);
end


function [scores, f] = smooth_sure_scores(sv, m, n, tau, omega, lambdas)
% Evaluate all threshold candidates simultaneously.  The pairwise spectral
% divergence is linear in a_i = sigma_i*f_i away from repeated singular
% values, so its coefficients need to be constructed only once per group.
    argument = max(-50, min(50, omega * (sv - lambdas.')));
    gate = 1 ./ (1 + exp(-argument));
    f = sv .* gate;
    fp = gate + omega .* sv .* gate .* (1 - gate);

    residual = sum((sv - f).^2, 1);
    divergence = abs(m - n) * sum(gate, 1) + sum(fp, 1);

    % Pair the (i,j) and (j,i) terms before summation.  For non-tied values,
    %   2*(sigma_i*f_i - sigma_j*f_j)/(sigma_i^2-sigma_j^2)
    % is accumulated through one coefficient per singular component.  The
    % near-tie branch uses the removable limit of the paired expression.
    k = numel(sv);
    pair_coefficients = zeros(k, 1);
    near_i = zeros(0, 1);
    near_j = zeros(0, 1);
    sv_squared = sv.^2;
    for i = 1:k-1
        for j = i+1:k
            denominator = sv_squared(i) - sv_squared(j);
            scale = max([sv_squared(i), sv_squared(j), 1]);
            if abs(denominator) > 1e-10 * scale
                coefficient = 2 / denominator;
                pair_coefficients(i) = pair_coefficients(i) + coefficient;
                pair_coefficients(j) = pair_coefficients(j) - coefficient;
            else
                near_i(end + 1, 1) = i; %#ok<AGROW>
                near_j(end + 1, 1) = j; %#ok<AGROW>
            end
        end
    end

    pair_term = pair_coefficients.' * (sv .* f);
    for pair_index = 1:numel(near_i)
        i = near_i(pair_index);
        j = near_j(pair_index);
        pair_term = pair_term + ...
            0.5 * (gate(i,:) + fp(i,:) + gate(j,:) + fp(j,:));
    end
    divergence = divergence + pair_term;
    scores = (-m * n * tau^2 + residual + 2 * tau^2 * divergence).';
end


function p = compute_psnr(clean, denoised)
    mse = mean((clean(:) - denoised(:)).^2);
    if mse < 1e-10
        p = 100;
    else
        p = 10 * log10(255^2 / mse);
    end
end


function s = compute_ssim(img1, img2)
    C1 = (0.01 * 255)^2;
    C2 = (0.03 * 255)^2;
    window = gaussian_window(11, 1.5);
    mu1 = conv2(img1, window, 'same');
    mu2 = conv2(img2, window, 'same');
    variance1 = conv2(img1.^2, window, 'same') - mu1.^2;
    variance2 = conv2(img2.^2, window, 'same') - mu2.^2;
    covariance = conv2(img1 .* img2, window, 'same') - mu1 .* mu2;
    ssim_map = ((2 * mu1 .* mu2 + C1) .* (2 * covariance + C2)) ./ ...
               ((mu1.^2 + mu2.^2 + C1) .* ...
                (variance1 + variance2 + C2));
    s = mean(ssim_map(:));
end


function window = gaussian_window(window_size, standard_deviation)
    radius = (window_size - 1) / 2;
    coordinates = -radius:radius;
    [x_grid, y_grid] = meshgrid(coordinates, coordinates);
    window = exp(-(x_grid.^2 + y_grid.^2) / ...
                 (2 * standard_deviation^2));
    window = window / sum(window(:));
end
