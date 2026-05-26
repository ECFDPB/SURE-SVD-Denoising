function [denoised, psnr_val, ssim_val] = sure_svd_denoising(noisy, sigma, clean)
% SURE_SVD_DENOISING  Patch-based SVD denoising with SURE rank selection.
%
%   Follows Guo et al. (IEEE TCSVT 2016) framework exactly, replacing only
%   the energy-matching rank selection (Eq. 22) with SURE limiting score.
%
%   Parameters matched to Guo et al.:
%     Patch size: 9x9 (sigma<20), 10x10 (20<=sigma<40), 11x11 (sigma>=40)
%     L = 85 similar patches, delta = 0.5, gamma = 0.65, search window = 35

    if nargin < 3, clean = []; end

    if sigma < 20, ps = 9; elseif sigma < 40, ps = 10; else, ps = 11; end
    L = 85; delta = 0.5; gamma_bp = 0.65; sw = 35; step = 3;

    % Stage 1
    x0 = one_pass(noisy, sigma, ps, L, sw, step, 'sure');

    % Back projection (Eq. 29)
    y_tilde = x0 + delta * (noisy - x0);

    % Noise update (Eq. 30): tau_new = gamma * sqrt(sigma^2 - ||y_tilde - x0||^2 / (H*W))
    % Note: y_tilde - x0 = delta*(y - x0), so ||y_tilde-x0||^2/(H*W) = delta^2 * MSE(y,x0)
    [H, W] = size(noisy);
    ytilde_x0_mse = sum((y_tilde(:) - x0(:)).^2) / (H*W);
    tau_new = gamma_bp * sqrt(max(sigma^2 - ytilde_x0_mse, 1));  % floor at 1 to avoid 0

    % Stage 2
    denoised = one_pass(y_tilde, tau_new, ps, L, sw, step, 'sure');
    denoised = max(0, min(255, denoised));

    if ~isempty(clean)
        psnr_val = compute_psnr(clean, denoised);
        ssim_val = compute_ssim(clean, denoised);
    else
        psnr_val = NaN; ssim_val = NaN;
    end
end


function est = one_pass(img, tau, ps, L, sw, step, method)
    [H, W] = size(img);
    m = ps*ps;
    N_row = H-ps+1; N_col = W-ps+1;

    % Extract all patches (column-major indexing)
    all_patches = zeros(m, N_row*N_col);
    idx = 0;
    for j = 1:N_col
        for i = 1:N_row
            idx = idx + 1;
            all_patches(:, idx) = reshape(img(i:i+ps-1, j:j+ps-1), [], 1);
        end
    end

    % Reference positions
    rows_ref = 1:step:N_row;
    if rows_ref(end) < N_row, rows_ref = [rows_ref, N_row]; end
    cols_ref = 1:step:N_col;
    if cols_ref(end) < N_col, cols_ref = [cols_ref, N_col]; end

    est_acc = zeros(H, W);
    weight_acc = zeros(H, W);

    for ci = 1:length(cols_ref)
        col = cols_ref(ci);
        for ri = 1:length(rows_ref)
            row = rows_ref(ri);
            ref_idx = (col-1)*N_row + row;
            ref = all_patches(:, ref_idx);

            % Find L+1 similar patches in search window
            rmin = max(row-sw,1); rmax = min(row+sw,N_row);
            cmin = max(col-sw,1); cmax = min(col+sw,N_col);
            cands = [];
            for c = cmin:cmax
                for r = rmin:rmax
                    cands = [cands; (c-1)*N_row + r];
                end
            end
            diffs = all_patches(:, cands) - ref;
            dists = sum(diffs.^2, 1)';
            [~, ord] = sort(dists);
            ns = min(L+1, length(cands));
            indc = cands(ord(1:ns));

            group = all_patches(:, indc);
            [mg, ng] = size(group);

            % SVD
            [U, S_mat, V] = svd(group, 'econ');
            sv = diag(S_mat);

            % Rank selection
            if strcmp(method, 'sure')
                h = rank_sure(sv, mg, ng, tau);
            else
                h = rank_energy(sv, mg, ng, tau);
            end

            % Truncation
            if h == 0
                denoised_group = zeros(size(group));
            else
                sv_trunc = [sv(1:h); zeros(length(sv)-h, 1)];
                denoised_group = U * diag(sv_trunc) * V';
            end

            % Weight (Guo Eq. 24)
            if h < ng, w_j = 1 - h/ng; else, w_j = 1/ng; end
            w_j = max(w_j, 1e-6);

            % Aggregate ALL patches in group back to image
            for p = 1:length(indc)
                pidx = indc(p);
                pr = mod(pidx-1, N_row) + 1;
                pc = floor((pidx-1) / N_row) + 1;
                patch_2d = reshape(denoised_group(:, p), [ps, ps]);
                est_acc(pr:pr+ps-1, pc:pc+ps-1) = ...
                    est_acc(pr:pr+ps-1, pc:pc+ps-1) + w_j * patch_2d;
                weight_acc(pr:pr+ps-1, pc:pc+ps-1) = ...
                    weight_acc(pr:pr+ps-1, pc:pc+ps-1) + w_j;
            end
        end
    end

    est = est_acc ./ max(weight_acc, 1e-10);
end


function h = rank_sure(sv, m, n, tau)
% SURE limiting score minimisation (Section 3 of our paper)
    k = length(sv); tau2 = tau^2; sv2 = sv.^2;
    scores = zeros(k+1, 1);
    scores(1) = sum(sv2) - m*n*tau2;  % S_0: zero estimator

    ratio_row = zeros(k, 1);
    for i = 1:k
        for j = 1:k
            if j ~= i
                denom = sv2(i) - sv2(j);
                if abs(denom) > 1e-10
                    ratio_row(i) = ratio_row(i) + sv2(i)/denom;
                end
            end
        end
    end
    cum_ratio = cumsum(ratio_row);

    for h = 1:k
        residual = sum(sv2(h+1:end));
        dof = abs(m-n)*h + h + 2*cum_ratio(h);
        scores(h+1) = -m*n*tau2 + residual + 2*tau2*dof;
    end
    [~, idx] = min(scores);
    h = idx - 1;
end


function h = rank_energy(sv, m, n, tau)
% Energy matching (Guo Eq. 22)
    sv2 = sv.^2;
    noise_e = m * n * tau^2;
    target = sum(sv2) - noise_e;
    if target <= 0, h = 0; return; end
    cumE = cumsum(sv2);
    h = find(cumE >= target, 1, 'first');
    if isempty(h), h = length(sv); end
end


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
