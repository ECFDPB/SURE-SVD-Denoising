function B = im2colstep(A, blksize, stepsize)
%IM2COLSTEP Rearrange matrix blocks into columns (pure MATLAB fallback).
if nargin < 3
    stepsize = ones(1, length(blksize));
end
if length(blksize) == 2
    [M, N] = size(A);
    n1 = blksize(1); n2 = blksize(2);
    s1 = stepsize(1); s2 = stepsize(2);
    rows = 1:s1:(M-n1+1);
    cols = 1:s2:(N-n2+1);
    B = zeros(n1*n2, length(rows)*length(cols));
    k = 0;
    for j = cols
        for i = rows
            k = k + 1;
            patch = A(i:i+n1-1, j:j+n2-1);
            B(:, k) = patch(:);
        end
    end
elseif length(blksize) == 3
    [M, N, P] = size(A);
    n1 = blksize(1); n2 = blksize(2); n3 = blksize(3);
    s1 = stepsize(1); s2 = stepsize(2); s3 = stepsize(3);
    rows = 1:s1:(M-n1+1);
    cols = 1:s2:(N-n2+1);
    deps = 1:s3:(P-n3+1);
    B = zeros(n1*n2*n3, length(rows)*length(cols)*length(deps));
    k = 0;
    for d = deps
        for j = cols
            for i = rows
                k = k + 1;
                patch = A(i:i+n1-1, j:j+n2-1, d:d+n3-1);
                B(:, k) = patch(:);
            end
        end
    end
end
end
