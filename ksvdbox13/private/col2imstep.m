function A = col2imstep(B, imsize, blksize, stepsize)
%COL2IMSTEP Rearrange columns into matrix blocks (pure MATLAB fallback).
if nargin < 4
    stepsize = ones(1, length(blksize));
end
if length(blksize) == 2
    n1 = blksize(1); n2 = blksize(2);
    s1 = stepsize(1); s2 = stepsize(2);
    M = imsize(1); N = imsize(2);
    A = zeros(M, N);
    rows = 1:s1:(M-n1+1);
    cols = 1:s2:(N-n2+1);
    k = 0;
    for j = cols
        for i = rows
            k = k + 1;
            A(i:i+n1-1, j:j+n2-1) = A(i:i+n1-1, j:j+n2-1) + reshape(B(:,k), [n1 n2]);
        end
    end
elseif length(blksize) == 3
    n1 = blksize(1); n2 = blksize(2); n3 = blksize(3);
    s1 = stepsize(1); s2 = stepsize(2); s3 = stepsize(3);
    M = imsize(1); N = imsize(2); P = imsize(3);
    A = zeros(M, N, P);
    rows = 1:s1:(M-n1+1);
    cols = 1:s2:(N-n2+1);
    deps = 1:s3:(P-n3+1);
    k = 0;
    for d = deps
        for j = cols
            for i = rows
                k = k + 1;
                A(i:i+n1-1, j:j+n2-1, d:d+n3-1) = A(i:i+n1-1, j:j+n2-1, d:d+n3-1) + reshape(B(:,k), [n1 n2 n3]);
            end
        end
    end
end
end
