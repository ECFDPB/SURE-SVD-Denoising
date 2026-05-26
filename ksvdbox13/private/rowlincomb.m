function y = rowlincomb(A, x, cols)
%ROWLINCOMB Linear combination of matrix rows (pure MATLAB fallback).
%  Y = ROWLINCOMB(A,X,COLS) computes X'*A(COLS,:), i.e. a linear combination
%  of the rows of A specified by COLS with coefficients X.
if nargin < 3
    y = x' * A;
else
    y = x' * A(cols, :);
end
end
