function y = collincomb(A, x, rows)
%COLLINCOMB Linear combination of matrix columns (pure MATLAB fallback).
%  Y = COLLINCOMB(A,X,ROWS) computes A(:,ROWS)*X, i.e. a linear combination
%  of the columns of A specified by ROWS with coefficients X.
if nargin < 3
    y = A * x;
else
    y = A(:, rows) * x;
end
end
