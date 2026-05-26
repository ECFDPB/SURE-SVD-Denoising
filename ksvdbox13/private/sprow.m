function [x, id] = sprow(A, j)
%SPROW Extract row of sparse matrix (pure MATLAB fallback).
%  X = SPROW(A,J) returns the nonzero values in row A(J,:).
%  [X,ID] = SPROW(A,J) also returns the column indices.
id = find(A(j,:));
x = full(A(j, id));
end
