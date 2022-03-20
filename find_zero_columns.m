% Given a matrix, find the index of zero columns

function [idx_zeros, idx_others] = find_zero_columns(x)
% Inputs:
%   x: a n-by-K matrix with possible zero columns,
% Outputs:
%   idx_zeros: a vector of the index of zero columns,
%   idx_others: a vector of the complement set of indx_zeros,

[n,K] = size(x);

idx_zeros = [];
for j = 1:K
    xj = x(:,j);
    nj = sum(xj == 0);
    if nj == n
        idx_zeros = [idx_zeros; j];
    end
end
idx_others = setdiff((1:K)',idx_zeros);
