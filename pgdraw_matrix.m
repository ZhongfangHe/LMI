% Draw a matrix of variables from PG(1,z) based on "pgdraw" that works for vectors

function x = pgdraw_matrix(z)
% Inputs:
%   z: a n-by-K matrix of scale parameters;
% Outputs:
%   x: a n-by-K matrix from PG(1,z);

[n,K] = size(z);
x = zeros(n,K);
for j = 1:K
    zj = z(:,j);
    x(:,j) = pgdraw(zj);
end
    