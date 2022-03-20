% Compute the RMSE of TVP estimates

function rmser = compute_rmse_tvp_beta(bdraws, btrue)
% Inputs:
%   bdraws: a K-by-1 cell of ndraws-by-n matrix of the posterior draws of b,
%   btrue: a n-by-K matrix of the true value of b,
% Outputs:
%   rmser: a n-by-K matrix of the RMSE for b.


[n,K] = size(btrue);
ndraws = size(bdraws{1},1);

rmser = zeros(n,K);
for j = 1:K
    er = bdraws{j} - repmat(btrue(:,j)',ndraws,1);
    ser = er.^2;
    mser = mean(ser)';
    rmser(:,j) = sqrt(mser);
end


