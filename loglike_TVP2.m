% Consider a TVP model:
% yt = xt' * bt + N(0,sig2t),
% bt = btm1 + N(0, diag(wt)), b0 ~ N(0,diag(w0))
%
% Compute loglike p(y|x,sig2,w,w0,b0) integrating out b

function loglike = loglike_TVP2(y, x, vary, w, b0)
% Inputs:
%   y: a n-by-1 vector of target (univariate),
%   x: a n-by-K matrix of regressors,
%   vary: a n-by-1 vector of measurement variance,
%   w: a n-by-K matrix of state variance,
%   b0: a K-by-1 vector of initial b,
% Outputs:
%   loglike: a scalar of the log likelihood.

[n,K] = size(x);
b0_mean = b0;
b0_cov = zeros(K,K); %p(b0) = N(b0_mean, b0_cov)

loglike = 0;
for t = 1:n
    % Collect items for t
    xt = x(t,:)';
    yt = y(t);
    sig2t = vary(t);
    wt = w(t,:)'; 
    
    % Compute moments for filtering distr p(bt|y^t)
    if t == 1
        mtm1 = b0_mean;
        Mtm1 = b0_cov;
    else
        mtm1 = mt;
        Mtm1 = Mt;
    end            
    [mt, Mt] = Kalman_iteration(mtm1, Mtm1, yt, xt, sig2t, wt);
    
    % Compute log likelihood
    if t == 1
        bt = b0;
        Bt = diag(wt);
    else
        bt = mtm1;
        Bt = diag(wt) + Mtm1;
    end
    tmp = sig2t + xt' * Bt * xt;
    loglike = loglike - 0.5*log(tmp) -0.5* ((yt - xt' * bt)^2) / tmp;
end




