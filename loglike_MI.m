% Consider the mixture innovation model:
% scalar: yt = xt' * bt + N(0,sig2t), 
% k-by-1: bt = btm1 + N(0,diag(wt)), 
%
% Compute the logarithm of the likelihood p(y|w) integrating out b
% Only the part of p(y|w) relevant to wt is kept


function loglike = loglike_MI(yt, xt, mut, omegat, mtm1, Mtm1, sig2t, wt)
% Inputs:
%   yt: a scalar of the target variable,
%   xt: a K-by-1 vector of regressors,
%   mut: a K-by-1 vector of GK output,
%   omegat: a K-by-K matrix of GK output,
%   mtm1: a K-by-1 vector from Kalman filter at t-1,
%   Mtm1: a K-by-K matrix from Kalman filter at t-1,
%   sig2t: a scalar of measurement variance,
%   wt: a K-by-1 vector of state variance,
% Outputs:
%   loglike: a scalar of the log likelihood relevant for wt.

k = length(xt);

[mt, Mt] = Kalman_iteration(mtm1, Mtm1, yt, xt, sig2t, wt);
if any(Mt(:)) %Mt is not a zero matrix
%     Tt = chol(Mt)';
    if rcond(Mt) < 1e-10
        Tt = robust_chol(Mt);
    else
        Tt = chol(Mt)';
    end
else %Mt is a zero matrix
    Tt = zeros(k,k);
end
phit = Tt' * (mut - omegat * mt);
rt = xt' * mtm1;
Rt = sig2t + xt' * Mtm1 * xt;

tmp = eye(k) + Tt' * omegat * Tt;
Rxt = Rt + xt' * diag(wt) * xt; 
[tmp_half,flag] = chol(tmp);
if flag ~= 0
    tmp_half = robust_chol(tmp);
    logdet_tmp = 2 * sum(log(abs(diag(tmp_half))));

    if rcond(tmp_half) > 1e-10
        ttmp = tmp_half \ phit;
    else
        tmp_half_inv = robust_inv(tmp_half);
        ttmp = tmp_half_inv * phit;
    end
    loglike = -0.5*log(Rxt) - 0.5 * logdet_tmp ...
        -0.5 * mt' * omegat * mt + mut' * mt + 0.5 * (ttmp' * ttmp) ...
        -0.5 * (yt-rt) * (yt-rt) / Rxt;
else
    logdet_tmp = 2 * sum(log(abs(diag(tmp_half))));
%     loglike = -0.5*log(Rxt) - 0.5 * logdet_tmp ...
%         -0.5 * mt' * omegat * mt + mut' * mt + 0.5 * phit' * (tmp \ phit) ...
%         -0.5 * (yt-rt) * (yt-rt) / Rxt;
    if rcond(tmp_half) > 1e-10
        ttmp = tmp_half \ phit;
    else
        tmp_half_inv = robust_inv(tmp_half);
        ttmp = tmp_half_inv * phit;
    end
    loglike = -0.5*log(Rxt) - 0.5 * logdet_tmp ...
        -0.5 * mt' * omegat * mt + mut' * mt + 0.5 * (ttmp' * ttmp) ...
        -0.5 * (yt-rt) * (yt-rt) / Rxt;    
end


