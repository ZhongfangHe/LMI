% Consider the model:
% scalar: yt = xt' * bt + N(0,sig2t), 
% k-by-1: bt = btm1 + N(0,diag(wt)),
%
% p(y^{t+1,n}|y^t,bt,w) \propto exp(-0.5*bt'*omegat*bt + bt'*mut)
% back recursion to compute omegat and mut

function [omega, mu] = MI_backward_recursion(y, x, sig2, w)
% Inputs:
%   y: a n-by-1 vector of target,
%   x: a n-by-k matrix of regressors,
%   sig2: a n-by-1 vector of target innovation variance,
%   w: a n-by-k matrix of the state variances,
% Outputs:
%   omega: a n-by-1 cell of k-by-k matrices,
%   mu: a n-by-k matrix stacking mut.

[n,k] = size(x);
omega = cell(n,1);
omega{n} = zeros(k,k);
mu = zeros(n,k);
t = n;
while t > 1
    yt = y(t);
    xt = x(t,:)';
    sig2t = sig2(t);
    wt = diag(w(t,:));
    
    omegat = omega{t};
    mut = mu(t,:)';
    
    wxt = wt * xt;
    rt = sig2t + xt' * wxt;
    Bt = wxt / rt;
    At = eye(k) - Bt * xt';
%     if ~any(diag(wt)) %zeros
%         Ct = zeros(k,k);
%     else
%         CCt = wt - (wxt * wxt') / rt;
%     %     Ct = chol(CCt)';
%         [uu,ss,~] = svd(CCt);
%         Ct = uu * diag(sqrt(diag(ss)));
%     end
    CCt = wt - (wxt * wxt') / rt;
%     Ct = chol(CCt)';
    [uu,ss,~] = svd(CCt);
    Ct = uu * diag(sqrt(diag(ss)));    
    Dt = eye(k) + Ct' * omegat * Ct;
    if rcond(Dt) > 1e-10
        tmp = omegat * Ct * (Dt \ Ct');
    else
        Dt_inv = robust_inv(Dt);
        tmp = omegat * (Ct * Dt_inv * Ct');
    end
    omegatm1 = At' * (omegat - tmp * omegat) * At + (xt * xt') / rt;
    mutm1 = At' * (eye(k) - tmp) * (mut - omegat * Bt * yt) + (xt * yt) / rt;
    
    omega{t-1} = omegatm1;
    mu(t-1,:) = mutm1';
    
    t = t - 1;
end
