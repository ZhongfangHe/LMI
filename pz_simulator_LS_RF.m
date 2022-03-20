% Estimate a RW-TVP model:
% yt = xt' * bt + N(0,sig2t), 
% b_jt = b_{j,t-1} + vj * N(0,d_jt), j = 1, ..., K
% d_jt = f(z_jt),
% z_jt = z_{j,t-1} + N(0,sj),
%
% use GCK algorithm for zt (adpative MH， multivariate)
% draw b0, v, s by integrating out bt (adaptive MH, multivariate)
% z0 is integrated out and is not sampled
% use ASIS for extra boosting
%
% rectifier link: d_jt = max(0, z_jt-cj)
% for identification, sj = 1, cj = 0



function [pz, ind2, logrw, count, pz_mean, pz_cov] = pz_simulator_LS_RF(y, x, vary,...
    r0, v2, pz_old, ind2_old, b0_mean, b0_cov, pstar, AMH_c, ...
    logrw_old, logrw_start, drawi_start, drawi, burnin, pz_mean_old, pz_cov_old)
% Inputs:
%   y: a n-by-1 vector of target,
%   x: a n-by-k matrix of regressors,
%   vary: a n-by-1 vector of target innovation variance,
%   v2: a k-by-1 vector of the invariant part of state variances,
%   pz_old: a n-by-k matrix stacking zt from previous sweep,
%   ind_old: a n-by-k matrix stacking ind_t from previous sweep,
%   b0_mean: a k-by-1 vector of prior mean of b0 (b0 if conditioned on b0,
%   b0_cov: a k-by-k matrix of prior covariance matrix of b0 (zero if conditioned on b0),
%   r0: a k-by-1 vector of the variance of z0,
%   a: a k-by-1 vector of the working parameter for PX-DA,
%   pstar: a scalar of optimal MH acceptance rate (0.25),
%   AMH_c: a scalar of the slope for MH update,
%   logrw_old: a n-by-1 vector of RW std from previous sweep,
%   logrw_start: a n-by-1 vector of the immediate start/re-start value of logrw,
%   drawi_start: a n-by-1 vector of the immediate start/re-start time of logrw,
%   drawi: a scaler of the # of the current sweep,
%   burnin: a scalar of the # of burn-ins (to count the MH acceptance),
%   z_mean_old: a n-by-K matrix stacking the sample mean of zt based on draws up to drawi-1,
%   z_cov_old: a n-by-1 cell of the sample cov matrix of zt based on draws up to drawi-1,
% Outputs:
%   pz: a n-by-k matrix stacking zt from current sweep,
%   ind: a n-by-k matrix stacking ind_t from current sweep,
%   logrw: a n-by-1 vector of log RW std from current sweep,
%   count: a n-by-1 vector of MH acceptance indicator.
%   z_mean: a n-by-K matrix stacking the sample mean of zt based on draws up to drawi,
%   z_cov: a n-by-1 cell of the sample cov matrix of zt based on draws up to drawi (for adaptive MH)

[n,K] = size(x);
zmax = 1e10; %bound zt by +/-zmax to avoid numerical issue


%% Setup for adaptive MH 
% pstar = 0.35;
% AMH_c = 1/(K * pstar * (1 - pstar)) + (1-1/K)*sqrt(pi)*exp(0.5*AMH_a*AMH_a)/(sqrt(2)*AMH_a);
% AMH_slope = AMH_c / drawi;
% AMH_intercept = -AMH_slope * pstar;



%% Backward recursion for omega, mu
w_old = ind2_old;
[omega, mu] = MI_backward_recursion(y, x, vary, w_old);


%% Forward simulation of pz by adaptive MH
ind2 = ind2_old;
pz = pz_old;
logrw = logrw_old;
count = zeros(n,1);
pz_mean = pz_mean_old;
pz_cov = pz_cov_old;
for t = 1:n
    % Collect items for t
    xt = x(t,:)';
    yt = y(t);
    sig2t = vary(t);    
    mut = mu(t,:)';
    omegat = omega{t};
    if t == 1
        mtm1 = b0_mean;
        Mtm1 = b0_cov;
    else
        mtm1 = mt;
        Mtm1 = Mt;
    end
    
    
    % Propose zt from a RW
    pzt_old = pz(t,:)';
    if drawi < 100
        A = eye(K);
    else  
        A = pz_cov_old{t} + 1e-6 * eye(K) / drawi;
    end
    if rcond(A) < 1e-15
        A_half = robust_chol(A);
        eta = A_half * randn(K,1);
    else
        eta = mvnrnd(zeros(K,1),A)';
    end 
    pzt_new = pzt_old + exp(logrw(t)) * eta;
    idx_outlier = find(abs(pzt_new) > zmax);
    if ~isempty(idx_outlier)
        nidx = length(idx_outlier);
        for jj = 1:nidx
            pztjj = pzt_new(idx_outlier(jj));
            pztjj_sign = sign(pztjj);
            pzt_new(idx_outlier(jj)) = zmax * pztjj_sign;
        end
    end
    
    

    % Compute log prior prob
    if t == 1
        pztp1 = pz(t+1,:)'; 
%         logprior_old = -1*sum(pzt_old.^2) + sum(pzt_old.*pztp1);
%         logprior_new = -1*sum(pzt_new.^2) + sum(pzt_new.*pztp1);
        tmp = ((2+r0)./(1+r0)); 
        logprior_old = -0.5*sum(tmp.*(pzt_old.^2)) + sum(pzt_old.*pztp1);
        logprior_new = -0.5*sum(tmp.*(pzt_new.^2)) + sum(pzt_new.*pztp1);               
    elseif t == n
        pztm1 = pz(t-1,:)';
        logprior_old = -0.5*sum(pzt_old.^2) + sum(pzt_old.*pztm1);
        logprior_new = -0.5*sum(pzt_new.^2) + sum(pzt_new.*pztm1);
    else
        pztm1 = pz(t-1,:)';
        pztp1 = pz(t+1,:)';
        logprior_old = -1*sum(pzt_old.^2) + sum(pzt_old.*(pztm1 + pztp1));
        logprior_new = -1*sum(pzt_new.^2) + sum(pzt_new.*(pztm1 + pztp1));
    end
    

    % Compute log likelihood
    tmp = double(pzt_old>0) .* pzt_old;
    wt_old = v2 .* tmp;
    loglike_old = loglike_MI(yt, xt, mut, omegat, mtm1, Mtm1, sig2t, wt_old); 

    tmp = double(pzt_new>0) .* pzt_new;
    wt_new = v2 .* tmp;    
    loglike_new = loglike_MI(yt, xt, mut, omegat, mtm1, Mtm1, sig2t, wt_new); 

    
    % Compute log proposal
    % Not needed for RW proposal
    
    
    % Compute log acceptance prob
    logprob = logprior_new + loglike_new - logprior_old - loglike_old;
    if log(rand) <= logprob
        pz(t,:) = pzt_new';
        if drawi > burnin
            count(t) = 1;
        end
    end
    tmp = double(pz(t,:)>0) .* pz(t,:);
    ind2(t,:) = v2' .* tmp;
    
    
    % Update proposal stdev
    p = exp(min(0,logprob));
    ei = max(200, drawi/K);
    ei_start = max(200, drawi_start(t)/K);
    d = max(ei - ei_start, 20);
    logrwt = logrw_old(t) + AMH_c * (p - pstar)/d;
    if abs(logrwt - logrw_start(t)) > 1.0986 %log(3) ~= 1.0986 
        drawi_start(t) = drawi;
        logrw_start(t) = logrw_old(t);
    end %restart when useful to allow for larger movement    
    logrw(t) = logrwt;
    
    
    % Use the new zt to update the sample covariance
    zt_mean_old = pz_mean_old(t,:)';
    zt_cov_old = pz_cov_old{t};
    zt_mean = (zt_mean_old * (drawi-1) + pz(t,:)') / drawi;
    zt_cov = (drawi - 1) * (zt_cov_old + zt_mean_old * zt_mean_old') / drawi + ...
        pz(t,:)' * pz(t,:) / drawi - zt_mean * zt_mean';
    pz_mean(t,:) = zt_mean';
    pz_cov{t} = zt_cov;
    
    
    % Use the new ind_t to update mt and Mt for use at t+1
    if t < n
        wt = ind2(t,:)';
        [mt, Mt] = Kalman_iteration(mtm1, Mtm1, yt, xt, sig2t, wt);
    end
end



