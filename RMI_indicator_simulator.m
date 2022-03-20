% Consider the mixture innovation model:
% scalar: yt = xt' * bt + N(0,sig2t), 
% k-by-1: bt = btm1 + N(0,diag(wt)), 
% where wt = ind_t .* v, ind_t is a k-by-1 vector of 0/1 indictors for shrinkage
%
% simulate ind_t from the posterior p(ind_t | y,x,sig2,v) integrating out b
%
% restrict scenarios of k such that zero, one or all betas are TVP,
% the number of scenarios reduces from 2^K to K+2


function ind_new = RMI_indicator_simulator(y, x, sig2, v, ind_old,...
    b0_mean, b0_cov, q, ind_scenarios)
% Inputs:
%   y: a n-by-1 vector of target,
%   x: a n-by-k matrix of regressors,
%   sig2: a n-by-1 vector of target innovation variance,
%   v: a k-by-1 vector of the invariant part of state variances,
%   ind_old: a n-by-k matrix stacking ind_t from previous sweep,
%   b0_mean: a k-by-1 vector of prior mean of b0,
%   b0_cov: a k-by-k matrix of prior covariance matrix of b0,
%   transit_mat: a k-by-1 cell of 2-by-2 transition matrices,
%   stationary_distr: a k-by-1 vector of stationary distr for ind = 0,
%   ind_scenarios: a (2^k)-by-k matrix stacking all possible scenarios of an indicator vector
% Outputs:
%   ind_new: a n-by-k matrix stacking ind_t from current sweep.

[n,k] = size(x);

%% Collect all possible values of indicators 
% ind_mat = indicator_matrix(k); 
% n2k = 2^k;
n2k = size(ind_scenarios,1);


%% Backward recursion for omega, mu
w_old = ind_old .* kron(ones(n,1),v');
[omega, mu] = MI_backward_recursion(y, x, sig2, w_old);


%% Forward simulation of indicators
ind_new = ind_old;
for t = 1:n
    % Collect items for t
    xt = x(t,:)';
    yt = y(t);
    sig2t = sig2(t);    
    mut = mu(t,:)';
    omegat = omega{t};
    if t == 1
        mtm1 = b0_mean;
        Mtm1 = b0_cov;
    else
        mtm1 = mt;
        Mtm1 = Mt;
    end

    % Compute posterior prob for each possible scenario of ind_t
    logpost_vec = zeros(n2k,1);
    for j = 1:n2k 
        ind_j = ind_scenarios(j,:)';
        wt = ind_j .* v;
        
        % log likelihood
        [mt, Mt] = Kalman_iteration(mtm1, Mtm1, yt, xt, sig2t, wt);
        Tt = chol(Mt)';
        phit = Tt' * (mut - omegat * mt);
        rt = xt' * mtm1;
        Rt = sig2t + xt' * Mtm1 * xt;

        tmp = eye(k) + Tt' * omegat * Tt;
        Rxt = Rt + xt' * diag(wt) * xt; 
        tmp_half = chol(tmp);
        logdet_tmp = 2 * sum(log(diag(tmp_half)));
        loglike = -0.5*log(Rxt) - 0.5 * logdet_tmp ...
            -0.5 * mt' * omegat * mt + mut' * mt + 0.5 * phit' * (tmp \ phit) ...
            -0.5 * (yt-rt) * (yt-rt) / Rxt;
        
        
%         % log prior p(ind_t|ind_tm1)
%         if t > 1
%             ind_tm1 = ind_new(t-1,:)';
%             pvec1 = transition_prob(ind_tm1, ind_j, transit_mat);
%         else
%             pvec1 = zeros(k,1);
%             idx0 = find(ind_j == 0);
%             idx1 = setdiff((1:k)',idx0);
%             pvec1(idx0) = stationary_distr(idx0);
%             pvec1(idx1) = 1 - stationary_distr(idx1);
% %             for kj = 1:k
% %                 if ind_j(kj) == 0
% %                     pvec1(kj) = stationary_distr(kj);
% %                 else
% %                     pvec1(kj) = 1 - stationary_distr(kj);
% %                 end
% %             end
%         end
%         
%         
%         
%         % log prior p(ind_tp1|ind_t)
%         if t < n
%             ind_tp1 = ind_new(t+1,:)';
%             pvec2 = transition_prob(ind_j, ind_tp1, transit_mat);
%         else
%             pvec2 = ones(k,1);
%         end
%         
%         
%         % log prior
%         logprior = sum(log(pvec1)) + sum(log(pvec2));


        % log prior
        if t == 1
            ind_tp1 = ind_new(t+1,:)';
            if isequal(ind_j, ind_tp1)
                p2 = q;
            else
                p2 = (1 - q)/(n2k-1);
            end
            logprior = log(p2);
        elseif t == n
            ind_tm1 = ind_new(t-1,:)';
            if isequal(ind_j, ind_tm1)
                p1 = q;
            else
                p1 = (1 - q)/(n2k-1);
            end            
            logprior = log(p1);
        else
            ind_tm1 = ind_new(t-1,:)';
            ind_tp1 = ind_new(t+1,:)';
            if isequal(ind_j, ind_tm1)
                p1 = q;
            else
                p1 = (1 - q)/(n2k-1);
            end  
            if isequal(ind_j, ind_tp1)
                p2 = q;
            else
                p2 = (1 - q)/(n2k-1);
            end
            logprior = log(p1) + log(p2);
        end
        
        
        
        % log posterior
        logpost_vec(j) = logprior + loglike; 
        if isinf(logpost_vec(j))
            error('inf logpost');
        end
    end
    
    % Simulate ind_t based on its posterior prob
    tmp = logpost_vec - mean(logpost_vec);
    ind_t_kernel = exp(tmp);
%     ind_t_kernel = exp(logpost_vec);
    ind_t_prob = ind_t_kernel/sum(ind_t_kernel);
    Kt = find(mnrnd(1, ind_t_prob) == 1);
    ind_t = ind_scenarios(Kt,:)';
    ind_new(t,:) = ind_t'; 
    
    % Use the new ind_t to update mt and Mt for use at t+1
    wt = ind_t .* v;
    [mt, Mt] = Kalman_iteration(mtm1, Mtm1, yt, xt, sig2t, wt);
end


