% Estimate a mixture-innovation RW-TVP model:
% yt = xt' * bt + N(0,sig2t), 
% b_jt - b_{j,t-1}~N(0,k_jt * wj),
% where
% vj=+/-sqrt(wj)~N(0, tau * tauj_j)  => tau*tauj_j = 10
% b_j0 ~ N(0, taul * phil_j) => taul, phil are IBs
%
% k_jt is 0 or 1 and is driven by a Markov chain Q 
%
% use GK algorithm for k
% use ASIS for v, b0
%
% restrict scenarios of k such that zero, one or all betas are TVP,
% the number of scenarios reduces from 2^K to K+2



function draws = RWTVP_RMI(y, x, burnin, ndraws, ind_SV, ind_sparse, ind_forecast, MI_scenarios)
% Inputs:
%   y: a n-by-1 vector of target data
%   x: a n-by-K matrix of regressor data (including constant)
%   burnin: a scalar of the number of burnins
%   ndraws: a scalar of the number of effective draws
%   ind_SV: an indicator if SV for measurement noise variance
%   ind_sparse: an indicator if sparsifying is performed
%   ind_forecast: an indicator if Kalman filter is run for subsequent forecasts
% Outputs:
%   draws: a structure of the final draws.


[n,K] = size(x);
minNum = 1e-100;
maxNum = 1e100;


%% Priors: initial beta, beta0 ~ N(0, taul * diag(phil)), taul, phil are IBs
phil_d = 1./gamrnd(0.5,1,K,1);
phil = 1./gamrnd(0.5*ones(K,1),phil_d); %local variances
taul_d = 1/gamrnd(0.5,1);
taul = 1/gamrnd(0.5, taul_d); %global variance
psil = taul*phil; 
beta0 = sqrt(psil) .* randn(K,1);


%% Priors: scaling factor for state noise 
% tau_d = 1/gamrnd(0.5,1);
% tau = 1/gamrnd(0.5, tau_d); %global variance
tau = 10;

% tauj_d = 1./gamrnd(0.5,1,K,1);
% tauj = 1./gamrnd(0.5*ones(K,1),tauj_d); %individual variances
tauj = ones(K,1);

v = sqrt(tau * tauj) .* randn(K,1); %scaling factor for state noise
v2 = v.^2;


%% Priors: mixture innovation
% q_j0 ~ Beta(aj0, bj0), q_j1 ~ Beta(aj1, bj1)
% p(Kjt = 1|Kjtm1 = 0) = 1 - q_j0
% q0_a = 50*ones(K,1); %30*ones(K,1);
% q0_b = 0.5*ones(K,1); %0.3*ones(K,1);
% q1_a = q0_a;
% q1_b = q0_b;
q0_a = 50;
q0_b = 0.5;
q = betarnd(q0_a, q0_b);

% q_a = [q0_a  q1_a];
% q_b = [q0_b  q1_b];
% qmat = betarnd(q_a, q_b);
% qmat = 0.95 * ones(K,2); %initialize transition prob
% transit_mat = cell(K,1);
% stationary_distr = zeros(K,1);
% for j = 1:K
%     transit_mat{j} = zeros(2,2);
%     transit_mat{j}(1,1) = qmat(j,1);
%     transit_mat{j}(1,2) = 1 - qmat(j,1);
%     transit_mat{j}(2,2) = qmat(j,2);
%     transit_mat{j}(2,1) = 1 - qmat(j,2);
%     tmp = Markov_chain_coverge(transit_mat{j});
%     stationary_distr(j) = tmp(1);
% end
ind = zeros(n,K); %initialize Kjt
ind_old = ind;
% ind_scenarios = indicator_matrix(K); %all possible 2^K scenarios of the ind vector
% ind_scenarios = [zeros(1,K); eye(K); ones(1,K)]; %restricted scenarios of the ind vector


%% Initiaze
x_star = x .* repmat(v',n,1);
state_var = cell(n,1);
for t = 1:n
    state_var{t} = diag(ind(t,:));
end %covar matrices of state noise for simulation smoother (AA)


%% Priors: SV or constant measurement noise variance
if ind_SV == 1
    % long-run mean: p(mu) ~ N(mu0, Vmu), e.g. mu0 = 0; Vmu = 10;
    % persistence: p(phi) ~ N(phi0, Vphi)I(-1,1), e.g. phi0 = 0.95; invVphi = 0.04;
    % variance: p(sig2) ~ G(0.5, 2*sig2_s), sig2_s ~ IG(0.5,1/lambda), lambda ~ IG(0.5,1)
    muh0 = 0; invVmuh = 1/10; % mean: p(mu) ~ N(mu0, Vmu)
    phih0 = 0.95; invVphih = 1/0.04; % AR(1): p(phi) ~ N(phi0, Vphi)I(-1,1)
    priorSV = [muh0 invVmuh phih0 invVphih 0 0]'; %collect prior hyperparameters
    muh = muh0 + sqrt(1/invVmuh) * randn;
    phih = phih0 + sqrt(1/invVphih) * trandn((-1-phih0)*sqrt(invVphih),(1-phih0)*sqrt(invVphih));

    lambdah = 1/gamrnd(0.5,1);
    sigh2_s = 1/gamrnd(0.5,lambdah);
    sigh2 = gamrnd(0.5,2*sigh2_s);
    sigh = sqrt(sigh2);

    hSV = log(var(y))*ones(n,1); %initialize by log OLS residual variance.
    vary = exp(hSV);
else %Jeffery's prior p(sig2) \prop 1/sig2
    sig2 = var(y); %initialize
    vary = sig2 * ones(n,1);
end    


%% MCMC
draws.taul = zeros(ndraws,2);
draws.phil = zeros(ndraws,2*K);
draws.beta0 = zeros(ndraws,K); %beta0

% draws.tau = zeros(ndraws,2);
% draws.tauj = zeros(ndraws,2*K);
draws.v = zeros(ndraws,K);
draws.beta = cell(K,1);
for j = 1:K
    draws.beta{j} = zeros(ndraws,n);
end %TVP

% draws.q0 = zeros(ndraws,K);
% draws.q1 = zeros(ndraws,K);
draws.q = zeros(ndraws,1);
draws.ind = cell(K,1);
for j = 1:K
    draws.ind{j} = zeros(ndraws,n);
end %MI

if ind_SV == 1
    draws.SVpara = zeros(ndraws,6); % [mu phi sig2 sig sig2_s lambda]
    draws.sig2 = zeros(ndraws,n); %residual variance
else
    draws.sig2 = zeros(ndraws,1);
end

draws.yfit = zeros(ndraws,n);

if ind_sparse == 1
    draws.v_sparse = zeros(ndraws,K);
    draws.beta0_sparse = zeros(ndraws,K);
    draws.beta_sparse = cell(K,1);
    for j = 1:K
        draws.beta_sparse{j} = zeros(ndraws,n);
    end
end

if ind_forecast == 1
    draws.bn_mean = zeros(ndraws,K);
    draws.bn_cov = cell(ndraws,1);
    for j = 1:ndraws
        draws.bn_cov{j} = zeros(K,K);
    end
    if ind_sparse == 1
        draws.bn_smean = zeros(ndraws,K);
        draws.bn_scov = cell(ndraws,1);
        for j = 1:ndraws
            draws.bn_scov{j} = zeros(K,K);
        end
    end    
end

tic;
ntotal = burnin + ndraws;
for drawi = 1:ntotal   
    % Draw indicator
    b0_mean = zeros(K,1);
    b0_cov = diag(taul * phil);
    ind = RMI_indicator_simulator(y, x, vary, v2, ind_old,...
        b0_mean, b0_cov, q, MI_scenarios);
    ind_old = ind;
    for t = 1:n
        state_var{t} = diag(ind(t,:));
    end %covar matrices of state noise for simulation smoother (AA)      
    
    
    % TVP, beta_star (AA)
    beta_star = Simulation_Smoother_DK(y-x*beta0, x_star, vary, state_var(2:n),...
        zeros(K,1), state_var{1}); 

    
    % Linear regression for beta0, v (AA)
    [idx_zeros, idx_others] = find_zero_columns(beta_star); %ind = 0 => beta_star = 0
    if ~isempty(idx_others)
        K1 = length(idx_others);
    
        z = x(:,idx_others) .* beta_star(:,idx_others);
        zz = [x z];
    
        psi = [taul * phil; tau * tauj(idx_others)];
        sigy = sqrt(vary);
        xstar = zz ./ repmat(sigy,1,K+K1);
        ystar = y ./ sigy;
        A_inv = diag(1./psi) + xstar' * xstar;
        if rcond(A_inv) > 1e-15
            A_inv_half = chol(A_inv);
            a = A_inv \ (xstar' * ystar);
            beta0v = a + A_inv_half \ randn(K+K1,1);
        else
            A = robust_inv(A_inv);
            A_half = robust_chol(A);
            a = A * (xstar' * ystar);
            beta0v = a + A_half * randn(K+K1,1);
        end

        beta0 = beta0v(1:K);
        v(idx_others) = beta0v(K+1:K+K1);
        if ~isempty(idx_zeros)
            v(idx_zeros) = sqrt(tau * tauj(idx_zeros)) .* randn(K-K1,1);
        end
    else %all v from prior
        zz = x;
    
        psi = taul * phil;
        sigy = sqrt(vary);
        xstar = zz ./ repmat(sigy,1,K);
        ystar = y ./ sigy;
        A_inv = diag(1./psi) + xstar' * xstar;
        if rcond(A_inv) > 1e-15
            A_inv_half = chol(A_inv);
            a = A_inv \ (xstar' * ystar);
            beta0v = a + A_inv_half \ randn(K,1);
        else
            A = robust_inv(A_inv);
            A_half = robust_chol(A);
            a = A * (xstar' * ystar);
            beta0v = a + A_half * randn(K,1);
        end

        beta0 = beta0v(1:K);
        v = sqrt(tau * tauj) .* randn(K,1);        
    end
    v2 = v.^2;
% beta0 = b0_true;
% v2 = v2true;
% v = sqrt(v2);
    v_sign = sign(v); 
         
    
    % ASIS: compute beta
    beta = beta_star .* repmat(v',n,1) + repmat(beta0',n,1);
    diff_beta = [beta(1,:)-beta0'; beta(2:n,:) - beta(1:(n-1),:)];
    diff_beta2 = diff_beta.^2;
    
    
    % ASIS: update v2 (SA)
    for j = 1:K
        indj = ind(:,j);
        idx = find(indj == 1);
        count_one = length(idx);
        if count_one > 0
            [v2(j),~] = gigrnd(0.5-0.5*count_one, 1/(tau*tauj(j)), sum(diff_beta2(idx,j)), 1);
            if v2(j) == 0
                v2(j) = minNum;
            end             
        else
            v2(j) = gamrnd(0.5, 2*tau*tauj(j)); %simulate from prior
        end
    end
    v = sqrt(v2) .* v_sign;
% v2 = v2true;
% v = sqrt(v2);    
    
    
    % ASIS: update beta0 (SA)
    psil = taul * phil;
    B_beta0 = 1./(1./psil + 1./(v2.*diag(state_var{1})));
    b_beta0 = beta(1,:)'./(1+(v2.*diag(state_var{1}))./psil);
    beta0 = b_beta0 + sqrt(B_beta0).*randn(K,1);     
% beta0 = b0_true;      
    
    % ASIS: compute back beta_star
    beta_star = (beta - repmat(beta0',n,1)) ./ repmat(v',n,1);


    % Update tau, tauj for v
%     [tau, tau_d, tauj, tauj_d] = Horseshoe_update_vector(v2, tau, tau_d, ...
%         tauj, tauj_d);          
    
    
    % Update hyperparameters of beta0
    tmp = beta0.^2; 
    [taul, taul_d, phil, phil_d] = Horseshoe_update_vector(tmp, taul, taul_d, ...
        phil, phil_d);   

    
    % Residual variance
    x_star = x .* repmat(v',n,1);
    yfit = sum(x_star .* beta_star,2)+ x * beta0;
    eps = y - yfit;
    if ind_SV == 1
        logz2 = log(eps.^2 + 1e-100);
        [hSV, muh, phih, sigh, sigh2_s, lambdah] = SV_update_asis(logz2, hSV, ...
            muh, phih, sigh, sigh2_s, lambdah, priorSV);    
        vary = exp(hSV); 
    else
        sig2 = 1/gamrnd(0.5*n, 2/(eps'*eps));
        vary = sig2 * ones(n,1); 
    end  
    
    % Draw transition prob for indicator
    tn = RMI_transition_numbers(ind);
    q = betarnd(q0_a + tn, q0_b + n - 1 - tn); 
%     
%     for j = 1:K
%         qmat(j,1) = betarnd(q0_a(j)+tn(1), q0_b(j)+tn(2));
%         qmat(j,2) = betarnd(q1_a(j)+tn(4), q1_b(j)+tn(3));
%         transit_mat{j}(1,1) = qmat(j,1);
%         transit_mat{j}(1,2) = 1 - qmat(j,1);
%         transit_mat{j}(2,2) = qmat(j,2);
%         transit_mat{j}(2,1) = 1 - qmat(j,2);
%         tmp = Markov_chain_coverge(transit_mat{j});
%         stationary_distr(j) = tmp(1);
%     end
    
    
    % Sparsify beta if needed
    if ind_sparse == 1
        z = x .* beta_star;
        v_sparse = SAVS_vector(v, z); 
        
        beta0_sparse = SAVS_vector(beta0,x);
        beta_sparse = beta_star .* repmat(v_sparse',n,1) + repmat(beta0_sparse',n,1);          
    end
    
    
    % Compute mean and covar of p(bn|y1,...,yn) for subsequent forecasts
    if ind_forecast == 1 
        P1 = state_var{1};
        a1 = zeros(K,1); %bstar1 = 0
        [bstarn_mean, bstarn_cov] = Kalman_filter_robust(y-x*beta0, x_star, ...
            vary, state_var(2:n), a1, P1);                  
        bn_mean = v.*bstarn_mean+ beta0;
        bn_cov = (v*v') .* bstarn_cov;
        if ind_sparse == 1
            bn_smean = v_sparse.*bstarn_mean+ beta0_sparse;
            bn_scov = (v_sparse*v_sparse') .* bstarn_cov;
        end       
    end    
    

    % Collect draws
    if drawi > burnin
        i = drawi - burnin;
%         draws.tau(i,:) = [tau  tau_d];
%         draws.tauj(i,:) = [tauj' tauj_d'];
        for j = 1:K
            draws.beta{j}(i,:) = beta(:,j)';
        end
        draws.v(i,:) = v';
        
        if ind_SV == 1
            draws.sig2(i,:) = vary';
            draws.SVpara(i,:) = [muh phih sigh^2 sigh sigh2_s lambdah];
        else
            draws.sig2(i) = sig2;
        end
        
        draws.beta0(i,:) = beta0';
        draws.taul(i,:) = [taul  taul_d];
        draws.phil(i,:) = [phil'  phil_d'];
        
        draws.yfit(i,:) = yfit';
        
%         draws.q0(i,:) = qmat(:,1)'; %zeros(ndraws,K);
%         draws.q1(i,:) = qmat(:,2)'; %zeros(ndraws,K);
        draws.q(i) = q;
        for j = 1:K
            draws.ind{j}(i,:) = ind(:,j)';
        end %MI        
        
        if ind_sparse == 1
            draws.v_sparse(i,:) = v_sparse';
            draws.beta0_sparse(i,:) = beta0_sparse';
            for j = 1:K
                draws.beta_sparse{j}(i,:) = beta_sparse(:,j)';
            end
        end
        
        if ind_forecast == 1
            draws.bn_mean(i,:) = bn_mean';
            draws.bn_cov{i} = bn_cov;
            if ind_sparse == 1
                draws.bn_smean(i,:) = bn_smean';
                draws.bn_scov{i} = bn_scov; 
            end            
        end        
    end
    
    
    % Display elapsed time
    if (drawi/1000) == round(drawi/1000)
        disp([num2str(drawi), ' out of ', num2str(ntotal),' draws have completed!']);
        toc;
    end    
end




    





