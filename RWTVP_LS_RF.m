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



function draws = RWTVP_LS_RF(y, x, burnin, ndraws, ind_SV, ind_sparse, ind_forecast)
% Inputs:
%   y: a n-by-1 vector of target data
%   x: a n-by-K matrix of regressor data (including constant)
%   burnin: a scalar of the number of burnins
%   ndraws: a scalar of the number of effective draws after burnin
%   ind_SV: an indicator if SV for measurement noise variance
%   ind_sparse: an indicator if sparsifying is performed (not applicable here; always set 0)
%   ind_forecast: an indicator if Kalman filter is run for subsequent forecasts
% Outputs:
%   draws: a structure of the final draws.


[n,K] = size(x);
% minNum = 1e-100;
zmax = 1e10;


%% Priors: initial beta, beta0 ~ N(0, taul * diag(phil)), taul, phil are IBs
phil_d = 1./gamrnd(0.5,1,K,1);
phil = 1./gamrnd(0.5*ones(K,1),phil_d); %local variances
taul_d = 1/gamrnd(0.5,1);
taul = 1/gamrnd(0.5, taul_d); %global variance
psil = taul*phil; 
beta0 = sqrt(psil) .* randn(K,1);
% beta0 = zeros(K,1);


%% Priors: scaling factor for state noise, v ~ N(0, diag(psi)) 
phi_d = 1./gamrnd(0.5,1,K,1);
% phi = 1./gamrnd(0.5*ones(K,1),phi_d); %local variances
phi = ones(K,1);
tau_d = 1/gamrnd(0.5,1);
% tau = 1/gamrnd(0.5, tau_d); %global variance
tau = 10;
psi = tau*phi; 
v = sqrt(psi) .* randn(K,1); %scaling factor for state noise
% v = zeros(K,1);
v2 = v.^2;




%% Initialize latent index
r0 = 10*ones(K,1); %prior variance of z0
pz = zeros(n,K);
ind2 = repmat(v2',n,1).*double(pz>0) .* pz;


%% Initiaze state variance
state_var = cell(n,1);
for t = 1:n
    state_var{t} = diag(ind2(t,:));
end %covar matrices of state noise for simulation smoother (SA)


%% Set up adaptive MH
pstar = 0.25;
tmp_const = -norminv(0.5*pstar);
AMH_c = 1/(K * pstar * (1-pstar)) + (1-1/K)*0.5*sqrt(2*pi)*...
    exp(0.5*tmp_const*tmp_const)/tmp_const;

logrw = zeros(n,1);
logrw_start = logrw;
drawi_start = zeros(n,1); %z

logrw_v = 0;
logrw_start_v = logrw_v;
drawi_start_v = 0; %v

logrw_beta0 = 0;
logrw_start_beta0 = logrw_beta0;
drawi_start_beta0 = 0; %beta0


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
draws.beta0 = zeros(ndraws,K); 
draws.count_beta0 = 0;
draws.logrw_beta0 = zeros(ndraws,1); %beta0

draws.tau = zeros(ndraws,2);
draws.phi = zeros(ndraws,2*K);
draws.v = zeros(ndraws,K);
draws.count_v = 0;
draws.logrw_v = zeros(ndraws,1); %v

draws.beta = cell(K,1);
for j = 1:K
    draws.beta{j} = zeros(ndraws,n);
end %TVP

draws.count_pz = zeros(n,1);
draws.logrw = zeros(ndraws,n);
draws.pz = cell(K,1);
draws.ind2 = cell(K,1);
for j = 1:K
    draws.pz{j} = zeros(ndraws,n);
    draws.ind2{j} = zeros(ndraws,n);
end %z

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
pz_mean = zeros(n,K);
pz_cov = cell(n,1);
for t = 1:n
    pz_cov{t} = zeros(K,K);
end
v_mean = zeros(K,1);
v_cov = zeros(K,K);
beta0_mean = zeros(K,1);
beta0_cov = zeros(K,K);
for drawi = 1:ntotal   
    % Draw indicator (SA, integrating out beta， multivariate)
    b0_mean = beta0;
    b0_cov = zeros(K,K);    
    pz_mean_old = pz_mean;
    pz_cov_old = pz_cov;
    pz_old = pz;
    ind2_old = ind2;
    logrw_old = logrw;     
    [pz, ind2, logrw, count_pz, pz_mean, pz_cov] = pz_simulator_LS_RF(y, x,...
        vary, r0, v2, pz_old, ind2_old, b0_mean, b0_cov, pstar, AMH_c, logrw_old,...
        logrw_start, drawi_start, drawi, burnin, pz_mean_old, pz_cov_old); 
    
    
    % Draw v (SA, integrating out beta， multivariate)
    count_v = 0;

    v_old = v;
    if drawi < 100
        A = eye(K);
    else  
        A = v_cov + 1e-6 * eye(K) / drawi; %add a small constant
    end
    eps = mvnrnd(zeros(K,1),A)'; %correlated normal
    v_new = v_old + exp(logrw_v) * eps;
    idx = find(v_new == 0);
    if ~isempty(idx)
        v_new(idx) = 1e-100;
    end    
    
    v2_old = v_old.^2;
    v2_new = v_new.^2;
    logprior_old = -0.5 * sum(v2_old./psi);%p(v)
    logprior_new = -0.5 * sum(v2_new./psi);

    w_old = ind2;
    loglike_old = loglike_TVP2(y, x, vary, w_old, beta0);
    w_new = ind2 .* repmat(v2_new'./v2_old',n,1); 
    loglike_new = loglike_TVP2(y, x, vary, w_new, beta0);

    logprob = logprior_new + loglike_new - logprior_old - loglike_old;
    if log(rand) <= logprob
        v = v_new;
        ind2 = w_new;
        if drawi > burnin
            count_v = 1;
        end
    end
     

    p = exp(min(0,logprob));
    ei = max(200, drawi/K);
    ei_start = max(200, drawi_start_v/K);
    d = max(ei - ei_start, 20);
    logrwj = logrw_v + AMH_c * (p - pstar)/d;
    if abs(logrwj - logrw_start_v) > 1.0986 %log(3) ~= 1.0986 
        drawi_start_v = drawi;
        logrw_start_v = logrw_v;
    end %restart when useful to allow for larger movement    
    logrw_v = logrwj; %update proposal stdev 
    
    
    v_mean_old = v_mean;
    v_cov_old = v_cov;
    v_mean = (v_mean_old * (drawi-1) + v) / drawi;
    v_cov = (drawi - 1) * (v_cov_old + v_mean_old * v_mean_old') / drawi + ...
        v * v' / drawi - v_mean * v_mean'; %update the sample covariance   
    
    
    % Draw beta0 (SA, integrating out beta， multivariate)
    count_beta0 = 0;

    beta0_old = beta0;
    if drawi < 100
        A = eye(K);
    else  
        A = beta0_cov + 1e-6 * eye(K) / drawi; %add a small constant
    end
    eps = mvnrnd(zeros(K,1),A)'; %correlated normal
    beta0_new = beta0_old + exp(logrw_beta0) * eps; 

    logprior_old = -0.5 * sum((beta0_old.^2) ./ psil);%p(beta0)
    logprior_new = -0.5 * sum((beta0_new.^2) ./ psil);
    
    w = ind2;
    loglike_old = loglike_TVP2(y, x, vary, w, beta0_old); 
    loglike_new = loglike_TVP2(y, x, vary, w, beta0_new);

    logprob = logprior_new + loglike_new - logprior_old - loglike_old;
    if log(rand) <= logprob
        beta0 = beta0_new;
        if drawi > burnin
            count_beta0 = 1;
        end
    end
     

    p = exp(min(0,logprob));
    ei = max(200, drawi/K);
    ei_start = max(200, drawi_start_beta0/K);
    d = max(ei - ei_start, 20);
    logrwj = logrw_beta0 + AMH_c * (p - pstar)/d;
    if abs(logrwj - logrw_start_beta0) > 1.0986 %log(3) ~= 1.0986 
        drawi_start_beta0 = drawi;
        logrw_start_beta0 = logrw_beta0;
    end %restart when useful to allow for larger movement    
    logrw_beta0 = logrwj; %update proposal stdev 
    
    
    beta0_mean_old = beta0_mean;
    beta0_cov_old = beta0_cov;
    beta0_mean = (beta0_mean_old * (drawi-1) + beta0) / drawi;
    beta0_cov = (drawi - 1) * (beta0_cov_old + beta0_mean_old * beta0_mean_old') / drawi + ...
        beta0 * beta0' / drawi - beta0_mean * beta0_mean'; %update the sample covariance     
    
    
    % Draw beta (SA) 
    for t = 1:n
        state_var{t} = diag(ind2(t,:));
    end %covar matrices of state noise for simulation smoother (SA)    
    beta = Simulation_Smoother_DK(y, x, vary, state_var(2:n),...
        beta0, state_var{1});     
    
    
     % Asis: compute beta_star (AA)
    beta_star = (beta - repmat(beta0',n,1)) ./ repmat(v',n,1);

    
    % Asis: update v, beta0 based on beta_star (AA)
    x_beta0v2 = [x  x.* beta_star];

    psi_beta0v = [psil; psi];
    sigy = sqrt(vary);
    xstar = x_beta0v2 ./ repmat(sigy,1,K+K);
    ystar = y ./ sigy;
    A_inv = diag(1./psi_beta0v) + xstar' * xstar;
    if rcond(A_inv) > 1e-15
        A_inv_half = chol(A_inv);
        a_beta0v = A_inv \ (xstar' * ystar);
        beta0v = a_beta0v + A_inv_half \ randn(K+K,1);
    else
        A_beta0v = robust_inv(A_inv);
        A_half = robust_chol(A_beta0v);
        a_beta0v = A_beta0v * (xstar' * ystar);
        beta0v = a_beta0v + A_half * randn(K+K,1);
    end
    beta0 = beta0v(1:K);
    v = beta0v(K+1:K+K);
    v2 = v.^2;
    
    
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % Linear regression for beta0, v (AA)
%     % be careful with collineartity when some beta_star close to constant
%     beta_star_std = std(beta_star)';
%     minstd = 1e-8;
%     idx_const = find(beta_star_std < minstd); %treat as constant
%     idx_others = find(beta_star_std >= minstd);    
%     if isempty(idx_const)
%         beta_star_scaled = beta_star ./repmat(beta_star_std',n,1);
%         z = x .* beta_star_scaled; 
%         zz = [x z];
%         yy = y;
%         psi_beta0v = [psil; (beta_star_std.^2).*psi];
%     elseif isempty(idx_others)
%         zz = x;
%         yy = y - (x.*beta_star) * v;
%         psi_beta0v = psil;
%     else
%         beta_star_scaled = beta_star(:,idx_others) ./repmat(beta_star_std(idx_others)',n,1);
%         z = x(:,idx_others) .* beta_star_scaled;
%         zz = [x z];
%         yy = y - (x(:,idx_const).*beta_star(:,idx_const)) * v(idx_const);
%         psi_beta0v = [psil; (beta_star_std(idx_others).^2).*(psi(idx_others))];
%     end
%     sigy = sqrt(vary);
%     nxz = size(zz,2);
%     xstar = zz ./ repmat(sigy,1,nxz);
%     ystar = yy ./ sigy;
%     A_inv = diag(1./psi_beta0v) + xstar' * xstar;
%     if rcond(A_inv) > 1e-15
%         A_inv_half = chol(A_inv);
%         a = A_inv \ (xstar' * ystar);
%         beta0v = a + A_inv_half \ randn(nxz,1);
%     else
%         A = robust_inv(A_inv);
%         A_half = robust_chol(A);
%         a = A * (xstar' * ystar);
%         beta0v = a + A_half * randn(nxz,1);
%     end
% 
%     beta0 = beta0v(1:K);
%     if nxz > K
%         v_scaled = beta0v(K+1:nxz);
%         v(idx_others) = v_scaled ./ beta_star_std(idx_others);
%     end
% %     v(isinf(v)) = maxNum;
%     v(v==0)=1e-100;
% %     v_sign = sign(v);
% %     log_v_abs = log(abs(v));
% %     log_v2 = 2 * log_v_abs;
% %     v2 = exp(log_v2);    
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    % Asis: compute back beta 
    beta = beta_star .* repmat(v',n,1)  + repmat(beta0',n,1); 
    
    
    % Finalize the process variance of the sweep
    ind2 = repmat(v2',n,1).*double(pz>0).*pz;
    
    
    % Update hyperparameters of v
%     tmp = v.^2; 
%     [tau, tau_d, phi, phi_d] = Horseshoe_update_vector(tmp, tau, tau_d, ...
%         phi, phi_d); 
%     tau = 10;
%     phi = ones(K,1);
%     psi = tau * phi;
    
    
    % Update hyperparameters of beta0
    tmp = beta0.^2; 
    [taul, taul_d, phil, phil_d] = Horseshoe_update_vector(tmp, taul, taul_d, ...
        phil, phil_d);   
    psil = taul * phil;    
    
    
    % Residual variance
    yfit = sum(x .* beta,2);
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
    
    
    % Sparsify beta if needed
    if ind_sparse == 1
        z = x .* beta_star;
        v_sparse = SAVS_vector(v, z); 
        
        beta0_sparse = SAVS_vector(beta0,x);
        beta_sparse = beta_star .* repmat(v_sparse',n,1) + repmat(beta0_sparse',n,1);          
    end
    
    
    % Compute mean and covar of p(bn|y1,...,yn) for subsequent forecasts
    if ind_forecast == 1 
        for t = 1:n
            state_var{t} = diag(ind2(t,:));
        end %covar matrices of state noise for simulation smoother (SA)        
        P1 = state_var{1};
        a1 = beta0; %bstar1 = 0
        [bn_mean, bn_cov] = Kalman_filter_robust(y, x, ...
            vary, state_var(2:n), a1, P1);                  
        if ind_sparse == 1
            bn_smean = v_sparse.*bstarn_mean+ beta0_sparse;
            bn_scov = (v_sparse*v_sparse') .* bstarn_cov;
        end       
    end    
    

    % Collect draws
    if drawi > burnin
        i = drawi - burnin;
        for j = 1:K
            draws.beta{j}(i,:) = beta(:,j)';
        end           
        
        draws.v(i,:) = v';
        draws.tau(i,:) = [tau  tau_d];
        draws.phi(i,:) = [phi'  phi_d'];        
        draws.count_v = draws.count_v + count_v; 
        draws.logrw_v(i) = logrw_v;%v        
        
        draws.beta0(i,:) = beta0';
        draws.taul(i,:) = [taul  taul_d];
        draws.phil(i,:) = [phil'  phil_d'];
        draws.count_beta0 = draws.count_beta0 + count_beta0; 
        draws.logrw_beta0(i) = logrw_beta0;%beta0        

        draws.count_pz = draws.count_pz + count_pz; 
        draws.logrw(i,:) = logrw';
        for j = 1:K
            draws.pz{j}(i,:) = pz(:,j)';
            draws.ind2{j}(i,:) = ind2(:,j)';
        end %z        
        
        draws.yfit(i,:) = yfit';                
        
        if ind_SV == 1
            draws.sig2(i,:) = vary';
            draws.SVpara(i,:) = [muh phih sigh^2 sigh sigh2_s lambdah];
        else
            draws.sig2(i) = sig2;
        end        
        
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
    if (drawi/5000) == round(drawi/5000)
        disp([num2str(drawi), ' out of ', num2str(ntotal),' draws have completed!']);
        toc;
    end    
end
draws.count_pz = draws.count_pz / ndraws;
draws.count_v = draws.count_v / ndraws;
draws.count_beta0 = draws.count_beta0 / ndraws;


    





