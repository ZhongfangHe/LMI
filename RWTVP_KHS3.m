% Estimate a heteroskedastic RW-TVP model with Horseshoe prior for state:
% yt = xt' * bt + N(0,sig2), 
% b_jt - b_{j,t-1}~N(0,tau * tauj_j * phi_jt),
% b_j0 ~ N(0, taul * phil_j)
% tau, tauj, taul, phil are IBs
% phi_jt follows AR(1) as in Kowal etc.(2019)


function draws = RWTVP_KHS3(y, x, burnin, ndraws, ind_SV, ind_forecast)
% Inputs:
%   y: a n-by-1 vector of target data
%   x: a n-by-K matrix of regressor data (including constant)
%   burnin: a scalar of the number of burnins
%   ndraws: a scalar of the number of effective draws
%   ind_SV: an indicator if SV for measurement noise variance
%   ind_forecast: an indicator if Kalman filter is run for subsequent forecasts
% Outputs:
%   draws: a structure of the final draws.

[n,K] = size(x);

%% Priors: initial beta, beta0 ~ N(0, taul * diag(phil)), taul, phil are IBs
phil_d = 1./gamrnd(0.5,1,K,1);
phil = 1./gamrnd(0.5*ones(K,1),phil_d); %local variances
taul_d = 1/gamrnd(0.5,1);
taul = 1/gamrnd(0.5, taul_d); %global variance
psil = taul*phil; 
beta0 = sqrt(psil) .* randn(K,1);


%% Priors: TVP
tau_d = 1/gamrnd(0.5,1);
tau = 1/gamrnd(0.5, tau_d); %global variance

tauj_d = zeros(K,1);
tauj = zeros(K,1);
for j = 1:K
    tauj_d(j) = pgdraw(0);
    log_taujj = randn / sqrt(tauj_d(j));
    tauj(j) = exp(log_taujj);
end %individual variances

rhoh_mean = 0.95; %AR coef rhoh~N(rhoh_mean, rhoh_std^2)I(-1,1)
rhoh_std = 1;%0.5;
% rhoh_std = 0.5;
prior_rhoh = [rhoh_mean  1/(rhoh_std^2)]';
tmp = rhoh_mean + rhoh_std * trandn((-1-rhoh_mean)/rhoh_std,(1-rhoh_mean)/rhoh_std); %initialize AR coef
rhoh = tmp * ones(K,1);
phi_d = pgdraw_matrix(zeros(n,K)); %precision of AR noise
eta = (1./sqrt(phi_d)) .* randn(n,K); %AR noise
Hphi = speye(n) - sparse(2:n,1:(n-1),rhoh(1)*ones(1,n-1),n,n);
phih = Hphi\eta; %log local variances
phi = exp(phih); %local variances

tmpmat = tau * (repmat(tauj',n,1) .* phi);
state_var = cell(n,1);
for t = 1:n
    state_var{t} = diag(tmpmat(t,:));
end %covar matrices of state noise for simulation smoother


%% Priors: SV or constant measurement noise variance
if ind_SV == 1
% long-run mean: p(mu) ~ N(mu0, Vmu), e.g. mu0 = 0; Vmu = 10;
% persistence: p(phi) ~ N(phi0, Vphi)I(-1,1), e.g. phi0 = 0.95; invVphi = 0.04;
% variance: p(sig2) ~ G(0.5, 2*sig2_s), sig2_s ~ IG(0.5,1/lambda), lambda ~ IG(0.5,1)    
    muh0 = 0; invVmuh = 1/10; % mean: p(mu) ~ N(mu0, Vmu)
    phih0 = 0.95; invVphih = 1/0.04; % AR(1): p(phi) ~ N(phi0, Vphi)I(-1,1)
    priorSV = [muh0 invVmuh phih0 invVphih]'; %collect prior hyperparameters
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


% %% Preparation if sparsification is needed
% ind_sparse = 0; %if beta should be sparsified


%% MCMC
draws.taul = zeros(ndraws,2);
draws.phil = zeros(ndraws,2*K);
draws.beta0 = zeros(ndraws,K); %initial beta0

draws.tau = zeros(ndraws,2);
draws.tauj = zeros(ndraws,2*K);
draws.rhoh = zeros(ndraws,K);
draws.phi = cell(K,1);
draws.phi_d = cell(K,1);
draws.beta = cell(K,1);
for j = 1:K
    draws.phi{j} = zeros(ndraws,n);
    draws.phi_d{j} = zeros(ndraws,n);
    draws.beta{j} = zeros(ndraws,n);
end %TVP
    
if ind_SV == 1
    draws.SVpara = zeros(ndraws,6); % [mu phi sig2 sig sig2_s lambda]
    draws.sig2 = zeros(ndraws,n); %residual variance
else
    draws.sig2 = zeros(ndraws,1);
end

draws.yfit = zeros(ndraws,n);

if ind_forecast == 1
    draws.bn_mean = zeros(ndraws,K);
    draws.bn_cov = cell(ndraws,1);
    for j = 1:ndraws
        draws.bn_cov{j} = zeros(K,K);
    end
end

tic;
ntotal = burnin + ndraws;
for drawi = 1:ntotal    
    % TVP beta
    beta = Simulation_Smoother_DK(y, x, vary, state_var(2:n), beta0, state_var{1}); 
    

    % Initial value beta0
    psil = taul * phil;
    B_beta0 = 1./(1./psil + 1./diag(state_var{1}));
    b_beta0 = beta(1,:)'./(1+diag(state_var{1})./psil);
    beta0 = b_beta0 + sqrt(B_beta0).*randn(K,1); 
    
    
    % ASIS: compute beta_star = beta - beta0
    beta_star = beta - repmat(beta0',n,1);
    
    
    % ASIS: use beta_star to update beta0
    ystar = y - sum(x.*beta_star,2);
    sigy = sqrt(vary);
    ystar = ystar./sigy;
    xstar = x./repmat(sigy,1,K);
    psi = taul*phil;
    A_inv = diag(1./psi) + xstar' * xstar;   
    if rcond(A_inv) > 1e-15
        A_inv_half = chol(A_inv);
        a = A_inv \ (xstar' * ystar);
        beta0 = a + A_inv_half \ randn(K,1);
    else
        A = robust_inv(A_inv);
        A_half = robust_chol(A);
        a = A * (xstar' * ystar);
        beta0 = a + A_half * randn(K,1);
    end      
    
    
    % ASIS: compute back beta
    beta = beta_star + repmat(beta0',n,1);    
    
    
    % Horseshoe prior for diff_beta
    diff_beta = [beta(1,:)-beta0'; beta(2:n,:) - beta(1:(n-1),:)];
    diff_beta2 = diff_beta.^2;
    [tau, tau_d, tauj, tauj_d, phi, phi_d, rhoh] = KHS_update3(diff_beta, diff_beta2,...
                tau, tau_d, tauj, tauj_d, phi, phi_d, rhoh, prior_rhoh);
    tmpmat = tau * (repmat(tauj',n,1) .* phi);
    for t = 1:n
        state_var{t} = diag(tmpmat(t,:));
    end %covar matrices of state noise for simulation smoother
    
    
    % Hyperparameters of beta0
    beta02 = beta0.^2;
    [taul, taul_d, phil, phil_d] = Horseshoe_update_vector(beta02,...
        taul, taul_d, phil, phil_d);    
    
    
    % Residual variance
    yfit = sum(x .* beta,2);
    eps = y - yfit;
    if ind_SV == 1
        logz2 = log(eps.^2 + 1e-100);
        [hSV, muh, phih, sigh, sigh2_s, lambdah] = SV_update_asis(logz2, hSV, ...
            muh, phih, sigh, sigh2_s, lambdah, priorSV);    
        vary = exp(hSV);  
%         vary(isinf(vary)) = maxNum;
    else
        sig2 = 1/gamrnd(0.5*n, 2/(eps'*eps));
        vary = sig2 * ones(n,1); 
    end   
    
    
    % Compute mean and covar of p(bn|y1,...,yn) for subsequent forecasts
    if ind_forecast == 1 
        P1 = state_var{1};
        a1 = beta0;
        [bn_mean, bn_cov] = Kalman_filter_robust(y, x, ...
            vary, state_var(2:n), a1, P1);        
    end      
   

    % Collect draws
    if drawi > burnin
        i = drawi - burnin;
        draws.tau(i,:) = [tau  tau_d];
        draws.tauj(i,:) = [tauj' tauj_d'];
        draws.rhoh(i,:) = rhoh';
        for j = 1:K
            draws.phi_d{j}(i,:) = phi_d(:,j)';
            draws.phi{j}(i,:) = phi(:,j)';
            draws.beta{j}(i,:) = beta(:,j)';
        end
        
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
        
        if ind_forecast == 1
            draws.bn_mean(i,:) = bn_mean';
            draws.bn_cov{i} = bn_cov;
        end
    end
    
    
    % Display elapsed time
    if (drawi/5000) == round(drawi/5000)
        disp([num2str(drawi), ' out of ', num2str(ntotal),' draws have completed!']);
        toc;
    end    
end


