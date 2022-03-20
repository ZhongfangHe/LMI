% db_jt ~ N(0, tau * tauj_j * phi_jt),
% tau, tauj follow IB(0.5,0.5) with the hyperparameters tau_d, tauj_d
% log(phi_jt) = rho_j * log(phi_{j,t-1}) + eta_jt, eta_jt~N(0,1/zeta_jt),
% zeta_jt ~ Polya-Gamma(1,0)
%
% update tau,tau_d, tauj, tauj_d, phi, zeta, rho
% use ASIS for tauj


function [tau, tau_d, tauj, tauj_d, phi, phi_d, rhoh] = KHS_update3(diff_beta, diff_beta2,...
            tau, tau_d, tauj, tauj_d, phi, phi_d, rhoh, prior_rhoh)
        
[n,K] = size(diff_beta2); 
minNum = 1e-100;
maxNum = 1e100;


%% Update global variances
z2 = diff_beta2 ./ phi; %z_jt~N(0,tau*tauj_j)
tau_a = 0.5+0.5*n*K;
tau_b = 1/tau_d + 0.5*sum(sum(z2./repmat(tauj',n,1)));
tau = 1/gamrnd(tau_a, 1/tau_b);
tau_d = 1/exprnd(1+1/tau);



%% Update local variances by dynamic HS of Kowal etc. (2019)
% Use ASIS for log(tauj)
% x2 = diff_beta2 ./(tau * repmat(tauj',n,1));
% logx2 = log(x2 + 1e-100);
diff_beta(diff_beta == 0) = minNum;
% logx2 = 2*log(abs(diff_beta)) - log(tau * repmat(tauj',n,1));
logx2 = 2*log(abs(diff_beta)) - log(tau);
for j = 1:K
    %update log variance and intercept
    logx2j = logx2(:,j);
    log_taujj = log(tauj(j));
    hj = log(phi(:,j)) + log_taujj;
    rhohj = rhoh(j);
    sig2j = 1./phi_d(:,j);
    mu_prior = [0  1/tauj_d(j)]; %mean and var
    [hj, log_taujj] = SV_h_mu(logx2j, rhohj, log_taujj, sig2j, hj, mu_prior);
    taujj = exp(log_taujj);
    if taujj == 0
        taujj = minNum;
    end
    if isinf(taujj)
        taujj = maxNum;
    end
    tauj(j) = taujj;
    
    
    %update rhohj
    rhohj = SV_slope(hj, log_taujj, sig2j, prior_rhoh, rhohj);
    rhoh(j) = rhohj;
    
    %update phi_d
    Hj = speye(n) - sparse(2:n,1:(n-1),rhohj*ones(1,n-1),n,n);
    etaj = Hj * hj - [log_taujj; (1-rhohj)*log_taujj*ones(n-1,1)];
    phi_d(:,j) = pgdraw(etaj);
    
    %update tauj_d
    tauj_d(j) = pgdraw(log_taujj);
    
    %update phi
    phi(:,j) = exp(hj - log_taujj);
end
phi(isinf(phi)) = maxNum;
phi(phi == 0) = minNum;
phi_d(phi_d == 0) = minNum;%avoid numerical errors

% tauj_d = 1./exprnd(1+1./tauj);





