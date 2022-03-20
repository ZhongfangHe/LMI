

function [h,mu] = SV_h_mu(logy2, phi, mu, sig2, h_old, mu_prior)

%% preparation
T = length(logy2);


%% 10-point normal mixture to approximate log(chi2(1))
pi =   [0.00609  0.04775 0.13057 0.20674  0.22715  0.18842  0.12047  0.05591  0.01575  0.00115];
mi =   [1.92677  1.34744 0.73504 0.02266 -0.85173 -1.97278 -3.46788 -5.55246 -8.68384 -14.65000];  %% means already adjusted!! %%
sigi = [0.11265  0.17788 0.26768 0.40611  0.62699  0.98583  1.57469  2.54498  4.16591  7.33342];
nm = length(pi);


%% Sample S from a 10-point distrete distribution
q = zeros(T,nm);
for j = 1:nm
    q(:,j) = pi(j) * exp(-0.5 * ((logy2 - h_old - mi(j)).^2) / sigi(j)) / sqrt(sigi(j));
end
q = q./repmat(sum(q,2),1,nm);
temprand = rand(T,1);
S = sum(repmat(temprand,1,nm) > cumsum(q,2),2)+1;
d = mi(S)';
v = 1./sigi(S)';



%% sample h using the precision-based algorithm
% y* = h + d + epsilon*, epsilon* ~ N(0,Sig_y*),
% Hphi h = alpha-tilde + zeta, zeta ~ N(0,Sig_h),
% where 
% d_t = E(epsilon*_t), Sig_y* = diag(var(epsilon*_1), ..., epsilon*_T),
% Sig_h = diag(sig2_1, sig2_2, ..., sig2_T)
Hphi = speye(T)-sparse(2:T,1:(T-1),phi*ones(1,T-1),T,T);
invSigh = sparse(1:T,1:T,1./sig2);
% d = mi(S)'; 
invSigystar = spdiags(1./sigi(S)',0,T,T);
alpha = Hphi\[mu; ((1-phi)*mu)*ones(T-1,1)];
Kh = Hphi'*invSigh*Hphi + invSigystar;
Ch = chol(Kh,'lower');
ystar = logy2-d;
hhat = Kh\(Hphi'*invSigh*Hphi*alpha + invSigystar*ystar);
h = hhat + Ch'\randn(T,1);


%% update mu
mu_mean = mu_prior(1);
mu_var = mu_prior(2);
Dmu = 1/(1/mu_var + 1/sig2(1) + (1-phi)*(1-phi)*sum(1./sig2(2:T)));
tmp = h(2:T) - phi * h(1:T-1); 
muhat = Dmu*(mu_mean / mu_var + h(1)/sig2(1) + ...
     (1-phi)*sum(tmp./sig2(2:T)));
mu = muhat + sqrt(Dmu)*randn;


%% ASIS: compute demeaned log vol
dh = h - mu;

%% ASIS: update mu in AA
yd = ystar - dh;
Dmu = 1/(1/mu_var + sum(v));
muhat = Dmu*(mu_mean / mu_var + sum(v.*yd));
mu = muhat + sqrt(Dmu)*randn;

%% ASIS: compute back h
h = dh + mu;




