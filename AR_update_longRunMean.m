% Consider the AR(1) model:
% zt = (1-phi) * mu + phi * ztm1 + N(0,s), z1~N(mu, s/(1-phi^2))
%
% zt is a scalar, s is given,
%
% p(mu)=N(0,A_mu), p(phi)=N(a_phi,A_phi)I{|phi|<1}
% 
% generate a draw of mu and phi based on z and s

function [mu,phi,count] = AR_update_longRunMean(z, s, phi_old, A_mu, a_phi, A_phi, drawi, burnin)
% Inputs:
%   z: a n-by-1 vector of target variable,
%   s: a scalar of the innovation variance,
%   phi_old: a scalar of previous draw of phi,
%   A_mu: a scalar of prior variance of mu,
%   a_phi: a scalar of prior mean of phi,
%   A_phi: a scalar of prior variance of phi,
%   drawi: a scalar of the # of current draw, (to count acceptance)
%   burnin: a scalar of the burn-in length, (to count acceptance),
% Outputs:
%   mu: a scalar of the updated mu,
%   phi: a scalar of the updated phi,
%   count: a 0/1 scalar of proposal acceptance.


n = length(z);
yz = z(2:n);
yz1 = z(1:n-1);
z1 = z(1);

%% p(mu|z,phi)
phi = phi_old;

A_inv = 1/A_mu + ((1-phi^2) + (n-1) * (1-phi) * (1-phi)) / s;
A = 1/A_inv;
a = A * ((1-phi^2)*z1 + (1-phi)*(sum(yz) - phi * sum(yz1))) / s;
mu = a + sqrt(A) * randn;



%% p(phi|z,mu)
% proposal
A_inv = 1/A_phi + (yz1-mu)' * (yz1-mu) / s;
A = 1/A_inv;
a = A * (a_phi/A_phi + (yz1-mu)' * (yz-mu) / s);
phi_new = a + sqrt(A) * randn;


% MH step
count = 0;
if abs(phi_new)>=1 
    phi = phi_old;
else
    % log prior   
    logprior_old = -0.5 * ((phi_old - a_phi)^2) / A_phi; % truncated normal, ignore constant
    logprior_new = -0.5 * ((phi_new - a_phi)^2) / A_phi;
    
    
    % log like;
    tmp_old = 1-phi_old^2; 
    loglike1_old = 0.5 * log(tmp_old) - 0.5 * tmp_old * ((z1 - mu)^2) / s;
    loglike_old = loglike1_old - 0.5 * sum((yz - (1-phi_old)*mu - phi_old * yz1).^2) / s;
    
    tmp_new = 1-phi_new^2; 
    loglike1_new = 0.5 * log(tmp_new) - 0.5 * tmp_new * ((z1 - mu)^2) / s;
    loglike_new = loglike1_new - 0.5 * sum((yz - (1-phi_new)*mu - phi_new * yz1).^2) / s;
    
    % log proposal;
    tmp_old = phi_old - a;
    logprop_old = -0.5 * tmp_old' * A_inv * tmp_old;
    
    tmp_new = phi_new - a;
    logprop_new = -0.5 * tmp_new' * A_inv * tmp_new; 
    
    % log acceptance prob
    logprob = (logprior_new + loglike_new - logprop_new) - ...
        (logprior_old + loglike_old - logprop_old);
    if logprob >= log(rand)
        phi = phi_new;
        if drawi > burnin
            count = 1;
        end
    else
        phi = phi_old;
    end   
end



