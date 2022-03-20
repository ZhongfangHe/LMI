% Consider the Horseshoe prior bt~N(0,tau * phit), tau and phi follow IB(0.5,0.5)
% Given a vector of bt, update tau and phit and their hyperparameters

function [tau, tau_d, phi, phi_d] = Horseshoe_update_vector(b2, tau, tau_d, phi, phi_d)
% Inputs:
%   b2: a n-by-1 vector of squared bt;
%   tau: a scalar of the global variance;
%   tau_d: a scalar of the hyperparameter of global variance;
%   phi: a n-by-1 vector of local variances;
%   phi_d: a n-by-1 vector of the hyperparameters of local variances;
% Outputs:
%   tau: a scalar of the updated global variance;
%   tau_d: a scalar of the updated hyperparameter of global variance;
%   phi: a n-by-1 vector of updated local variances;
%   phi_d: a n-by-1 vector of the updated hyperparameters of local variances;  

n = length(phi);

tmp = 1./phi_d + 0.5*b2/tau;
phi = 1./exprnd(tmp);    
phi_d = 1./exprnd(1+1./phi); 

tau_a = 0.5 + 0.5 * n;
tau_b = 1/tau_d + 0.5*sum(b2./phi);
tau = 1/gamrnd(tau_a, 1/tau_b);
tau_d = 1/exprnd(1+1/tau);   

 


