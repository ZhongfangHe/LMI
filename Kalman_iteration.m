% Consider the model:
% scalar: yt = xt' * betat + N(0,sig2t), 
% k-by-1: betat = betatm1 + N(0,diag(wt)),
%
% Kalman filter: p(betat|y^t) = N(bt,Bt)
% propogate from (btm1, Btm1) to (bt, Bt)

function [bt, Bt] = Kalman_iteration(btm1, Btm1, yt, xt, sig2t, wt)
% Inputs:
%   btm1: a k-by-1 vector of previous filter mean,
%   Btm1: a k-by-k matrix of previous filter covariance matrix,
%   yt: a scalar of target at t,
%   xt: a k-by-1 vector of regressor at t,
%   sig2t: a scalar of target innovation variance at t,
%   wt: a k-by-1 vector of state variances,
% Outputs:
%   bt: a k-by-1 vector of filter mean,
%   Bt: a k-by-k matrix of filter covariance matrix,

Rt = Btm1 + diag(wt);

Rxt = Rt * xt;
St = xt' * Rxt + sig2t;
Kt = Rxt/St;
ythat = xt' * btm1;

bt = btm1 + Kt * (yt - ythat);
Bt = Rt - Kt * Rxt'; 

% Check if Bt is pd
% [~,flag] = chol(Bt);
% if flag ~= 0
%     Bt_half = robust_chol(Bt);
%     Bt = Bt_half * Bt_half';
% end


