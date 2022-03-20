% Consider the model:
% yt = zt * bt + N(0,ht);
% btp1 = bt + N(0,qt);
% b1 ~ N(a1,P1);
%
% Compute the mean and covar matrix of the filtered last state p(bn|y)
%
% yt could be univariate or multivariate
%
% In case the state equation becomes:
% btp1 = bt + N(0,wtp1), t = 0,1,...,(n-1)
% Reset a1 = b0, P1 = w1, qt = wtp1 for t = 1,...,(n-1) to apply the filter

function [bn_mean, bn_cov] = Kalman_filter_robust(y, z, h, q, a1, P1)
% Inputs:
%   y: a n-by-p matrix of targets;
%   z: regressors -> a n-by-m matrix if p = 1 or a n-by-1 cell of p-by-m matrices if p > 1;
%   h: measurement noise var/covar -> a n-by-1 vector if p = 1 or a n-by-1 cell of p-by-p matrices if p > 1;
%   q: state noise covar -> a (n-1)-by-1 cell of m-by-m matrices ;
%   P1_inv_times_a1: a m-by-1 vector of the covector of the initial state b1;
%   P1_inv: a m-by-m matrix of the precision matrix of the initial state b1;
% Outputs:
%   bn_mean: a m-by-1 vector of E(bn|y);
%   bn_cov: a m-by-m matrix of V(bn|y);

[n,p] = size(y);
m = length(a1);

for t = 1:n
    if p == 1 %univariate
        zt = z(t,:);
        yt = y(t);
        ht = h(t);
    else % multivariate
        zt = z{t};
        yt = y(t,:)';
        ht = h{t};
    end
    if t == 1
        rt = P1;
        mtm1 = a1;
    else
        rt = ct + q{t-1};
        mtm1 = mt;
    end 

    ythat = zt * mtm1;
    st = zt * rt * zt' + ht;
    st_inv = st\eye(p);
    kt = rt * zt' * st_inv;
    mt = mtm1 + kt * (yt - ythat);
    ct = (eye(m) - kt * zt) * rt;
end
bn_cov = ct;
bn_mean = mt;


