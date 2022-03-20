% Consider the model:
% yt = zt * bt + ut, ut ~ N(0,Ht), t = 1,2,...,n
% btp1 = bt + vt, vt ~ N(0,Qt), t = 1,2,...,(n-1)
% b1 ~ N(a1,P1)
%
% Nee to produce a draw of b from p(b|y)
%
% yt could be univariate or multivariate
%
% In case the state equation becomes:
% btp1 = bt + N(0,Wtp1), t = 0,1,...,(n-1)
% Reset a1 = b0, P1 = W1, Qt = Wtp1 for t = 1,...,(n-1) to apply the DK smoother

function b = Simulation_Smoother_DK(y, z, h, q, a1, P1)
% Inputs:
%   y: a n-by-p matrix of targets;
%   z: regressors -> a n-by-m matrix if p = 1 or a n-by-1 cell of p-by-m matrices if p > 1;
%   h: measurement noise var/covar -> a n-by-1 vector if p = 1 or a n-by-1 cell of p-by-p matrices if p > 1;
%   q: state noise covar -> a (n-1)-by-1 cell of m-by-m matrices ;
%   a1: a m-by-1 vector of the mean of the initial state b1;
%   P1: a m-by-m matrix of the covar matrix of the initial state b1;
% Outputs:
%   b: a n-by-m matrix of simulated b ~ p(b|y);

[n,p] = size(y);
m = length(a1);

%% Forward
tmp1 = zeros(n,m); %store zt' * inv(ft) * vt for backward pass
tmp2 = zeros(n,m); %store zt' * inv(ft) * vvt for backward pass
L = cell(n,1); %store Lt for backward pass
b1 = mvnrnd(a1,P1)';
bb = zeros(n,m);
bb(1,:) = b1';
for t = 1:n
    % Preparation
    if p == 1 %univariate
        yt = y(t);
        Zt = z(t,:);
        Ht = h(t);
        Ht_half = sqrt(Ht);
    else %multivariate
        yt = y(t,:)';
        Zt = z{t};
        Ht = h{t};
        Ht_half = chol(Ht)';
    end
    if t < n
        Qt = q{t};
    end
    if t == 1
        at = a1;
        aat = a1;
        Pt = P1;
        bbt = b1;
    else
        at = atp1;
        aat = aatp1;
        Pt = Ptp1;
        bbt = bbtp1;
    end    
     
    % Simulate unconditionally
    yyt = Zt * bbt + Ht_half * randn(p,1);
    if t < n
        bbtp1 = bbt + mvnrnd(zeros(m,1),Qt)';
        bb(t+1,:) = bbtp1';
    end
    
    % Run Kalman filter    
    Ft = Zt * Pt * Zt' + Ht;
    if p == 1
        Ft_inv = 1 / Ft;
    else
        Ft_inv = Ft\eye(p);
    end
%     Kt = Pt * Zt' * Ft_inv;
    tmp = Zt' * Ft_inv;
    Kt = Pt * tmp;
    Lt = eye(m) - Kt * Zt;
    vt = yt - Zt * at;
    vvt = yyt - Zt * aat;

    t1 = tmp * vt;
    t2 = tmp * vvt;
    if t < n
        atp1 = at + Pt * t1;
        aatp1 = aat + Pt * t2;
        Ptp1 = Pt * Lt' + Qt;
    end
    
    % Store relevant values for backward pass
    tmp1(t,:) = t1';
    tmp2(t,:) = t2';
    L{t} = Lt;
end


%% Backward
r = zeros(n,m);
rr = zeros(n,m);
t = n;
while t >= 1
    if t == n
        rtm1 = tmp1(t,:)';
        rrtm1 = tmp2(t,:)';
    else
        rtm1 = tmp1(t,:)' + L{t}' * r(t+1,:)';
        rrtm1 = tmp2(t,:)' + L{t}' * rr(t+1,:)';
    end
    r(t,:) = rtm1';
    rr(t,:) = rrtm1';
    t = t - 1;
end

%% Forward iteration to compute conditional means
a = zeros(n,m);
a(1,:) = a1' + r(1,:) * P1';
aa = zeros(n,m);
aa(1,:) = a1' + rr(1,:) * P1';
for t = 1:(n-1)
    atp1 = a(t,:)' + q{t} * r(t+1,:)';
    a(t+1,:) = atp1';
    
    aatp1 = aa(t,:)' + q{t} * rr(t+1,:)';
    aa(t+1,:) = aatp1';    
end

%% Compute the draw of b
b = bb - aa + a;
% bn_mean2 = a(n,:)';


    






