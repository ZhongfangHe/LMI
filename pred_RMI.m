% Compute predictive likelihood p(y(t+1) | y(1:t), x(1:t+1))
% Use after running "RWTVP_RMI"

function [ytp1_pdf, ytp1_mean, ytp1_var, ytp1_pdf_vec, ind_valid] = pred_RMI(draws, xtp1,...
    ytp1, ind_SV, MI_scenarios)
% Inputs:
%   draws: a structure of posterior draws (from "RWTVP" or "RWTVP_DHS")
%   xtp1: a nx-by-1 vector of x(t+1)
%   ytp1: a scalar of y(t+1)
%   ind_SV: an indicator if SV model is used
%   ind_homo: an indicator if homoskedastic or heteroskedastic states
%   ind_KHS: an indicator if AR(1) for local variances
% Outputs:
%   ytp1_pdf: a scalar of the predictive likelihood
%   ytp1_mean: a ndraws-by-1 vector of predictive mean
%   ytp1_var: a ndraws-by-1 vector of predictive variance
%   ytp1_pdf_vec: a ndraws-by-1 vector of the conditional predictive likelihoods
%   ind_valid: a ndraws-by-1 indicator if each conditional pred likelihood is valid or not

%ndraws = size(draws.sig2,1);
[ndraws,n] = size(draws.beta{1});
K = size(draws.beta0,2);
% ind_scenarios = [zeros(1,K); eye(K); ones(1,K)]; %restricted scenarios of the ind vector
n2k = size(MI_scenarios,1);

ytp1_mean = zeros(ndraws,2);
ytp1_var = zeros(ndraws,2);
ytp1_pdf_vec = zeros(ndraws,2);
ind_valid = zeros(ndraws,2);
ytp1_pdf = zeros(2,1);
count = zeros(2,1);
for drawi = 1:ndraws
    bt_mean = draws.bn_mean(drawi,:)';
    bt_cov = draws.bn_cov{drawi};    
       
    btp1_mean = bt_mean; %RW 
    vi = draws.v(drawi,:)';
    vi2 = vi.^2;    
    indt = zeros(K,1);
    for j = 1:K
        indt(j) = draws.ind{j}(drawi,n);
    end
    etatp1_var = zeros(K,1);
    for j = 1:n2k
        if isequal(indt, MI_scenarios(j,:)')
            qi = draws.q(drawi);
        else
            qi = (1- draws.q(drawi)) / (n2k-1);
        end
        etatp1_var = etatp1_var + qi * (MI_scenarios(j,:)' .* vi2);
    end
    btp1_cov = bt_cov + diag(etatp1_var);     
    
    
%     prob_indtp1 = zeros(K,1);
%     for j = 1:K
%         ind_tj = draws.ind{j}(drawi,n);
%         if ind_tj == 0
%             prob_indtp1(j) = 1 - draws.q0(drawi,j); %p01
%         else
%             prob_indtp1(j) = draws.q1(drawi,j); %p11
%         end
%     end
%     vi = draws.v(drawi,:)';
%     vi2 = vi.^2;
%     btp1_cov = bt_cov + diag(prob_indtp1 .* vi2);  
    
    
    ytp1_mean(drawi,1) = xtp1' * btp1_mean; 
    if ind_SV == 0 %constant variance
        sig2tp1 = draws.sig2(drawi);
    else %SV variance
        muh = draws.SVpara(drawi,1);
        phih = draws.SVpara(drawi,2);
        sigh2 = draws.SVpara(drawi,3);
        n = length(draws.sig2(drawi,:));
        htp1 = (1-phih)*muh + phih * log(draws.sig2(drawi,n)) + sqrt(sigh2) * randn;
        sig2tp1 = exp(htp1);
    end
    ytp1_var(drawi,1) = xtp1' * btp1_cov * xtp1 + sig2tp1;
    if ytp1_var(drawi,1) <= 0
        error('negative ytp1 var');
    end
    
    
    tmp1 = (ytp1 - ytp1_mean(drawi,1))^2;
    tmp = exp(-0.5*tmp1/ytp1_var(drawi,1)') / sqrt(2*pi*ytp1_var(drawi,1)');
    if and(~isnan(tmp), ~isinf(tmp))
        ind_valid(drawi,1) = 1;
        ytp1_pdf_vec(drawi,1) = tmp;
        count(1) = count(1) + 1;
        ytp1_pdf(1) = ytp1_pdf(1) + tmp;
    end
%     
%     
%     tmp1 = (ytp1*ones(2,1) - ytp1_mean(drawi,:)').^2;
%     tmp = exp(-0.5*tmp1./ytp1_var(drawi,:)') ./ sqrt(2*pi*ytp1_var(drawi,:)');
%     for j = 1:2
%         if and(~isnan(tmp(j)), ~isinf(tmp(j)))
%             ind_valid(drawi,j) = 1;
%             ytp1_pdf_vec(drawi,j) = tmp(j);
%             count(j) = count(j) + 1;
%             ytp1_pdf(j) = ytp1_pdf(j) + tmp(j);
%         end
%     end
    
end
ytp1_pdf = ytp1_pdf ./ count;
    
        
    
    
    
    












