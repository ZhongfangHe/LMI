% Compute predictive likelihood p(y(t+1) | y(1:t), x(1:t+1))
% Use after running "RWTVP_HS" or "RWTVP_KHS"

function [ytp1_pdf, ytp1_mean, ytp1_var, ytp1_pdf_vec, ind_valid] = pred_TVP_HS(draws, ...
    xtp1, ytp1, ind_SV, ind_KHS)
% Inputs:
%   draws: a structure of posterior draws (from "RWTVP" or "RWTVP_DHS")
%   xtp1: a nx-by-1 vector of x(t+1)
%   ytp1: a scalar of y(t+1)
%   ind_SV: an indicator if SV model is used
%   ind_KHS: an indicator if AR(1) for local variances
% Outputs:
%   ytp1_pdf: a scalar of the predictive likelihood
%   ytp1_mean: a ndraws-by-1 vector of predictive mean
%   ytp1_var: a ndraws-by-1 vector of predictive variance
%   ytp1_pdf_vec: a ndraws-by-1 vector of the conditional predictive likelihoods
%   ind_valid: a ndraws-by-1 indicator if each conditional pred likelihood is valid or not

ndraws = size(draws.sig2,1);

ytp1_mean = zeros(ndraws,2);
ytp1_var = zeros(ndraws,2);
ytp1_pdf_vec = zeros(ndraws,2);
ind_valid = zeros(ndraws,2);
ytp1_pdf = zeros(2,1);
count = zeros(2,1);
for drawi = 1:ndraws
    bt_mean = draws.bn_mean(drawi,:)';
    bt_cov = draws.bn_cov{drawi};
%     bt_smean = draws.bn_smean(drawi,:)';
%     bt_scov = draws.bn_scov{drawi};    
      
    
    btp1_mean = bt_mean; %RW 
    K = length(xtp1);
    vi2 = draws.tau(drawi,1)*draws.tauj(drawi,1:K)';
    if ind_KHS == 1
        phi_d = pgdraw(zeros(K,1));
        eta = (1./sqrt(phi_d)) .* randn(K,1);
        phi = zeros(K,1);
        for j = 1:K
            n = length(draws.phi{j}(drawi,:));
            htp1 = draws.rhoh(j) * log(draws.phi{j}(drawi,n)) + eta(j);
            phi(j) = exp(htp1);
        end
    else
        phi_d = 1./gamrnd(0.5,1,K,1);
        phi = 1./gamrnd(0.5*ones(K,1),phi_d);
    end
    wtp1 = diag(phi.*vi2);
    btp1_cov = bt_cov + wtp1;
    
    
%     btp1_smean = bt_smean; %RW
%     vi = draws.v_sparse(drawi,:)';
%     vi2 = vi.^2;
%     wtp1 = diag(phi.*vi2);
%     btp1_scov = bt_scov + wtp1;
    
    
    ytp1_mean(drawi,1) = xtp1' * btp1_mean;
%     ytp1_mean(drawi,2) = xtp1' * btp1_smean;
    
    
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
%     ytp1_var(drawi,2) = xtp1' * btp1_scov * xtp1 + sig2tp1;
    
    
    tmp1 = (ytp1 - ytp1_mean(drawi,1))^2;
    tmp = exp(-0.5*tmp1/ytp1_var(drawi,1)) / sqrt(2*pi*ytp1_var(drawi,1));
    for j = 1:1
        if and(~isnan(tmp(j)), ~isinf(tmp(j)))
            ind_valid(drawi,j) = 1;
            ytp1_pdf_vec(drawi,j) = tmp(j);
            count(j) = count(j) + 1;
            ytp1_pdf(j) = ytp1_pdf(j) + tmp(j);
        end
    end
end

% for drawi = 1:ndraws
%     bt_mean = zeros(K,1);
% %     bt_smean = zeros(K,1);
%     for j = 1:K
%         bt_mean(j) = draws.beta{j}(drawi,n);
% %         bt_smean(j) = draws.beta_sparse{j}(drawi,n);
%     end       
%       
%     
%     btp1_mean = bt_mean; %RW 
%     K = length(xtp1);
%     vi2 = draws.tau(drawi,1)*draws.tauj(drawi,1:K)';
%     if ind_KHS == 1
%         phi_d = pgdraw(zeros(K,1));
%         eta = (1./sqrt(phi_d)) .* randn(K,1);
%         phi = zeros(K,1);
%         for j = 1:K
%             n = length(draws.phi{j}(drawi,:));
%             htp1 = draws.rhoh(j) * log(draws.phi{j}(drawi,n)) + eta(j);
%             phi(j) = exp(htp1);
%         end
%     else
%         phi_d = 1./gamrnd(0.5,1,K,1);
%         phi = 1./gamrnd(0.5*ones(K,1),phi_d);
%     end
%     wtp1 = diag(phi.*vi2);
%     btp1_cov = wtp1;
%     
%     
% %     btp1_smean = bt_smean; %RW
% %     vi = draws.v_sparse(drawi,:)';
% %     vi2 = vi.^2;
% %     wtp1 = diag(phi.*vi2);
% %     btp1_scov = bt_scov + wtp1;
%     
%     
%     ytp1_mean(drawi,1) = xtp1' * btp1_mean;
% %     ytp1_mean(drawi,2) = xtp1' * btp1_smean;
%     
%     
%     if ind_SV == 0 %constant variance
%         sig2tp1 = draws.sig2(drawi);
%     else %SV variance
%         muh = draws.SVpara(drawi,1);
%         phih = draws.SVpara(drawi,2);
%         sigh2 = draws.SVpara(drawi,3);
%         n = length(draws.sig2(drawi,:));
%         htp1 = (1-phih)*muh + phih * log(draws.sig2(drawi,n)) + sqrt(sigh2) * randn;
%         sig2tp1 = exp(htp1);
%     end
%     ytp1_var(drawi,1) = xtp1' * btp1_cov * xtp1 + sig2tp1;
% %     ytp1_var(drawi,2) = xtp1' * btp1_scov * xtp1 + sig2tp1;
%     
%     
%     tmp1 = (ytp1 - ytp1_mean(drawi,1))^2;
%     tmp = exp(-0.5*tmp1/ytp1_var(drawi,1)) / sqrt(2*pi*ytp1_var(drawi,1));
%     for j = 1:1
%         if and(~isnan(tmp(j)), ~isinf(tmp(j)))
%             ind_valid(drawi,j) = 1;
%             ytp1_pdf_vec(drawi,j) = tmp(j);
%             count(j) = count(j) + 1;
%             ytp1_pdf(j) = ytp1_pdf(j) + tmp(j);
%         end
%     end
% end


ytp1_pdf(1) = ytp1_pdf(1) ./ count(1);
    
        
    
    
    
    












