% Compute the effective sample size portion by the Grey method
% Greyer, "practical markov chain monte carlo", Statistical Science, 1992, V7, No. 4, page 473-511
% Stan user manual: https://mc-stan.org/docs/2_21/reference-manual/effective-sample-size-section.html
%
% univariate ess

function ess = effective_sample_size_portion(x)

[n,K] = size(x);
ess = zeros(K,1);
for Kj = 1:K
    z = x(:,Kj);
    k = n - 1;
    p = zeros(n,1);
    q = zeros(n,1);
    for j = 0:k
        rho_2j = corr(z(1:n-2*j), z((1+2*j):n));
        rho_2jp1 = corr(z(1:n-2*j-1), z((1+2*j+1):n)); 
        p(j+1) = rho_2j + rho_2jp1;
        q(j+1) = min(p(1:j+1));

        if p(j+1) <= 0
            m = j - 1;
            break;
        end   
    end
    % vall = -1 + 2 * sum(p(1:m+1)); %initial positive sequence estimator
    vall = -1 + 2 * sum(q(1:m+1)); %initial monotone sequence estimator
    ess(Kj) = 1/vall;
end




