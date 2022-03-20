% Given a vector of indicators, count the number of identical transitions:
% restricted sceanrios of indicators

function tn = RMI_transition_numbers(ind)
% Inputs:
%   ind: a n-by-k matrix of indicators, each column being one variable,
% Outputs:
%   tn: a k-by-4 matrix of counts for (00,01,10,11)

n = size(ind,1);
tn = 0;
for t = 2:n
    indt = ind(t,:)';
    indtm1 = ind(t-1,:)';
    if isequal(indt,indtm1)
        tn = tn + 1;
    end
end

% [n,K] = size(ind); 
% tn = zeros(K,4);
% for j = 1:K
%     indj = ind(:,j);
%     indj1 = indj(1:n-1);
%     indj2 = indj(2:n);
%     
%     tn(j,1) = sum(and(indj1 == 0, indj2 == 0));
%     tn(j,2) = sum(and(indj1 == 0, indj2 == 1));
%     tn(j,3) = sum(and(indj1 == 1, indj2 == 0));
%     tn(j,4) = sum(and(indj1 == 1, indj2 == 1));   
% end


