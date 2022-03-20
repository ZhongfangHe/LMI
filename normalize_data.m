% normalize data

function x = normalize_data(y)
% Inputs:
%   y: a n-by-k matrix of original data
% Outputs:
%   x: a n-by-k matrix of normalized data x = (y-mean(y))/std(y)

nobs = size(y,1);
ymean = mean(y);
ystd = std(y);
x = (y - kron(ones(nobs,1),ymean)) ./ kron(ones(nobs,1),ystd); 