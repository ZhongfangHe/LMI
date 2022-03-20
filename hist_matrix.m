

function hist_matrix(x)

K = size(x,2);
m = K/2;
if m == round(m)
    for j = 1:K
        subplot(m,2,j);
        histogram(x(:,j),100);
        title(['para ',num2str(j)]);
    end
else
    m = (K+1)/2;
    for j = 1:K
        subplot(m,2,j);
        histogram(x(:,j),100);
        title(['para ',num2str(j)]);
    end
end
