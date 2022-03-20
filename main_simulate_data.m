% Simulate data

clear;
rng(123456);


%% Simulate
nrep = 20; %number of simulated series
n = 300;
K = 5;
for j = 1:nrep
    % generate x
    x = randn(n,K);

    % generate beta
    btrue = zeros(n,K);
    if K > 1
        btrue(:,1) = cumsum(0.1*randn(n,1)); %DGP1: RW
        
        bp = [round(n/3)  round(2*n/3)];
        btrue(bp(1):bp(2),2) = 2;
%         btrue(bp(2)+1:n,2) = -1; %DGP2: chang point
        
        btrue(bp(1):bp(2),3) = cumsum(0.1 * randn(bp(2)-bp(1)+1,1));
        btrue(bp(2)+1:n,3) = 1; %DGP 3: mixture innovation
        
        btrue(:,4) = ones(n,1); %DGP4: ones
        
    else
        bp = [round(n/3)  round(2*n/3)];
        btrue(bp(1):bp(2)) = 1;
        btrue(bp(2)+1:n) = -1;
    end
    
    % determine noise level
    noise_level = {'L','M','S'};
    rsquare = [0.2  0.5  0.9];
    nr = length(rsquare);
    s = 1./rsquare - 1;
    yfit = sum(btrue .* x, 2);
    sig2true = s * var(yfit);
    sigtrue = sqrt(sig2true);

    % generate y with different noise level, write output
    for rj = 1:nr
        y = yfit + sigtrue(rj) * randn(n,1);

        write_file = ['SimData_', noise_level{rj}, '.xlsx'];
        write_sheet = ['D',num2str(j)];
        title = {['y(sig=',num2str(sigtrue(rj)) ,')'],'x1','x2','x3','x4','x5',...
            'RW','CP','Mix','One','Zero'};
        writecell(title, write_file, 'Sheet', write_sheet, 'Range', 'A1');
        writematrix([y x btrue], write_file, 'Sheet', write_sheet, 'Range', 'A2');
    end
end

