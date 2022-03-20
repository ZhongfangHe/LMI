% Estimate TVP models for equity premium


clear;
dbstop if warning;
dbstop if error;
rng(123456789);


mdl = {'LMI','RMI','DHS'};
for mdlj = 1:1 
    disp(mdl{mdlj});



    %% Read data to get y and x
    read_file = 'Equity_Qtrly_Github.xlsx';
    read_sheet = 'Data';
    data = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'B2:N297');
    [ng,nr] = size(data);
    equity = data(:,1);
    reg = data(:,2:nr);
    y = equity(2:ng);
    x = [ones(ng-1,1) equity(1:(ng-1)) reg(1:(ng-1),:)]; %full
    
    
    %% Set the size of estimation/prediction sample
    [n,nx] = size(x);
    npred = 40; %number of predictions >= 0
    nest = n - npred; %number of estimation data
    disp(['nobs = ', num2str(n), ', nx = ', num2str(nx)]);
    disp(['nest = ', num2str(nest), ', npred = ', num2str(npred)]); 
    
    
    %% Configuration
    ind_SV = 1; %if SV for measurement noise variance
    ind_sparse = 0; %if sparsifying is needed
    disp(['SV = ', num2str(ind_SV),', sparse = ', num2str(ind_sparse)]);    



    %% MCMC
    ndraws = 5000*2;
    burnin = 2000;
    disp(['burnin = ', num2str(burnin), ', ndraws = ', num2str(ndraws)]);

    tic;
    if npred == 0 %in-sample estimation only 
        ind_forecast = 0;   
        yest = y;
        xest = [ones(n,1)  normalize_data(x(:,2:nx))];
        switch mdlj
            case 1 %LMIAR
                draws = RWTVP_LMI_AR(yest, xest, burnin, ndraws, ind_SV, ind_sparse, ind_forecast);
            case 2 %RMI
                MI_scenarios = [zeros(1,nx); [1 zeros(1,nx-1)]; ones(1,nx)];
                draws = RWTVP_RMI(yest, xest, burnin, ndraws, ind_SV, ind_sparse, ind_forecast, MI_scenarios);  
            case 3 %DHS
                draws = RWTVP_KHS3_scale(yest, xest, burnin, ndraws, ind_SV, ind_forecast);
            otherwise
                error('Wrong model');
        end
        disp([mdl{mdlj},' is completed!']);
        save(['Est_',mdl{mdlj},'_Equity', '.mat'], 'draws');
        toc;
    else
        ind_forecast = 1;
        logpredlike = zeros(npred,1);
        valid_percent = zeros(npred,1); %count conditional likelihoods that are not NaN or Inf     
        for predi = 1:npred
            % process data
            nesti = nest + predi - 1;
            yi = y(1:nesti,:);
            xi = x(1:nesti,:); %rescaling x is possible 
            
            yest = yi;
            xest = [ones(nesti,1)  normalize_data(xi(:,2:nx))];

            % estimate the model
            switch mdlj
                case 1 %LMI
                    draws = RWTVP_LMI_AR(yest, xest, burnin, ndraws, ind_SV, ind_sparse, ind_forecast);       
                case 2 %RMI
                    MI_scenarios = [zeros(1,nx); [1 zeros(1,nx-1)]; ones(1,nx)];
                    draws = RWTVP_RMI(yest, xest, burnin, ndraws, ind_SV, ind_sparse, ind_forecast, MI_scenarios);  
                case 3 %DHS
                    draws = RWTVP_KHS3_scale(yest, xest, burnin, ndraws, ind_SV, ind_forecast);
                otherwise
                    error('Wrong model');
            end
            
            % prediction
            xtp1 = x(nesti+1,:)'; 
            ytp1 = y(nesti+1);
            
            xmean = mean(xi(:,2:nx))';
            xstd = std(xi(:,2:nx))';
            xtp1_normalized = [1; (xtp1(2:nx) - xmean)./xstd];
            
            if mdlj == 2 %RMI
                [ytp1_pdf, ytp1_mean, ytp1_var, ytp1_pdf_vec, ind_valid] = pred_RMI(draws,...
                    xtp1_normalized, ytp1, ind_SV, MI_scenarios);
            elseif mdlj==3 %DHS 
                ind_KHS = 1;
                [ytp1_pdf, ytp1_mean, ytp1_var, ytp1_pdf_vec, ind_valid] = pred_TVP_HS(draws,...
                    xtp1_normalized, ytp1, ind_SV, ind_KHS);
            else %LMI
                [ytp1_pdf, ytp1_mean, ytp1_var, ytp1_pdf_vec, ind_valid] = pred_LMI_AR(draws,...
                    xtp1_normalized, ytp1, ind_SV);               
            end

            % store log likelihoods and prediction error
            logpredlike(predi) = log(ytp1_pdf(1))';
            valid_percent(predi) = sum(ind_valid(:,1))/ndraws;
           
            disp(['Prediction ', num2str(predi), ' out of ', num2str(npred), ' is finished!']);
            toc;   
            disp(' ');              
        end %end of prediction loop
        if predi == npred
            save(['Pred_',mdl{mdlj},'_Equity.mat'],'draws','ytp1_pdf_vec','valid_percent');
        end
        
        write_column = {'C','D','E'};
        writematrix(logpredlike(:,1), read_file, 'Sheet', 'LPL',...
            'Range', [write_column{mdlj},'2']);      
    end %end of prediction choice
end %end of model loop













    





