% Estimate TVP models for simulated data


clear;
dbstop if warning;
dbstop if error;
rng(12345);


%% Read data
mdl = {'MI', 'LMIAR'};
noise_level = {'S','M','L'};
n_sheet = 20;

tic;
for mdlj = 1:2
    disp(mdl{mdlj});
    
    for ni = 1:1
    disp(noise_level{ni});

        for sheet_i = 1:1
            disp(['work sheet ', num2str(sheet_i)]);
            
            read_file = ['SimData_', noise_level{ni}, '.xlsx'];
            read_sheet = ['D',num2str(sheet_i)];
            data = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'A2:K301');
            y = data(:,1);
            x = data(:,2:6);
            btrue = data(:,7:11);


            % Set up
            ndraws = 5000*2;
            burnin = 2000;

            ind_SV = 0;
            ind_forecast = 0;
            ind_sparse = 0;
            switch mdlj
                case 1
                    draws = RWTVP_MI(y, x, burnin, ndraws, ind_SV, ind_sparse, ind_forecast);
                case 2
                    draws = RWTVP_LMI_AR(y, x, burnin, ndraws, ind_SV, ind_sparse, ind_forecast);
                otherwise
                    error('Wrong model');
            end
            disp([mdl{mdlj},', ',noise_level{ni},', Sheet ', num2str(sheet_i), ' is completed!']);
            toc;
            save(['Est_',mdl{mdlj},'_', noise_level{ni},num2str(sheet_i), '.mat'], 'draws');

            
            % RMSE
            rmse = compute_rmse_tvp_beta(draws.beta, btrue);
            write_col = {'A','B'};
            K = size(x,2);
            for j = 1:K
                write_sheet = ['Para',num2str(j),'_',num2str(sheet_i)];
                writematrix(rmse(:,j),read_file,'Sheet',write_sheet,'Range',[write_col{mdlj}, '2']);
            end
            
        end %loop of sheet
    end %loop of noise level
end %loop of model













    





