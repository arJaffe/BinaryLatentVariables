clear;
close all;
clc;

addpath('../tensor_util');
addpath('../homotopy/TenEig-2.0/');
addpath('../NCM_functions');
addpath('../matrix_tensor_completion');
addpath('../max_matching');
addpath(genpath('../code_mfbc'));

%% initialize
d = 6;                    % number of hidden nodes
m = 30;                   % number of visible nodes
n = 100000;                % number of independent samples
num_runs = 10;
em_flag = 0;              % flags for running different methods
slaw_flag = 1;
denoising_method = 2;     % denoising method
                          % 1 - matrix completion,2 - remove sigma,3- Exact moments 

% go over noise level sigma
sigma_vec = [0.01 0.1:0.1:0.8];

% error analysis for binary rounding and
err_ctr_br = zeros(num_runs,length(sigma_vec));
err_ctr_ks = zeros(num_runs,length(sigma_vec));

res_em = zeros(num_runs,length(sigma_vec));
res_spectral_br = zeros(num_runs,length(sigma_vec));
res_spectral_ks = zeros(num_runs,length(sigma_vec));
res_spectral_ls = zeros(num_runs,length(sigma_vec));
res_spectral_em = zeros(num_runs,length(sigma_vec));
res_slaw = zeros(num_runs,length(sigma_vec));
res_slaw_em = zeros(num_runs,length(sigma_vec));

H_fail = [];
W_fail = [];

% generate H params
[W,mu,R_h] = generate_model_parameters(d,m);

%% run
for j = 1:num_runs
    
    % generate random hidden and visible samples
    [H_bin,Z_c] = generate_hidden_visible_samples(W,mu,R_h,n);
    
    % sample random Gaussian noise
    N = randn(size(Z_c));
    
    for sigma_idx = 1:length(sigma_vec)
        fprintf('sigma idx %d - iteration %d\n',sigma_idx,j);
        sigma = sigma_vec(sigma_idx);
        Z = Z_c+sigma*N;
        
        % ALS iterations, random initialization
        if em_flag
            [H_em,W_em,~] = olc_em_iterations(Z,d,10^-2);
            res_em(j,sigma_idx) = check_matrix_diff(W,W_em,0);
        end
        
        % Slawski - 50 repetitions
        if slaw_flag
            opt_linear = opt_Integerfac_findvert('nonnegative', false,...
                'affine', false,'verbose',false,'aggr','bestsingle');
            [H_slaw, W_slaw, status] = Integerfac_findvert_cpp(Z', d, [0 1], opt_linear);
            
            %Slawski + ALS iterations
            [H_slaw_em,W_slaw_em,~] = olc_em_iterations(Z,d,10^-10,H_slaw');
            [res_slaw(j,sigma_idx),W_slaw] = check_matrix_diff(W,W_slaw,0);
            res_slaw_em(j,sigma_idx) = check_matrix_diff(W,W_slaw_em,0);
        end
        
        % stage A - compute candidate set
        [X_NCM,lambda,M,V,D] = ...
            compute_candidate_set(Z,d,sigma,denoising_method,W,H_bin);
        
        % Filter via normalized binary rounding
        [ev_idx_BR,score_BR] = filter_eigenvectors_gaussian_model(X_NCM,lambda,d,Z,M,sigma,2);
        
        % Filter via KS test
        [ev_idx_KS,score_KS] = filter_eigenvectors_gaussian_model(X_NCM,lambda,d,Z,M,sigma,3);
        
        % error analysis: (2- fail in candidate set, 1 - fail in filtering)        
        X_th = W*M';
        X_th_inv = X_th^-1;
        X_th_inv_n = X_th_inv./repmat(sqrt(sum( (X_th_inv).^2,1)),d,1);
        err_ctr_br(j,sigma_idx) = errors_analysis(X_NCM,ev_idx_BR,X_th_inv_n);
        err_ctr_ks(j,sigma_idx) = errors_analysis(X_NCM,ev_idx_KS,X_th_inv_n);
        if err_ctr_br(j,sigma_idx)==2
            % save H,W in case of type 2 failure
            H_fail = cat(3,H_fail,H_bin);
            W_fail = cat(3,W_fail,W);
        end
                
        % get results for binary rounding
        W_spec_BR = pinv(diag(1./lambda(ev_idx_BR))*X_NCM(:,ev_idx_BR)'*M)';
        [res_spectral_br(j,sigma_idx),~] = check_matrix_diff(W,W_spec_BR,0);
        
        % get results for kolmogorov-smirnoff test
        W_spec_KS = pinv(diag(1./lambda(ev_idx_KS))*X_NCM(:,ev_idx_KS)'*M)';
        [res_spectral_ks(j,sigma_idx),~] = check_matrix_diff(W,W_spec_KS,0);
        
        % add one step of least square
        H_spec = round(diag(1./lambda(ev_idx_KS))*X_NCM(:,ev_idx_KS)'*M*Z);
        H_spec(H_spec>1)=1;H_spec(H_spec<0)=0;
        W_spec_ls = ((H_spec*H_spec')^-1)*H_spec*Z';
        [res_spectral_ls(j,sigma_idx),~] = check_matrix_diff(W,W_spec_ls,0);
        
        % add ALS iterations
        [H_spec_em,W_spec_em,~] = olc_em_iterations(Z,d,10^-10,H_spec);
        res_spectral_em(j,sigma_idx) = check_matrix_diff(W,W_spec_em,0);
        
    end
end


close all;
fig = figure;
FS = 16;
loglog(sigma_vec,median(res_slaw,1),'--or','linewidth',2,...
    'markersize',12,'markerfacecolor', 'r');hold on;grid on;
loglog(sigma_vec,median(res_spectral_ls,1),'--og','linewidth',2,...
    'markersize',12,'markerfacecolor', 'g');hold on;grid on;
loglog(sigma_vec,median(res_spectral_ks,1),'-om','linewidth',2,...
    'markersize',12,'markerfacecolor','m');
legend({'SHL','Spectral-KS+LS','Spectral-KS'},...
    'interpreter','latex','fontsize',FS,'location','southwest');
fig.CurrentAxes.FontSize = FS;
xlabel('$\sigma$','interpreter','latex','fontsize',FS);
ylabel('$\|W-\hat W\|_F^2$','interpreter','latex','fontsize',FS);

save(sprintf('./results/W_est_sigscan_f_%d_m_%d_n_%d_b.mat',d,m,n),'res_spectral_br',...
    'res_spectral_ks','err_ctr_br','err_ctr_ks','res_slaw','res_em',...
    'res_spectral_ls','res_spectral_em','W_fail','H_fail');

