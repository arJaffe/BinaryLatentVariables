
% The following script implements a spectral method for learning
% a binary latent variable model. The method is described in 
% the paper:
% Learning Binary Latent Variable Models: A Tensor Eigenpair Approach
% Authors: 
% Ariel Jaffe, Roi Weiss, Shai Carmi, Yuval Kluger and Boaz Nadler (2018)
% The output of the script is Figure 2 in the paper.
% 
% Acknowledgment: In the following code we use an implementation of
% the Hungarian method, by Niclas Borlin, Department of Computing Science, 
% Ume? University, Sweden. 

clear;
close all;
clc;
addpath('./max_matching');
addpath('./tensor_util');
addpath('./NCM_functions');


% parameters
d = 6;                    % number of hidden nodes
m = 30;                   % number of visible nodes
n = 100000;               % number of independent samples
num_runs = 20;
denoising_method = 2;     % denoising method
                          % 1 - matrix completion,
                          % 2 - remove sigma,            

% go over noise level sigma
sigma_vec = 0.1:0.1:0.8;

% results for ALS, spectral and oracle
res_als = zeros(num_runs,length(sigma_vec));
res_spectral_ks = zeros(num_runs,length(sigma_vec));
res_spectral_wls = zeros(num_runs,length(sigma_vec));
res_oracle = zeros(num_runs,length(sigma_vec));

% generate H parameters and weight matrix W
    [W,mu,R_h] = generate_model_parameters(d,m);

%% run
for j = 1:num_runs   
    
    % generate random hidden and visible samples
    [H_bin,Z_c] = generate_hidden_visible_samples(W,mu,R_h,n);
    
    % sample random Gaussian noise
    N = randn(size(Z_c));
    
    % Go over different noise levels
    for sigma_idx = 1:length(sigma_vec)
        fprintf('sigma idx %d - iteration %d\n',sigma_idx,j);
        sigma = sigma_vec(sigma_idx);
        Z = Z_c+sigma*N;
        
        % get error of oracle with full knowledge of H_bin
        W_oracle = ((H_bin*H_bin')^-1)*H_bin*Z';
        res_oracle(j,sigma_idx) = check_matrix_diff(W,W_oracle);
                
        % ALS iterations, random initialization        
        [H_als,W_als,~] = ALS_iterations(Z,d,10^-2);
        res_als(j,sigma_idx) = check_matrix_diff(W,W_als);
                
        % stage A of spectral method - compute candidate set
        [X,lambda] =  compute_candidate_set(Z,d,sigma,denoising_method);
        
        % Filter via KS test        
        [ev_idx_KS,score_KS] = ...
            filter_eigenvectors_gaussian_model(X,lambda,d,Z,sigma,2);
        
        % get results for Kolmogorov-Smirnov test
        W_spec_KS = pinv(X(:,ev_idx_KS));
        [res_spectral_ks(j,sigma_idx),~] = check_matrix_diff(W,W_spec_KS);        
        
        % add weighted least squares
        W_spec_wls = single_step_wls(Z,W_spec_KS,sigma,d);
        [res_spectral_wls(j,sigma_idx),~] = check_matrix_diff(W,W_spec_wls);
        
    end    
end

close all;
fig = figure;
FS = 16;
j = num_runs;
d = 6;
m = 30;
loglog(sigma_vec,mean(res_oracle(1:j,:),1)/(m*d),'-.sk','linewidth',2,...
    'markersize',12,'markerfacecolor', 'k');hold on;grid on;
loglog(sigma_vec,mean(res_als(1:j,:),1)/(m*d),'--x','Color', [0.5,0.5,0.5],'linewidth',2,...
    'markersize',12,'markerfacecolor', [0.5,0.5,0.5]);
loglog(sigma_vec,mean(res_spectral_ks(1:j,:),1)/(m*d),':hb','linewidth',2,...
    'markersize',12,'markerfacecolor', 'b');
loglog(sigma_vec,mean(res_spectral_wls(1:j,:),1)/(m*d),'-dm','linewidth',2,...
    'markersize',12,'markerfacecolor','m');
legend({'Oracle','ALS','Spectral','Spectral+WLS'},...
    'interpreter','latex','fontsize',FS,'location','northwest');
fig.CurrentAxes.FontSize = FS;
xticks([0.1 0.2 0.4 0.8]);
xlabel('Noise level $\sigma$','interpreter','latex','fontsize',FS);
ylabel('Error $\|W-\hat W\|_F^2\big/ (m\cdot d)$','interpreter','latex','fontsize',FS);

