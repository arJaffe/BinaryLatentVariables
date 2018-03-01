function [X,lambda] = compute_candidate_set(Z,d,sigma,denoising_method)
    
    % [X_NCM,lambda,M,V,D] = ...
    %        compute_candidate_set(Z,d,sigma,denoising_method,W,H_bin)
    %
    % Description: 
    % Given n observed samples of a binary latent variable model,
    % computes a candidate set for the columns of the pseudo inverse
    % of the weight matrix W. 
    % Described in "Learning Binary Latent Variable Models:
    % A Tensor Eigenpair Approach" (2018)
    %
    % Input: Z - m x n matrix of (noisy) n realizations of the observed
    %            data
    %        d - number of hidden variables
    %        sigma - noise level
    %        denoising method - which method to use to denoise the low
    %                           order moments:
    %                           1. Removing sigma^2 - Gaussian noise
    %                           2. Matrix completion - For Binomial data
    %        
    % Output:
    %        X - A candidate set for the columns of the pseudo inverse
    %            of the weight matrix W
    %        lambda - eigenvalues corresponding to candidate eigenvectors
    %
    % Written by Ariel Jaffe, Roi Weiss, Shai Carmi, Yuval Kluger
    % and Boaz Nadler, 2018
    
    
    % compute moments
    [m,n] = size(Z);
    R_z_s = (Z*Z')/n;
    T_z_s = compute_joint_covariance_tensor(Z,1);
    
    if denoising_method==1
        % perform matrix completion
        R_z_s = complete_matrix_diagonal_elements(R_z_s,d,10^-12,1000);
        [V,D] = eigs(R_z_s,d);
        [~,sort_idx] = sort(diag(D),'descend');
        V = V(:,sort_idx(1:d));
        D = D(sort_idx(1:d),sort_idx(1:d));
        
        % estimate whitened tensor
        K = (D^-0.5)*V';
        K_inv = (D^0.5)*V';
        T_w_s = estimate_whitened_tensor(T_z_s,K_inv);        
    end
    if denoising_method==2
        R_z_s = R_z_s-(sigma^2)*eye(m);
        [V,D] = eig(R_z_s);
        [~,sort_idx] = sort(diag(D),'descend');
        V = V(:,sort_idx(1:d));
        D = D(sort_idx(1:d),sort_idx(1:d));
        
        % estimate whitened tensor
        K = (D^-0.5)*V';
        mu = mean(Z,2);
        for i = 1:m
            e_i = [zeros(i-1,1) ;1 ;zeros(m-i,1)];
            T_z_s = T_z_s - (sigma^2)*(tensor_outer_prod([mu e_i e_i])+...
                tensor_outer_prod([e_i e_i mu])+tensor_outer_prod([ e_i mu e_i]));
        end
        T_w_s = symmetric_tensor_mode_product(T_z_s,K',1:3);
    end
    
    % get all eigenpairs
    [X_NCM,lambda] = get_all_eigenvectors_parallel_filter(T_w_s,0.01);
    
    % Compute candidate set
    X = (diag(1./lambda)*X_NCM'*K)';
    
end