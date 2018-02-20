function [X_NCM,lambda,M,V,D] = compute_candidate_set(Z,d,sigma,denoising_method,W,H_bin)

% compute moments
[m,n] = size(Z);
R_z_s = (Z*Z')/n;
T_z_s = compute_joint_covariance_tensor(Z,1);

if denoising_method==1
    % perform matrix completion
    R_z_s = complete_matrix_diagonal_elements(R_z_s,d,10^-12);
    [V,D] = eigs(R_z_s,d);
    [~,sort_idx] = sort(diag(D),'descend');
    V = V(:,sort_idx(1:d));
    D = D(sort_idx(1:d),sort_idx(1:d));
        
    % estimate whitened tensor
    M = (D^-0.5)*V';
    M_inv = (D^0.5)*V';
    T_w_s = estimate_whitened_tensor(T_z_s,M_inv);
    %T_w_s_b = estimate_whitened_tensor_b(T_z_s,M_inv);
end
if denoising_method==2
    R_z_s = R_z_s-(sigma^2)*eye(m);
    [V,D] = eig(R_z_s);
    [~,sort_idx] = sort(diag(D),'descend');
    V = V(:,sort_idx(1:d));
    D = D(sort_idx(1:d),sort_idx(1:d));
    
    % estimate whitened tensor
    M = (D^-0.5)*V';    
    mu = mean(Z,2);
    for i = 1:m
        e_i = [zeros(i-1,1) ;1 ;zeros(m-i,1)];
        T_z_s = T_z_s - (sigma^2)*(tensor_outer_prod([mu e_i e_i])+...
            tensor_outer_prod([e_i e_i mu])+tensor_outer_prod([ e_i mu e_i]));
    end
    T_w_s = symmetric_tensor_mode_product(T_z_s,M',1:3);
end
if denoising_method==3
    R_z_p = W'*(H_bin*H_bin'/n)*W;
    T_h = compute_joint_covariance_tensor(H_bin,1);
    T_z_p = symmetric_tensor_mode_product(T_h,W,1:3);
    [V,D] = eigs(R_z_p,d);
    
    % estimate whitened tensor
    M = (D^-0.5)*V';    
    T_w_s = symmetric_tensor_mode_product(T_z_p,M',1:3);
end

% get all eigenvectors
[X_NCM,lambda] = get_all_eigenvectors_parallel_filter(T_w_s,0.01);

end