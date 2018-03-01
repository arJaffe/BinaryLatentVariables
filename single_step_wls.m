function W_wls = single_step_wls(Z,W,sigma,nc)
    
    % W_ls = single_step_wls(Z,W,sigma)
    %
    % Description: 
    % Given a matrix of n samples from the binary latent model 
    % described in "Learning Binary Latent Variable Models:
    % A Tensor Eigenpair Approach", and an initial estimate of the weight
    % matrix W, the function improves the estimate with a single weighted
    % least square step
    %
    % Input: 
    %        W - initial estimate of weight matrix
    %        Z - Matrix of n samples from the binary latent model    
    %        sigma - noise level
    %        nc - number of candidate binary vectors for each sample
    %
    % Output:
    %        W_wls - improved estimate
    %
    % Written by Ariel Jaffe, Roi Weiss, Shai Carmi, Yuval Kluger
    % and Boaz Nadler, 2018
    
    n = size(Z,2);
    [d,m]= size(W);
    H_bin = de2bi(0:2^d-1)';    
    H_ls = zeros(d,nc,n);
    W_ls = zeros(nc,n);
    for i = 1:n
       res_vec = sum( (W'*H_bin -repmat(Z(:,i),1,size(H_bin,2))).^2,1);
       [min_val,min_idx] = sort(res_vec,'ascend');
       H_ls(:,:,i) = H_bin(:,min_idx(1:nc)); 
       W_ls(:,i) = min_val(1:nc);
    end
    
    H_ls = reshape(H_ls,d,nc*n);
    Z_ls = reshape(repmat(Z,nc,1),m,nc*n);
    weight_vec = exp(-1*W_ls/(2*(sigma^2)));
    weight_vec = weight_vec./(sum(weight_vec,1));
    weight_vec = weight_vec(:);
        
    for i = 1:nc*n
       H_ls(:,i) = weight_vec(i)*H_ls(:,i);
       Z_ls(:,i) = weight_vec(i)*Z_ls(:,i);
    end
    
    W_wls = ((H_ls*H_ls')^-1)*H_ls*Z_ls';

end