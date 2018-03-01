function [H,W,itr_ctr] = ALS_iterations(Z,d,delta,H_init)
    
    % [H,W,itr_ctr] = ALS_iterations(Z,d,delta,H_init)
    %
    % Description: 
    % Computes binary decomposition of a m x n matrix Z
    % into two matrices of dimensions d x m and d x n where d<m,n
    % via alternating least square iterations
    % described in "Learning Binary Latent Variable Models:
    % A Tensor Eigenpair Approach"
    %
    % Input: Z - m x n real matrix
    %        d - required inner dimension
    %        delta - convergence threshold
    %        H_init(opt.) - initial value of H
    %
    % Output:
    %        W - d x m real matrix
    %        H - m x n binary matrix
    %        itr_ctr - number of iterations
    %
    % Written by Ariel Jaffe, Roi Weiss, Shai Carmi, Yuval Kluger
    % and Boaz Nadler, 2018
    
    
    [m,n] = size(Z);
    if nargin==4
        H = H_init;
    else
        H = randsrc(d,n,[1 0]);
    end
    
    H_bin = de2bi(0:2^d-1)';
    R = inf;
    W_c = zeros(d,m);
    itr_ctr = 0;
    while R>delta
        
        % estimate W given H
        W = ((H*H')^-1)*H*Z';
        
        % optimal estimate of H given W        
        parfor i = 1:n
            res_vec = sum( (W'*H_bin -repmat(Z(:,i),1,size(H_bin,2))).^2,1);
            [~,min_idx] = min(res_vec);
            H(:,i) =  H_bin(:,min_idx);
        end
        
        R = norm(W-W_c,'fro');        
        W_c = W;
        itr_ctr = itr_ctr+1;
    end
end