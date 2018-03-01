function M_hat = complete_matrix_diagonal_elements(M,d,delta,max_itr)
    
    % M_hat = complete_matrix_diagonal_elements(M,r,delta)
    %
    % Description: Complete diagonal elements of symmetric matrix with 
    % assumed rank d
    %
    % Input: M - symmetric low rank matrix
    %        d - assumed rank
    %        delta -convergence threshold
    %        max_itr - maximal iterations
    %
    %        M_hat - completed matrix
    
    m = size(M,1);
    res = inf;
    M_hat = M;
    ctr = 1;    
    while (res>delta && ctr<max_itr)
        [V,D] = eigs(M_hat,d);
        M_hat_r = V*D*V';
        res = norm(diag(M_hat)-diag(M_hat_r));
        M_hat(logical(eye(m)))=diag(M_hat_r);
        ctr = ctr+1;        
    end    
end
