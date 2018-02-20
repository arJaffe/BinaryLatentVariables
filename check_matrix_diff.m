function [res,X_hat_perm] = check_matrix_diff(X,X_hat,flag)
    
    % compute cost matrix    
    if flag==0
    D = pdist2(X_hat,X);
    else
    D = pdist2(X_hat,X);       
    end
    [C,res] = hungarian(D.^2);
    X_hat_perm = X_hat(C,:); 
    
    
end