function [res,X_hat_perm] = check_matrix_diff(X,X_hat)
    
    % [res,X_hat_perm] = check_matrix_diff(X,X_hat)
    %
    % Description: 
    % The function applies the Hungarian method to 
    % compute a permutation on the rows of X_hat such that the 
    % Frobenius norm of X-X_hat is minimized
    % The function makes use of an implementation by Niclas Borlin, 
    % Department of Computing Science, Umeå University, Sweden. 
    % 
    % Input: two matrices X,X_hat
    % Output: X_hat_perm - A permuation of X_hat
    %         res - Frobenius norm of X-X_hat_perm
    
    % compute cost matrix        
    D = pdist2(X_hat,X);
    
    % Hungarian method
    [C,res] = hungarian(D.^2);
    X_hat_perm = X_hat(C,:);     
    
end