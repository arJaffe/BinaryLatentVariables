function [X,lambda] = get_all_eigenvectors_parallel_filter(T_w,res)
    
    % Description: 
    % Get all real eigenvectors of a symmetric tensor via
    % Newton correction method, described in
    % Newton correction methods for computing real eigenpairs of
    % symmetric tensors" (2018)
    %
    % The function runs 4000*r iterations of O-NCM, removes iterations
    % where the method failed to converge or found eigenvalues below 1
    % and filter similar eigenvectors
    %
    % Input: T_w - symmetric tensor
    %        res - resolution of eigenvectors, 
    %
    % Output: X - matrix of computed eigenvectors
    %         lambda - corresponding eigenvalues
    
    
    r = size(T_w,1);
    num_runs = 4000*r;    
    X = zeros(r,num_runs);
    lambda = zeros(1,num_runs);
    converge = zeros(1,num_runs);   
    
    % run parfor loop
    parfor i=1:num_runs       
           [X(:,i),lambda(i),~,~,converge(i)] = ...
                orthogonal_newton_correction_method(T_w,300,10^-12);    
    end
    
    % remove eigenvectors failed to converge or with lambda < 1
    X(:,converge==0)=[];
    lambda(converge==0)=[];
    X(:,lambda<0)=-1*X(:,lambda<0);
    lambda(lambda<0) = -1*lambda(lambda<0);
    X(:,lambda<1)=[];
    lambda(lambda<1)=[];
    
    % remove similar eigenvectors \|x_j-x_i\|<res
    ctr = 2;
    while ctr<=size(X,2)       
        if max(abs(X(:,ctr)'*X(:,1:(ctr-1))))>(1-res)
               X(:,ctr) = [];
               lambda(ctr)=[];               
        else
           ctr = ctr+1; 
        end
    end 
    
    % sort lambda descending order    
    [lambda,sort_idx] = sort(lambda,'descend');
    X = X(:,sort_idx); 
    
end


