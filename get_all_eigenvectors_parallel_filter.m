function [X,lambda] = get_all_eigenvectors_parallel_filter(T_w,res)
    %parpool(8)
    
    % first run
    
    r = size(T_w,1);
    num_runs = 4000*r;    
    X = zeros(r,num_runs);
    lambda = zeros(1,num_runs);
    converge = zeros(1,num_runs);   
    
    % run parfor loop
    parfor i=1:num_runs       
           [X(:,i),lambda(i),~,~,converge(i)] = ...
                orthogonal_newton_correction_method_b(T_w,300,10^-12);    
    end    
    X(:,converge==0)=[];
    lambda(converge==0)=[];
    X(:,lambda==0)=[];
    lambda(lambda==0)=[];   
    ctr = 2;
    while ctr<=size(X,2)       
        if max(abs(X(:,ctr)'*X(:,1:(ctr-1))))>(1-res)
               X(:,ctr) = [];
               lambda(ctr)=[];               
        else
           ctr = ctr+1; 
        end
    end 
    X(:,lambda<0)=-1*X(:,lambda<0);
    lambda(lambda<0) = -1*lambda(lambda<0);
        
    [lambda,sort_idx] = sort(lambda,'descend');
    X = X(:,sort_idx); 
    
    X(:,lambda<1)=[];
    lambda(lambda<1)=[];
    
end


