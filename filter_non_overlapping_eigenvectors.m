function [idx_hat,residual] = filter_non_overlapping_eigenvectors(score,X,H_hat_r,d,threshold)
    
    
   [sort_val,sort_idx] = sort(score,'ascend');
    residual = zeros(1,length(score));
    idx_hat = zeros(1,d);
    idx_hat(1) = sort_idx(1);
    
    % check for rest of 
    ctr_a = 1; %idx of location in idx_hat
    ctr_b = 2; %idx of lovation in sort_idx
    while ((ctr_a<d) && (ctr_b<=length(sort_idx)))
        %corr_i_j = (2*H_hat_r(sort_idx(ctr_b),:)-1)*...
        %    (2*H_hat_r(idx_hat(1:ctr_a),:)'-1)/size(H_hat_r,2);
        corr_i_j = max(abs(X(:,sort_idx(ctr_b))'*X(:,idx_hat(1:ctr_a))));
        if max(abs(corr_i_j))<threshold && mean(H_hat_r(sort_idx(ctr_b),:) )>0.01...
            && mean(H_hat_r(sort_idx(ctr_b),:))<0.99
            
           ctr_a = ctr_a+1;
           idx_hat(ctr_a)= sort_idx(ctr_b);
           residual(ctr_a) = sort_val(ctr_b);
        else
           %max(abs(corr_i_j)) 
        end
        ctr_b=ctr_b+1;
    end  
    
end