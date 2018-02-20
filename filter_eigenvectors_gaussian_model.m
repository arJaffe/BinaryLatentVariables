function [idx_hat,residual] = filter_eigenvectors_gaussian_model(X,lambda,d,Z,M,sigma,type)

if type ==1
    
    % filter by distance to 0,1
    H_hat = diag(1./lambda)*X'*M*Z;
    H_hat_r = round(H_hat);
    H_hat_r(H_hat_r>1)=1;
    H_hat_r(H_hat_r<0)=0;    
    
    res = sum( (H_hat-H_hat_r).^2,2);
    
    mean_val = mean(H_hat_r,2);
    res( (mean_val<0.01) | (mean_val>0.99))=inf;
    %[sort_val,sort_idx] = sort(res,'ascend');
    [idx_hat,~] = filter_non_overlapping_eigenvectors(res,X,H_hat_r,d,0.95);
    residual = res;
    
end

if type ==2
    
    % filter by distance to 0,1
    H_hat = diag(1./lambda)*X'*M*Z;
    H_hat_r = round(H_hat);
    H_hat_r(H_hat_r>1)=1;
    H_hat_r(H_hat_r<0)=0;    
    l = size(H_hat,1);    
    res = sum( (H_hat-H_hat_r).^2,2)';
    sigma_i_vec = zeros(1,l);
    for i = 1:l
       sigma_i_vec(i) = norm(X(:,i)'*M/lambda(i)); 
    end
    res = res./(sigma_i_vec.^2);
    
    mean_val = mean(H_hat_r,2);
    res( (mean_val<0.01) | (mean_val>0.99))=inf;
    [idx_hat,~] = filter_non_overlapping_eigenvectors(res,X,H_hat_r,d,0.9);  
    residual = res;
end

% use colmogorov smirnoff distance
if type==3
   
   l = size(X,2);
   
   H_hat = diag(1./lambda)*X'*M*Z;
   H_hat_r = round(H_hat);
   H_hat_r(H_hat_r>1)=1;
   H_hat_r(H_hat_r<0)=0;    
   residual = zeros(1,l);
   for i = 1:l
       
       % for each vector, obtain CDF and compare to theoretical
       
       % compute p and sigma
       p_th = 1/lambda(i)^2;
       sigma_i = (sigma/lambda(i))*norm(X(:,i)'*M);
       
       % get empirical CDF
       [cdf_emp,x_emp] = ecdf(H_hat(i,:));
              
       % comptue theoretical CDF       
       cdf_th = (1-p_th)*normcdf(x_emp,0,sigma_i)+p_th*normcdf(x_emp,1,sigma_i);
       %pdf_th = (1-p_th)*normpdf(x_emp,0,sigma_i)+p_th*normpdf(x_emp,1,sigma_i);
   
       
       %figure;
       %plot(x_emp,pdf_th,'linewidth',2);grid on;
       %plot(x_emp,cdf_th,'linewidth',2); hold on;grid on;
       %plot(x_emp,cdf_emp,'r','linewidth',2);
       %obj = gmdistribution([0;1],sigma_i,[1-p_th ;p_th]);
       %[h,p_val,ksstat(i),cv] = kstest(H_hat(i,:)','CDF',obj);
       %[h,p_val,ksstat(i),cv] = kstest(H_hat(i,:)','CDF',[x_emp cdf_emp]);
       residual(i) = max(abs(cdf_emp-cdf_th));
       close all
   end
   %[sort_val,sort_idx] = sort(residual,'ascend');   
   [idx_hat,res] = filter_non_overlapping_eigenvectors(residual,X,H_hat_r,d,0.9);
end

% use colmogorov smirnoff distance
if type==4
   
   l = size(X,2);
   
   H_hat = diag(1./lambda)*X'*M*Z;
   H_hat_r = round(H_hat);
   H_hat_r(H_hat_r>1)=1;
   H_hat_r(H_hat_r<0)=0;    
   residual = zeros(1,l);
   for i = 1:l
       
       % for each vector, obtain CDF and compare to theoretical
       
       % compute p and sigma
       p_th = 1/lambda(i)^2;
       sigma_i = (sigma/lambda(i))*norm(X(:,i)'*M);
       
       % get empirical CDF
       [cdf_emp,x_emp] = ecdf(H_hat(i,:));
              
       % comptue theoretical CDF       
       cdf_th = (1-p_th)*normcdf(x_emp,0,sigma_i)+p_th*normcdf(x_emp,1,sigma_i);
       %pdf_th = (1-p_th)*normpdf(x_emp,0,sigma_i)+p_th*normpdf(x_emp,1,sigma_i);
   
       
       figure;
       %plot(x_emp,pdf_th,'linewidth',2);grid on;
       plot(x_emp,cdf_th,'linewidth',2); hold on;grid on;
       plot(x_emp,cdf_emp,'r','linewidth',2);
       close all
       residual(i) = mean(abs(cdf_emp-cdf_th));
   end
   [idx_hat,score] = filter_non_overlapping_eigenvectors(residual,X,H_hat_r,d,0.9);
end



end
