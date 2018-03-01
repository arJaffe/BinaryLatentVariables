function [idx_hat,score] = ...
        filter_eigenvectors_gaussian_model(X,lambda,d,Z,sigma,filter_method)
    
    % [idx_hat,residual] = ...
    %    filter_eigenvectors_gaussian_model(X,lambda,d,Z,K,sigma,filter_method)
    %
    % Description: 
    % Given a matrix of n samples from a binary latent model 
    % described in "Learning Binary Latent Variable Models:
    % A Tensor Eigenpair Approach", and a candidate set for the pseudo
    % inverse matrix W, determines the correct
    % subset by one of two methods: 1. Binary Rounding, 1. Kolmogorov 
    % Smirnov test.
    %
    % Input: X - A matrix of l candidate vectors     
    %        lambda - eigenvalues of corresponding eigenvectors 
    %        d - Number of hidden variables of the binary latent model
    %        Z - Matrix of n samples from the binary latent model    
    %        sigma - noise level
    %        filter_method: 1. Binary rounding, 2. KS test
    %
    % Output:
    %        idx_hat - vector of d vectors with lowest scores
    %        score - score of all candidate sets
    %
    % Written by Ariel Jaffe, Roi Weiss, Shai Carmi, Yuval Kluger
    % and Boaz Nadler, 2018
        
    if filter_method == 1
        
        % filter by distance to 0,1
        H_hat = X'*Z;
        H_hat_r = round(H_hat);
        H_hat_r(H_hat_r>1)=1;
        H_hat_r(H_hat_r<0)=0;
        l = size(H_hat,1);
        res = sum( (H_hat-H_hat_r).^2,2)';
        sigma_i_vec = zeros(1,l);
        for i = 1:l
            sigma_i_vec(i) = norm(X(:,i));
        end
        res = res./(sigma_i_vec.^2);
        
        mean_val = mean(H_hat_r,2);
        res( (mean_val<0.01) | (mean_val>0.99))=inf;
        
        score = res;
    end
    
    % use Kolmogorov - Smirnov distance
    if filter_method==2
        
        l = size(X,2);
        
        H_hat = X'*Z;
        score = zeros(1,l);
        for i = 1:l
            
            % for each vector, obtain CDF and compare to theoretical
            
            % compute p and sigma
            p_th = 1/lambda(i)^2;
            sigma_i = sigma*norm(X(:,i));
            
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
            score(i) = max(abs(cdf_emp-cdf_th));
            close all
        end
        [~,sort_idx] = sort(score,'ascend');idx_hat = sort_idx(1:d);
        
    end    
end
