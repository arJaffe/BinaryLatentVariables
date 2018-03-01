function [W,Mu,Sigma_h] = generate_model_parameters(d,m)

% function [W,H_bin,Z_n] = generate_hidden_visible_samples(W,mu,R_h,n)
%
% Description: 
% Samples the parameters of the latent variable model
% described in "Learning Binary Latent Variable Models:  
% A Tensor Eigenpair Approach"
% 
% Input: d - number of hidden variables
%        m - number of visible variables
%
% Output: 
%        W - d x m matrix of weights
%        Mu - d x 1 vector of values between [0,1] 
%        Sigma_h - d x d PSD matrix 
%
% Written by Ariel Jaffe, Roi Weiss, Shai Carmi, Yuval Kluger
% and Boaz Nadler, 2018

    % generate hidden units parameters  (Mu,sigma_h)
    Mu = 0.2+0.2*rand(d,1);
    V = orth(randn(d));
    D = diag(rand(d,1));
    Sigma_h = V*D*V';
    
    % generate weight matrix W
    W = randn(d,m);
    W = W./repmat(sqrt(sum(W.^2,1)),d,1);

end