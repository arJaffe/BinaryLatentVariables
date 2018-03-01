function [H_bin,Z] = generate_hidden_visible_samples(W,mu,R_h,n)

% [W,H_bin,Z_n] = generate_hidden_visible_samples(W,mu,R_h,n)
%
% Description: 
% Computes n independent random realizations of a binary 
% latent variable  model, described in "Learning Binary Latent Variable 
% Models:  A Tensor Eigenpair Approach"
% The function first samples from a multivariate Gaussian dist. N(mu,R_h)
% and then takes its binary rounding
%
% Input: W  - d x m matrix of weights 
%        mu - d x 1 vector of mean for Gaussian dist.
%        R_h - d x d PSD matrix
%        n  - number of independent realizations
%
% Output: 
%        Z_n - m x n matrix of n independent realizations of the observed
%              variables
%        H_bin - d x n binary matrix of n independent realization of the latent 
%                variables 
%
% Written by Ariel Jaffe, Roi Weiss, Shai Carmi, Yuval Kluger and Boaz
% Nadler, 2018

% sample from a Gaussian dist.
H_G = mvnrnd(repmat(mu',n,1),R_h)';

% take binary rounding
H_bin = round(H_G);
H_bin(H_bin>1)=1;H_bin(H_bin<0)=0;

% create visible nodes with linear dependencies
Z = W'*H_bin;

end