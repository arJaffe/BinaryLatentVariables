function [H_bin,Z] = generate_hidden_visible_samples(W,mu,sigma_h,n)

% function generate_hidden_visible_samples
% W - Weight matrix
% mu - mean of hidden elements
% sigma_h - covariance of hidden elements (approximately)
% n - number of samples

% generate correlated normal samples
H_norm = mvnrnd(repmat(mu',n,1),sigma_h)';

% round samples to binary
H_bin = round(H_norm);
H_bin(H_bin>1)=1;H_bin(H_bin<0)=0;

% create visible nodes with linear dependencies
Z = W'*H_bin;

end