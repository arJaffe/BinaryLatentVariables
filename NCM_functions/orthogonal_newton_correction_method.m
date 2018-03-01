function [x,lambda,ctr,runtime,converge] = ...
    orthogonal_newton_correction_method_final(T,max_itr,delta,x_init)
% function [x,lambda,ctr,run_time,converge] = ...
%      orthogonal_newton_correction_method(T,max_itr,delta,x_init)
%
% Code by Ariel Jaffe, Roi Weiss and Boaz Nadler
% 2017, Weizmann Institute of Science
% ---------------------------------------------------------
% DESCRIPTION:
% 	This function implements the orthogonal Newton correction method
%   for computing real eigenpairs of symmetric tensors
%
% Input:    T - A cubic real and symmetric tensor
%           max_itr - maximum number of iterations until
%                     termination
%           delta - convergence threshold
%           x_init(opt) - initial point
%
% DEFAULT:
%   if nargin<4 x_init is chosen randomly over the unit sphere
%
% Output:   x - output eigenvector
%           ctr - number of iterations till convergence
%           run_time - time till convergence
%           convergence (1- converged)
% ---------------------------------------------------------
% FOR MORE DETAILS SEE:
%   A. Jaffe, R. Weiss and B. Nadler.
%   Newton correction methods for computing
%   real eigenpairs of symmetric tensors (2017)
% ---------------------------------------------------------

tic;

% get tensor dimensionality and order
n_vec = size(T);
m = length(n_vec);
n = size(T,1);
R = 1;
converge = 0;

% if not given as input, randomly initialize
if nargin<4
    x_init = randn(n,1);
    x_init = x_init/norm(x_init);
end

% init lambda_(k) and x_(k)
lambda = symmetric_tv_mode_product(T,x_init,m);
x_k = x_init;
ctr = 1;

while(R>delta && ctr<max_itr)
    
    % compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
    T_x_m_2 = symmetric_tv_mode_product(T,x_k,m-2);
    T_x_m_1 = T_x_m_2*x_k;
    g = -lambda*x_k+T_x_m_1;
    
    % compute Hessian H(x_k) and projecected Hessian H_p(x_k)
    U_x_k = null(x_k');
    H = (m-1)*T_x_m_2-lambda*eye(n);
    H_p = U_x_k'*H*U_x_k;
    H_p_inv = H_p^-1;
    
    %fix eigenvector
    y = -U_x_k*H_p_inv*U_x_k'*g;
    x_k_n = (x_k + y)/(norm(x_k + y));
    
    % update residual and lambda
    R = norm(x_k-x_k_n);
    x_k = x_k_n;    
    lambda = symmetric_tv_mode_product(T,x_k,m);
        
    ctr = ctr+1;    
end

x = x_k;
runtime = toc;
if ctr<max_itr
    converge=1;
end
end