function [W,Mu,sigma_h] = generate_model_parameters(d,m)

    % generate hidden units parameters  (Mu,sigma_h)
    Mu = 0.2+0.2*rand(d,1);
    V = orth(randn(d));
    D = diag(rand(d,1));
    sigma_h = V*D*V';
    
    % generate weight matrix W
    W = randn(d,m);
    W = W./repmat(sqrt(sum(W.^2,1)),d,1);

end