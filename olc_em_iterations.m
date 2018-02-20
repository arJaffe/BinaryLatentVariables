function [H,W,ctr] = olc_em_iterations(Z,d,delta,H_init)

n = size(Z,2);
if nargin==4
   H = H_init; 
else
   H = randsrc(d,n,[1 0]);    
end

H_bin = de2bi(0:2^d-1)';    
R = inf;
f_norm = inf;
ctr = 0;
W_c = zeros(d,size(Z,1));
while R>delta
    ctr = ctr+1;
    
    % estimate W
    W = ((H*H')^-1)*H*Z';
    
    % M step 
    if 0
    H = ((W*W')^-1)*W*Z;
    H(H<0)=0;
    H(H>1)=1;
    end
    
    % optimal estimate of H
    if 1
    parfor i = 1:n
       res_vec = sum( (W'*H_bin -repmat(Z(:,i),1,size(H_bin,2))).^2,1);
       [~,min_idx] = min(res_vec);
       H(:,i) =  H_bin(:,min_idx);
    end    
    end
    
    %f_norm_c = norm(Z-W'*H,'fro');
    R = norm(W-W_c,'fro');
    %f_norm = f_norm_c;
    W_c = W;
    
end