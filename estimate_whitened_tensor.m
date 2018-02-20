function T_w_ten = estimate_whitened_tensor(T_z_s,M_inv)

[d,m] = size(M_inv);

% estimate whitened tensor using least square
T_z_mat = reshape(T_z_s,m^2,m);
%T_w_mat = reshape(T_w_s,d^2,d);

%M_kron = kron(M,M);
M_inv_kron = kron(M_inv,M_inv);
M_inv_kron = M_inv_kron';

%norm(M_kron*T_z_mat*M'-T_w_mat)
%norm(M_inv_kron*T_w_mat*M_inv-T_z_mat)

%create result matrix B and A

B = zeros(m*(m-1),d);
A = zeros(m*(m-1),d^2);
ctr = 1;
for i = 1:m
    for j = 1:m
        if (i~=j)
           idx = (i-1)*m+j; 
           A(ctr,:) = M_inv_kron(idx,:);
           idx_vec = setdiff(1:m,[i,j]); 
           M_hat = pinv(M_inv(:,idx_vec));
           B(ctr,:) = T_z_mat(idx,idx_vec)*M_hat; 
           ctr = ctr+1;
        end        
    end
end

%get least square
T_w_hat = ((A'*A)^-1)*A'*B;
%norm(T_w_hat-T_w_mat)
T_w_ten = reshape(T_w_hat,d,d,d);

end