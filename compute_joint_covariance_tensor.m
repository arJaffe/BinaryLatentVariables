function T_z = compute_joint_covariance_tensor(Z_n,par_flag)

    
[m,n] = size(Z_n);

if par_flag==0
T_z = zeros(m,m,m);
for i = 1:n    
    T_z = T_z + reshape(kron(Z_n(:,i)*Z_n(:,i)',Z_n(:,i)),m,m,m);
end
T_z = T_z/n;
else
   a = rem(n,100);
   Z_n = [Z_n zeros(m,100-a)];
   n_hat = size(Z_n,2);
   Z_n = reshape(Z_n,m,n_hat/100,100); 
   T_z = zeros(m,m,m,100);
   parfor i = 1:100
       T_z(:,:,:,i) = compute_joint_covariance_tensor(Z_n(:,:,i),0)
   end
   T_z = T_z*n_hat/100;
   T_z = sum(T_z,4)/n;
end

end