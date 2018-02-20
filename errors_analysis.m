function err_ctr = errors_analysis(X_NCM,ten_ev_idx,X_th_inv_n)
    
    if sum(ten_ev_idx==0)>0
        err_ctr=2;
    else
        
        err_ctr=0;
        d = size(X_NCM,1);
        th_idx = zeros(1,d);
        res = zeros(1,d);
        for i = 1:d
            [res(i),th_idx(i)] = max(abs(X_NCM(:,ten_ev_idx)'*X_th_inv_n(:,i)));
        end
        if (min(res)<0.9)
            err_ctr = 1;
            l = size(X_NCM,2);
            res_b = zeros(1,d);
            for i=1:d
                res_b(i) = max(abs(X_th_inv_n(:,i)'*X_NCM));
            end
            if (min(res_b)<0.9)
                err_ctr = 2;
            end
        end
    end
end