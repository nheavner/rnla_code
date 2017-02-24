function [L,U] = mylu(A)
%given an m x n matrix A, this code performs unpivoted Gaussian elimination
% to compute a unit lower triangular matrix L and an upper triangular
% matrix U such that A = L*U.

%NOTE: This function does not use pivoting and is therefore unstable!!

[m,n] = size(A);

if (m == n) || (m < n)
    for k=1:m-1 % sums over rows
        for rho=k+1:m % sums over rows
            A(rho,k) = A(rho,k)/A(k,k);
            A(rho,k+1:end) = A(rho,k+1:end) - A(rho,k)*A(k,k+1:end);
        end    
    end
    
    L = eye(m,n) + tril(A,-1);
    L = L(:,1:m);
    U = triu(A);
    
else    % in this case, m > n
    for k=1:n % sums over cols
        for rho = k+1:m % also sums over rows
            A(rho,k) = A(rho,k)/A(k,k);
            if k < n
                A(rho,k+1:end) = A(rho,k+1:end) - A(rho,k)*A(k,k+1:end);
            end
        end
    end
    
    L = eye(m,n) + tril(A,-1);
    U = triu(A);
    U = U(1:n,:);
    
end

end

