function [L,U,Pmat] = plu(A)
% Given an m x n matrix A with m >= n, this function uses the partial 
% pivoted LU-factorization
% algorithm to compute an m x m tril matrix L, an n x n triu matrix U, and
% an n x 1 "permutation matrix" P such that P*A = L*U

% NOTE: matrix A must be full rank for this algorithm!

% initialize matrices
[m,n] = size(A);
P = zeros(m,1);
for i=1:m
    P(i) = i;
end

for i=1:n

    [mval,mind] = max(abs(A(i:end,i)));
    mind = mind + (i-1);    % need to add on i to refer to the correct row of A
    
    % Record which rows are swapped
    temp = P(mind);
    P(mind) = P(i);
    P(i) = temp;
    
    % Swap the rows of A
    temp = A(mind,:);
    A(mind,:) = A(i,:);
    A(i,:) = temp;
    
    if A(i,i) ~= 0
        if i < n
            rho = i+1:m;               
            % perform the pivot
            A(rho,i) = A(rho,i)/A(i,i); % determines multipliers and stores them in lower part of A
            A(rho,i+1:end) = A(rho,i+1:end) - A(rho,i)*A(i,i+1:end); % updates rest of rows
            
        elseif (i == n) && (m > n)
            rho = i+1:m;
            A(rho,i) = A(rho,i)/A(i,i);
        else
            % in this case, you are in the bottom right corner, so do
            % nothing
        end
        
    end  

end

L = eye(m,n) + tril(A,-1);
U = triu(A);
U = U(1:n,:);
Pmat = eye(m,m);
Pmat = permrow(Pmat,P);

end