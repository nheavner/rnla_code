function [Q,R,Pmat] = gsqr(A)
% This function preforms the modified Gram-Schmidt algorithm on the columns
% of a full rank m x n matrix A where (m >= n) to yield a factorization
% A*P=Q*R, where P is an n x n  permutation
% matrix, Q is an m x n matrix with orthonormal
% columns, and R is an upper triangular n x n matrix whose diagonal
% elements decay.

[m,n] = size(A);

Q = zeros(m,n);
P = zeros(n,1);
for i=1:n
    P(i) = i;
end

for i=1:n
    l = zeros(n,1);
    l(i:n) = sqrt(sum(A(:,i:n).^2));
    [~,ind] = max(l);
    
    % update P
    temp = P(ind);
    P(ind) = P(i);
    P(i) = temp;
    
    % calculate A*P
    temp = A(:,ind);
    A(:,ind) = A(:,i);
    A(:,i) = temp;
    
    if i == 1
        Q(:,i) = A(:,i)/norm(A(:,i));
    else
        Q(:,i) = A(:,i) - Q*Q'*A(:,i);
        Q(:,i) = Q(:,i) / norm(Q(:,i));
    end
    
end

R = Q'*A;
Pmat = eye(n,n);
Pmat = permcol(Pmat,P);

end

