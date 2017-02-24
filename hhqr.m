function [Q,R,P] = hhqr(A)

% This function implements a QR factorization of an input matrix A using
% Householder reflections; yields factorization AP = QR

% A must not be underdetermined for this algorithm!

[m,n] = size(A);

Q = eye(m,m);
P = eye(n,n);

for j=1:n-1

    % determine pivot column
    l = zeros(n,1);
    for i=j:n
        l(i) = norm(A(j:end,i));
    end
    [~,ind] = max(l);
    
    % create permutation matrix that swaps the largest column with the
    % first unfinished column
    Pn = eye(n,n);
    Pn(j,j) = 0;
    Pn(ind,j) = 1;
    Pn(ind,ind) = 0;
    Pn(j,ind) = 1;
    
    % create the Householder reflector associated with the largest column
    e1 = zeros(m-j+1,1);
    e1(1) = 1;
    beta = -sign(A(j,ind))*norm(A(j:end,ind));
    v = (beta*e1 - A(j:end,ind)) / norm(beta*e1 - A(j:end,ind));
    H = eye(m-j+1,m-j+1) - 2*(v*v');
    
    Qn = eye(m,m);
    Qn(j:end,j:end) = H;
    
    % update A, Q, and P
    A = Qn*A*Pn;
    Q(:,j:end) = Q(:,j:end)*H;
    %Q = Q*Qn;
    P = P*Pn;

end

R = A;

end

