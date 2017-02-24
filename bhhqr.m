function [Q,R] = bhhqr(A,r)

% This function implements a blocked QR factorization of an input matrix A using
% Householder reflections; yields factorization A = QR

% A must not be underdetermined for this algorithm!

[m,n] = size(A);

Q = eye(m,m);
lambda = 1; % lambda keeps track of which column the current block starts at

while lambda <= n
    
    tau = min(lambda+(r-1),n); %tau keeps track of which column the current block ends at
  
    % create the Householder reflectors for the current block
    for j=lambda:tau
        e1 = zeros(m-j+1,1);
        e1(1) = 1;
        beta = -sign(A(j,j))*norm(A(j:end,j));
        v = (beta*e1 - A(j:end,j)) / norm(beta*e1 - A(j:end,j));
        v = [zeros(j-1,1) ; v];
        
        % update only the columns of A in the current block
        A(1:end,j:tau) = (eye(m,m) - 2*(v*v'))*A(1:end,j:tau);
        
        % update YW form of Q
        if j==lambda
            Y = v;
            W = 2*v;
        else
            z  = 2*v - 2*W*(Y'*v);
            W = [W z];
            Y = [Y v];
        end
    end
    
    % update the columns of A outside the block and Q
    if tau < n
        A(:,tau+1:n) = A(:,tau+1:n) - Y*(W'*A(:,tau+1:n));
    end
    Qn = eye(m,m) - W*Y';
    Q(:,lambda:m) = Q(:,lambda:m)*Qn(lambda:m,lambda:m);
    
    lambda = tau + 1;
end

R = A;

end

