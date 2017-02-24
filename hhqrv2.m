function [Q,R,P] = hhqrv2(A)

% This function implements a QR factorization of an input matrix A using
% Householder reflections; yields factorization AP = QR

% A must not be underdetermined for this algorithm!

[m,n] = size(A);

%W = zeros(m,m);
%Y = zeros(m,m);
Q = eye(m,m);
P = zeros(n,1);
for i=1:n
    P(i) = i;
end
t = zeros(n,1);
for j=1:n
    
    if j < m % This statement prevents the last iteration of the loop if A is square;
             % in this case, the final iteration is not needed
        
        % determine pivot column
        l = zeros(n,1);
        l(j:n) = sqrt(sum(A(j:end,j:end).^2))'; % norms of the cols of A;
        
        [~,ind] = max(l);

        % Record which columns are swapped
        temp = P(ind);
        P(ind) = P(j);
        P(j) = temp;

        % create the Householder reflector associated with the largest column
        e1 = zeros(m-j+1,1);
        e1(1) = 1;
        beta = -sign(A(j,ind))*l(ind);
        v = (beta*e1 - A(j:end,ind)) / norm(beta*e1 - A(j:end,ind));

        % update YW form of Q
        if j==1
            Y(:,1) = v;
            W(:,1) = 2*v;
        else
            vn = [zeros(j-1,1) ; v];
            z = 2*vn - 2*W(:,1:j-1)*(Y(:,1:j-1)'*vn);
            W(:,j) = z;
            Y(:,j) = vn;
        end  
    
    % update A and Q
    A(j:end,j:end) = A(j:end,j:end) - 2*v*(v'*A(j:end,j:end));
    temp = A(:,ind);
    A(:,ind) = A(:,j);
    A(:,j) = temp;
    % Q(:,j:end) = Q(:,j:end) - 2*(Q(:,j:end)*v)*v'; % if we want to
    % compute Q directly with Householder products, use this
    
    end
    
end

Q = eye(m,m) - W*Y';
R = A;

end
