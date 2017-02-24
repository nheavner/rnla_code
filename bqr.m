function [Q,R,P] = bqr(A,b,q)

%Given an m x n matrix A, m >= n, this function calculates an n x n matrix
%P, an m x m matrix Q, and an m x n matrix R s.t. A*P=Q*R,Q is
%orthonormal, and P is a permutation matrix

[m,n] = size(A);

P = eye(n,n);
Q = eye(m,m);

for i=1:floor(n/b)
    
    % determine pivoting matrix via randomized sampling
    Y = (A(1+(i-1)*b:m,1+(i-1)*b:n)'*A(1+(i-1)*b:m,1+(i-1)*b:n))^q ...
        *A(1+(i-1)*b:m,1+(i-1)*b:n)'*normrnd(0,1,m-(i-1)*b,min(2*b,n-(i-1)*b)); 
         % The over-sampling parameter is p = b as seen in the
         % final argument of the normrnd call
    [~,~,Ptil] = hhqrv2(Y); 
    
    % apply the pivot to A
    A(:,1+(i-1)*b:n) = A(:,1+(i-1)*b:n)*Ptil;
    
    % compute the QR factorization of the current block of A
    [Qtil,Rtil,Pprime] = hhqrv2(A(1+(i-1)*b:m,1+(i-1)*b:i*b));
    Rtil = Rtil(1:b,1:b);
    %[Uprime,Dtil,Vtil] = svd(Rtil);
    %Util = Qtil;
    %placeholder = eye(m-(i-1)*b,m-(i-1)*b);
    %placeholder(1:b,1:b) = Uprime;
    %Util = Util*placeholder;
    
    % update A
    A(1+(i-1)*b:m,1+i*b:n) = Qtil'*A(1+(i-1)*b:m,1+i*b:n);
    A(1+(i-1)*b:m,1+(i-1)*b:i*b) = 0;
    A(1+(i-1)*b:i*b,1+(i-1)*b:i*b) = Rtil;
    A(1:(i-1)*b,1+(i-1)*b:i*b) = A(1:(i-1)*b,1+(i-1)*b:i*b)*Pprime;

    % update Q
    placeholder = eye(m,m);
    placeholder(1+(i-1)*b:m,1+(i-1)*b:m) = Qtil;
    Q = Q*placeholder;
    
    % update P
    placeholder = eye(n,n);
    placeholder(1+(i-1)*b:n,1+(i-1)*b:n) = Ptil;
    P = P*placeholder;
    placeholder = eye(n,n);
    placeholder(1+(i-1)*b:i*b,1+(i-1)*b:i*b) = Prpime;
    P = P*placeholder;
    
end

% compute QR factorization of final block
[Qtil,Rtil,Ptil] = hhqrv2(A(1+(i-1)*b:m,1+(i-1)*b:n));

% update A
A(1+(i-1)*b:m,1+(i-1)*b:n) = Rtil;
A(1:(i-1)*b,1+(i-1)*b:n) = A(1:(i-1)*b,1+(i-1)*b:n)*Ptil;

% update Q
placeholder = eye(m,m);
placeholder(1+(i-1)*b:m,1+(i-1)*b:m) = Qtil;
Q = Q*placeholder;

% update P
placeholder = eye(n,n);
placeholder(1+(i-1)*b:n,1+(i-1)*b:n) = Ptil;
P = P*placeholder;

R = A;


end

