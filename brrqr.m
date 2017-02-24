function [Q,R,P] = brrqr(A,b,q,p)

%Given an m x n matrix A, m >= n, this function calculates an n x n matrix
%P, an m x m matrix Q, and an m x n matrix R s.t. A*P=Q*R, P and Q are
%orthonormal, R is upper triangular, and the diagonal entries of R
% approximate the singular values of A. 
% b: block size (typically b is approx. 50)
% p: oversampling parameter (typically p = 0,5,10, or b)
% q: number of "power iterations" to perform (typically q=0,1, or 2)

[m,n] = size(A);

P = eye(n,n);
Q = eye(m,m);

for i=1:floor(n/b)
    
    % update index vectors for accessing A
    if i > 1
        I1 = 1:b*(i-1);
    else
        I1 = [];
    end
    I2 = 1+b*(i-1):b*i;
    I3r = b*i+1:m;
    I3c = b*i+1:n;
    
    % determine pivoting matrix via randomized sampling
    Y = (A([I2 I3r],[I2 I3c])'*A([I2 I3r],[I2 I3c]))^q ...
        *A([I2 I3r],[I2 I3c])'*normrnd(0,1,m-(i-1)*b,min(b+p,n-(i-1)*b+p)); 
    [Ptil,~,~] = parthhqr(Y,b);
    %[Ptil,~,~] = svd(Y);
    
    % apply the pivot to A
    A([I1 I2 I3r],[I2 I3c]) = A([I1 I2 I3r],[I2 I3c])*Ptil;
    
    % compute the SVD of the current block of A
    [Qtil,Rtil] = qr(A([I2 I3r],I2));
    Rtil = Rtil(1:b,1:b);
    [Uprime,Dtil,Vtil] = svd(Rtil);
    Util = Qtil;
    placeholder = eye(m-(i-1)*b,m-(i-1)*b);
    placeholder(1:b,1:b) = Uprime;
    Util = Util*placeholder;
    
    % update A
    A([I2 I3r],I3c) = Util'*A([I2 I3r],I3c);
    A(I2,I2) = Dtil;
    A(I3r,I2) = 0;
    A(I1,I2) = A(I1,I2)*Vtil;

    % update Q
    placeholder = eye(m,m);
    placeholder([I2 I3r],[I2 I3r]) = Util;
    Q = Q*placeholder;
    
    % update P
    placeholder = eye(n,n);
    placeholder([I2 I3c],[I2 I3c]) = Ptil;
    P = P*placeholder;
    placeholder = eye(n,n);
    placeholder(I2,I2) = Vtil;
    P = P*placeholder;
    
end

% update index vectors for final step
I1 = 1:b*(i-1);
I2 = 1+b*(i-1):b*i;
I3r = b*i+1:m;
I3c = b*i+1:n;

% compute svd of final block
[Util,Dtil,Vtil] = svd(A(I3r,I3c));

% update A
A(I3r,I3c) = Dtil;
A([I1 I2],I3c) = A([I1 I2],I3c)*Vtil;

% update Q
placeholder = eye(m,m);
placeholder(I3r,I3r) = Util;
Q = Q*placeholder;

% update P
placeholder = eye(n,n);
placeholder(I3c,I3c) = Vtil;
P = P*placeholder;

R = A;

end

