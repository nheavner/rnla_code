function [C,X] = id(A,k)
% Given an m x n matrix A of rank k, this algorithm uses randomized
% sampling techniques to produce a matrix C whose columns are a subset of
% the columns of A, and a matrix X whose entries are bounded by 2 and a
% subset of whose columns are the columns of the identity matrix, such that
% A = C*X;

[m,n] = size(A);

Omega = normrnd(0,1,k,m);

% Take a row sample
Y = Omega*A;

% Compute ID of Y
[Q,R,P] = qr(Y,0);
Pmat = eye(n,n);
Pmat = permcol(Pmat,P);

T = R(:,1:k) \ R(:,k+1:end);

X = [eye(k,k) T]*Pmat';

% for factorization Q = Q(:,J)*X, Q(:,J) = Qhat*R(:,1:k) contains the first
% k columns of Q*P (possibly in permuted order), so J corresponds to the col indices of original cols of
% Q; Thus, for A = C*X = A(:,J)*X, A(:,J) contains the first k columns
% of A*P

C = permcol(A,P);
C = C(:,1:k);

end