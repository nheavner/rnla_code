function [x] = lusolve(L,U,PR,PC,b)
%Given a lower triangular n x n matrix L, an upper triangular n x n matrix
%U, n x n permutation matrices PR and PC, and an n x 1 vector b, returns an
%n x 1 vector x s.t. PR'*L*U*PC'*x = b.

b = PR*b;

x2 = backsolve(L,b);
x1 = backsolve(U,x2);
x = PC*x1;


end

