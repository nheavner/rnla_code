function [A,U,S,V] = illcondmat(n)
% Given n, this function outputs an n x n matrix A whose singular values
% decay exponentially, with sigma_1 = 1, sigma_n = 10^(-5)

[U,~] = qr(randn(n,n),0);
[V,~] = qr(randn(n,n));

S = logspace(0,-13,n);
S = diag(S);

A = U*S*V';

end

