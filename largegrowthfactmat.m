function [A] = largegrowthfactmat(n)
%Outputs an n x n matrix A which has a growth factor of 2^(n-1)

M = -1*ones(n);
M = M - triu(M,1);
M = M - 2*diag(diag(M));

T = normrnd(0,1,n-1,n-1);
T = T - tril(T,-1);

d = zeros(n,1);
for i=1:n
    d(i) = 2^(i-1);
end

theta = 20;

A = zeros(n);
A(1:end-1,1:end-1) = T;
A(:,end) = theta;
A = M*A;


end

