function [A] = counterexample_GEPP(n,r,u,v);
%       Function counterexample_GEPP generates a matrix which fails
% GEPP in terms of large element growth.
%       This is a generalization of the Wilkinson matrix.
%
if (nargin == 2)
u = rand(n,r);
v = rand(n,r);
A = - triu(u * v');
for k = 2:n
umax = max(abs(A(k-1,k:n))) * (1 + 1/n);
A(k-1,k:n)  = A(k-1,k:n) / umax;
end
A = A - diag(diag(A));
A = A' + eye(n);
A(1:n-1,n) = ones(n-1,1);
else
A = triu(u * v');
A = A - diag(diag(A));
A = A' + eye(n);
A(1:n-1,n) = ones(n-1,1);
end


end

