function [A,U,S,V] = testlumat(n)
%outputs an n x n matrix designed to create a poor LU factorization when
%only partial, instead of complete, pivoting is used

A = zeros(n,n);

% matrix 1: growth factor around n-1 for partial, (n-1)/2 for complete
% for i=1:n
%     for j=1:n
%         A(i,j) = cos((i-1)*(j-1)*pi/(n-1));
%     end
% end

% Volterra equation: the big guns
L = 40;
h = L/(n-1);
k = 1;
C = 1;

A(1:n-1,1:n-1) = (1-k*h/2)*diag(diag(ones(n-1)));
A(1,1) = 1;
A(2:n,2:n-1) = A(2:n,2:n-1) + (-k*h)*tril(ones(n-1,n-2),-1);
A(2:n,1) = (-k*h/2)*ones(n-1,1);
A(1:n-1,n) = (-1/C)*ones(n-1,1);
A(end,end) = 1 - 1/C - k*h/2;

% two-point bvp
% m=2;
% h = 60/n;
% M = [-1/6 1;1 -1/6];
% for i=1:(n/2)-1
%     A(2*(i-1)+1:2*i,2*(i-1)+1:2*i) = eye(2);
%     A(2*i+1:2*i+2,2*(i-1)+1:2*i) = -expm(M*h);
% end
% A(n-1,n-1) = 1;
% A(n,n) = 1;
% A(1,n-1) = 1;
% A(2,n) = 1;

end

