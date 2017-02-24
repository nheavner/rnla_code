function brrqrtest

% ptest
accuracy_test

end

function accuracy_test
% runs tests to compare accuracy of randUTV vs SVD and CPQR for different
% values of q

rng('default')
rng(0)

n = 300; % size of test matrix
b = 50; % block size
q = [0 1 2]; % power iteration parameter
p = 0;

% construct a test matrix
%A = test_mat_fast(n,n);
A = test_mat_s_shape(n,n);
%A = test_mat_slow_decay(n,n);
%A = test_mat_kahan(n);

% each column of E is the error vector for a value of p
E2 = zeros(n,length(q)); % error in spectral norm
Ef = zeros(n,length(q)); % error in frobenius norm

% error vectors for column-pivoted qr truncations
ecpqr2 = zeros(n,1);
ecpqrf = zeros(n,1);

for i=1:length(q)
    [Q,R,P] = brrqr(A,b,q(i),p);
    for k=1:n
        E2(k,i) = norm(A-Q(:,1:k)*R(1:k,:)*P',2);
        Ef(k,i) = norm(A-Q(:,1:k)*R(1:k,:)*P','fro');
    end
    
end

% compare our results again column-pivoted QR
[Q,R,P] = qr(A);
for k=1:n
    ecpqr2(k) = norm(A-Q(:,1:k)*R(1:k,:)*P',2);
    ecpqrf(k) = norm(A-Q(:,1:k)*R(1:k,:)*P','fro');
end

% compute the errors from truncating the SVD
[~,D,~] = svd(A);
ss = diag(D);
ssf = sqrt(triu(ones(length(ss)))*(ss.*ss));

% create error plots
figure(1)
subplot(1,2,1)
hold off
semilogy(1:n,ss,'k-',...
         1:n,E2(:,1)','r.-',1:n,E2(:,2)','b--',1:n,E2(:,3)','g-.',1:n,ecpqr2','m:')
axis([1,n,ss(end),ss(1)])
legend('SVD','randUTV, q=0','randUTV, q=1','randUTV, q=2','CPQR')
title('Spectral norm errors')
subplot(1,2,2)
hold off
semilogy(1:n,ssf,'k-',...
         1:n,Ef(:,1),'r.-',1:n,Ef(:,2),'b--',1:n,Ef(:,3),'g-.',1:n,ecpqrf','m:')
axis([1,n,ssf(end),ssf(1)])
legend('SVD','randUTV, q=0','randUTV, q=1','randUTV, q=2','CPQR')
title('Frobenius norm errors')

keyboard

end

function ptest
% runs tests to determine effect of oversampling parameter p


rng('default')
rng(0)

n = 300; % size of test matrix
b = 50; % block size
q = 0; % power iteration parameter

% construct a test matrix
%A = test_mat_fast(n,n);
A = test_mat_s_shape(n,n);
%A = test_mat_slow_decay(n,n);

% run accuracy tests for different values of p
p = [0 5 10 b];

% each column of E is the error vector for a value of p
E2 = zeros(n,length(p)); % error in spectral norm
Ef = zeros(n,length(p)); % error in frobenius norm

% error vectors for column-pivoted qr truncations
ecpqr2 = zeros(n,1);
ecpqrf = zeros(n,1);

for i=1:length(p)
    [Q,R,P] = brrqr(A,b,q,p(i));
    for k=1:n
        E2(k,i) = norm(A-Q(:,1:k)*R(1:k,:)*P',2);
        Ef(k,i) = norm(A-Q(:,1:k)*R(1:k,:)*P','fro');
    end
    
    % compare our results again column-pivoted QR
    [Q,R,P] = qr(A);
    for k=1:n
        ecpqr2(k) = norm(A-Q(:,1:k)*R(1:k,:)*P',2);
        ecpqrf(k) = norm(A-Q(:,1:k)*R(1:k,:)*P','fro');
    end
end

% compute the errors from truncating the SVD
[~,D,~] = svd(A);
ss = diag(D);
ssf = sqrt(triu(ones(length(ss)))*(ss.*ss));

% create error plots
figure(1)
subplot(1,2,1)
hold off
semilogy(1:n,ss,'k-',...
         1:n,E2(:,1)','r.-',1:n,E2(:,2)','b--',1:n,E2(:,3)','g-.',1:n,E2(:,4)','m:')
axis([1,n,ss(end),ss(1)])
legend('SVD','randUTV, q=0, p=0','randUTV, q=0, p=5','randUTV, q=0, p=10','randUTV, q=0, p=b')
title('Spectral norm errors')
subplot(1,2,2)
hold off
semilogy(1:n,ssf,'k-',...
         1:n,Ef(:,1),'r.-',1:n,Ef(:,2),'b--',1:n,Ef(:,3),'g-.',1:n,Ef(:,4),'m:')
axis([1,n,ssf(end),ssf(1)])
legend('SVD','randUTV, q=0, p=0','randUTV, q=0, p=5','randUTV, q=0, p=10','randUTV, q=0, p=b')
title('Frobenius norm errors')

figure(2)
subplot(1,2,1)
hold off
semilogy(1:n,abs(E2(:,1)-E2(:,4)) ./ abs(E2(:,1)))
% axis([1,n,ss(end),ss(1)])
title('Spectral norm error difference between p = 0 and p = b')
subplot(1,2,2)
hold off
semilogy(1:n,abs(Ef(:,1)-Ef(:,4)) ./ abs(Ef(:,1)))
% axis([1,n,ssf(end),ssf(1)])
title('Frobenius norm error difference between p = 0 and p = b')

keyboard

end

function [A] = test_mat_fast(m,n)
% this function creates an m x n matrix, m >= n,
% with rapidly-decaying singular values s_i = (10^(-5))^[(i-1)/(n-1)], i=1..n


U = orth(randn(m,m));
D = diag((10^(-5).^(0:(n-1))/(n-1)));
V = orth(randn(n,n));

A = U*D*V';

end

function [A] = test_mat_s_shape(m,n)
% this function creates an m x n matrix, m >= n, with singular values whose
% plot forms an S shape

x = linspace(0,1,n);

U = orth(randn(m,m));

% to get this function, we just modified the error function until we got
% the shape we wanted
D = diag(exp(3*(1+erf((-x+.4)/(.1*sqrt(2))))-log(403.35214)));
V = orth(randn(n,n));

A = U*D*V';

end

function A = test_mat_slow_decay(m,n)
% this function creates an m x n matrix, m >= n, with singular values
% s_i = 1 / (i)

% Determine the "inner dimension"
r = min([m,n]);

U = orth(randn(m,r));
V = orth(randn(n,r));
ss    = 1./(1:r);
D    = [diag(ss); zeros(m-n,n)];
A     = U*D*V';

end

function A = test_mat_kahan(n)
% this function generates an n x n Kahan matrix with parameter xi=0.99

xi = 0.99;

S = diag(xi.^(0:n-1));
K = eye(n);
K = K - triu(ones(n),1)*sqrt(1-xi^2);

A = S*K;

end
