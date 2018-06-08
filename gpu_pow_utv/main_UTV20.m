function main_UTV20


%LOCAL_updatetest
%LOCAL_singlestep
LOCAL_singletest
%LOCAL_errorplot
%LOCAL_timings
%LOCAL_timings_gpu
%LOCAL_plotdiags

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function LOCAL_updatetest
rng(0)
b      = 25;    % Step size.
n      = 150;   % Matrix dimension        
m      = 200;   % Matrix dimension        
n_iter = 3;     % Parameter in power iteration.
p      = 0;

A  = LOCAL_fast_decay(m,n,3*n);
ss = svd(A);

G = randn(m,b+p);
Y = A'*G;
for i = 1:n_iter
  Y = A'*(A*Y);
end
[V,~] = LOCAL_nonpiv_QR(Y,b);
Z = Y';

B = A*V;
Z = Z*V;

disp(norm(B - B*pinv(Z)*Z))
disp(ss(b+1))

keyboard


return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This driver function tests the accuracy in the first step of randUTV
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function LOCAL_singlestep

%%% Seed the random number generated, and set various parameters:
rng(0)
b      = 25;    % Step size.
n      = 150;   % Matrix dimension        
m      = 200;   % Matrix dimension        
n_iter = 2;     % Parameter in power iteration.

%%% Create a test matrix, and compute its singular values.
%A = LOCAL_fast_decay(n,n,3*n);
%A = LOCAL_slow_decay(n,n);
%A = randn(n);
A = LOCAL_S_curve(m,n,0.4*n,1e-2);

%%% Draw a Gaussian random matrix.
G = randn(m,b);

%%% Compute a sampling matrix Y whose columns form an approximate basis
%%% for the b dominant right singular vectors of A.
Y = A'*G;
for i = 1:n_iter
  Y = A'*(A*Y);
end

%%% Build the right transformation matrix.
[V,~,~] = qr(Y);

%%% Apply V from the right.
A1p = A*V(:,1:b);
A2p = A*V(:,(b+1):n);

%%% Compute the full SVD of the leading b columns of A*V.
[U,D,Wtilde] = svd(A1p);

%%% Apply U' to the trailing columns of A*V.
B2 = U'*A2p;

%%% Build the transformed matrix B.
B = [D, B2];

%%% Update V to reflect the rotation in Wtilde (from the SVD of the diagonal block).
V(:,1:b) = V(:,1:b)*Wtilde;

%%% Compute various errors:
fprintf(1,'norm(A - U*B*trans(V))         = %12.5e\n',norm(A - U*B*V'))
fprintf(1,'norm(trans(U)*U - eye(m))      = %12.5e\n',norm(U'*U - eye(m)))
fprintf(1,'norm(trans(V)*V - eye(n))      = %12.5e\n',norm(V'*V - eye(n)))
[Q,~,~] = qr(Y,0);
fprintf(1,'norm(A - A*Q*trans(Q))         = %12.5e (error in randomized sampling)\n',norm(A - (A*Q)*Q'))
fprintf(1,'norm(A - A*V1*trans(V1))       = %12.5e (error in randomized sampling)\n',norm(A - (A*V(:,1:b))*V(:,1:b)'))
fprintf(1,'norm([B12;B22])                = %12.5e (error in randomized sampling)\n',norm(B2))
fprintf(1,'norm(A - U1*trans(U1)*A)       = %12.5e (error in rank-b approximation)\n',norm(A - U(:,1:b)*(U(:,1:b)'*A)))
fprintf(1,'norm(A - U1*B(1:b,:)*trans(V)) = %12.5e (error in rank-b approximation)\n',norm(A - U(:,1:b)*B(1:b,:)*V'))
fprintf(1,'norm(B22)                      = %12.5e (error in rank-b approximation)\n',norm(B2((b+1):m,:)))
Z = A'*G;
for i = 1:n_iter
  Z = A'*(A*Z);
end
Z = A*Z;
[Q,~,~] = qr(Z,0);
fprintf(1,'norm(A - Q*trans(Q)*A)         = %12.5e (error in randomized sampling with q+0.5 power iterations)\n',norm(A - Q*(Q'*A)))
keyboard
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This driver routine calls randUTV once.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function LOCAL_singletest

%%% Seed the random number generated, and set various parameters:
rng(0)
b      = 128;            % Step size.
n      = 1000;           % Matrix size.
m      = 1000;           % Matrix size.
n_iter = 1;             % Parameter in power iteration.
p      = 0;             % Amount of oversampling to be done.

%%% Create a test matrix, and compute its singular values.
%A = LOCAL_fast_decay(m,n,3*n);
%A = LOCAL_slow_decay(n,n);
%A = randn(n);
A = LOCAL_S_curve(n,n,round(0.5*n),1e-2);

%%% Compute reference factorizations (SVD and CPQR).
ss = svd(A);
[Qref,Rref,Pref] = qr(A);

%%% Compute the errors for the reference factorizations.
err_svd_spec  =    ss(1)*ones(n,1);
err_svd_frob  = norm(ss)*ones(n,1);
err_cpqr_spec =    ss(1)*ones(n,1);
err_cpqr_frob = norm(ss)*ones(n,1);
err_rutv_spec =    ss(1)*ones(n,1);
err_rutv_frob = norm(ss)*ones(n,1);
for i = 1:(n-1)
  ind                = (i+1):n;
  err_svd_spec(i+1)  = ss(i+1);
  err_svd_frob(i+1)  = norm(ss(ind));
  err_cpqr_spec(i+1) = norm(Rref(ind,ind));
  err_cpqr_frob(i+1) = norm(Rref(ind,ind),'fro');
end

%%% Call randUTV
rng(0)
%[T,U,V] = LOCAL_randUTV(A,b,n_iter,p);
%[T,U,V] = LOCAL_randUTV_econ(A,b,n_iter,p);
[T,U,V] = LOCAL_randUTV_gpu(A,b,n_iter,p);
fprintf(1,'Factorization error ||A - U*T*trans(V)|| = %12.5e\n',norm(A - U*T*V'))

%%% Compute errors:
for i = 1:(n-1)
  ind                = (i+1):n;
  err_rutv_spec(i+1) = norm(T(ind,ind));
  err_rutv_frob(i+1) = norm(T(ind,ind),'fro');
end

%%% Generate plots
nstep = ceil(n/b);
xlim  = b*(1:(nstep-1));
ax    = [0,n,min(ss),norm(ss)];

figure(1)
subplot(1,2,1)
semilogy(0:(n-1),err_svd_spec,'k',...
         0:(n-1),err_cpqr_spec,'b',...
         0:(n-1),err_rutv_spec,'r',...
         [1;1]*xlim,[err_svd_spec(1);err_svd_spec(n)]*ones(1,nstep-1),'g:')
title(sprintf('Spectral norm errors: p=%d',p))
xlabel('k')
ylabel('||A - A_k||')
axis(ax)
legend('svds','col piv QR',sprintf('randUTV: q=%d',n_iter),'Location','NorthEast')

subplot(1,2,2)
semilogy(0:(n-1),err_svd_frob,'k',...
         0:(n-1),err_cpqr_frob,'b',...
         0:(n-1),err_rutv_frob,'r',...
         [1;1]*xlim,[err_svd_frob(1);err_svd_frob(n)]*ones(1,nstep-1),'g:')
title(sprintf('Frobenius norm errors: p=%d',p))
xlabel('k')
ylabel('||A - A_k||')
axis(ax)
legend('svds','col piv QR',sprintf('randUTV: q=%d',n_iter),'Location','NorthEast')

figure(2)
subplot(1,2,1)
plot(1:(n-1),err_rutv_spec(2:n)./err_svd_spec(2:n),'r',...
     1:(n-1),err_cpqr_spec(2:n)./err_svd_spec(2:n),'b')

subplot(1,2,2)
plot(1:(n-1),err_rutv_frob(2:n)./err_svd_frob(2:n),'r',...
     1:(n-1),err_cpqr_frob(2:n)./err_svd_frob(2:n),'b')

keyboard

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This driver function calls randUTV for several different parameter values
% and compares the results in a single plot.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function LOCAL_errorplot
tic
%%% Seed the random number generated, and set various parameters:
rng(0)
b     = 40;      % Step size.
n     = 400;     % Matrix size.
m     = 400;     % Matrix size.
q_max = 2;       % Maximal power iteration parameter to be tested.
p     = 10;      % Amount of oversampling to be done.
%kvec  = 0:(n-1); % The vector of k's for which errors are computed.        
kvec  = 0:1:(n-1); % The vector of k's for which errors are computed.        

%%% Create a test matrix, and compute its singular values.
%A = LOCAL_fast_decay(m,n,3*n);
%A = LOCAL_slow_decay(m,n);
%A = randn(n);m=n;
A = LOCAL_S_curve(m,n,0.5*n,1e-2);
%A  = LOCAL_gap_decay(m,n);

%%% Compute reference factorizations (SVD and CPQR).
ss = svd(A);
[~,Rref,~] = qr(A);
[~,Lref,~] = LOCAL_QLP(A);

%%% Compute the errors for the reference factorizations.
err_svd_spec     =    ss(1)*ones(n,1);
err_cpqr_spec    =    ss(1)*ones(length(kvec),1);
err_qlp_spec     =    ss(1)*ones(length(kvec),1);
err_rutv_spec    =    ss(1)*ones(length(kvec),q_max+1);
err_rutv_spec_p0 =    ss(1)*ones(length(kvec),q_max+1);
err_svd_frob     = norm(ss)*ones(n,1);
err_cpqr_frob    = norm(ss)*ones(length(kvec),1);
err_qlp_frob     = norm(ss)*ones(length(kvec),1);
err_rutv_frob    = norm(ss)*ones(length(kvec),q_max+1);
err_rutv_frob_p0 = norm(ss)*ones(length(kvec),q_max+1);
dd               = zeros(n,q_max+1);
dd_p0            = zeros(n,q_max+1);
for k = 1:(n-1)
  Jrem              = (k+1):n;
  err_svd_spec(k+1) = ss(k+1);
  err_svd_frob(k+1) = norm(ss(Jrem));
end
fprintf(1,'Calculating errors from CPQR ... ')
for i = 2:length(kvec)
  k                = kvec(i);
  Irem             = (k+1):m;
  Jrem             = (k+1):n;
  err_cpqr_spec(i) = norm(Rref(Irem,Jrem));
  err_cpqr_frob(i) = norm(Rref(Irem,Jrem),'fro');
  err_qlp_spec(i)  = norm(Lref(Irem,Jrem));
  err_qlp_frob(i)  = norm(Lref(Irem,Jrem),'fro');
end
fprintf(1,'Done.\n')

fprintf(1,'Calculating errors from UTV ... ')
for q = 0:q_max
  [T,U,V] = LOCAL_randUTV_econ(A,b,q,p);
  %%% Store away the diagonal of T.
  dd(:,q+1) = diag(T);
  %%% Compute errors:
  for i = 2:length(kvec)
    k                    = kvec(i);
    Irem                 = (k+1):m;
    Jrem                 = (k+1):n;
    err_rutv_spec(i,q+1) = norm(T(Irem,Jrem));
    err_rutv_frob(i,q+1) = norm(T(Irem,Jrem),'fro');
  end
  [T,U,V] = LOCAL_randUTV_econ(A,b,q,0);
  %%% Store away the diagonal of T.
  dd_p0(:,q+1) = diag(T);
  %%% Compute errors:
  for i = 2:length(kvec)
    k                       = kvec(i);
    Irem                    = (k+1):m;
    Jrem                    = (k+1):n;
    err_rutv_spec_p0(i,q+1) = norm(T(Irem,Jrem));
    err_rutv_frob_p0(i,q+1) = norm(T(Irem,Jrem),'fro');
  end
end
fprintf(1,'Done.\n')

Rref_diag = diag(Rref);
Lref_diag = diag(Lref);

toc

%clear A Irem Jrem Pref Qref Rref T U V
%
%save data_n=300

%%% Generate plots
nstep = ceil(n/b);
xlim = b*(1:(nstep-1));
ax    = [0,n,min(ss),norm(ss)];

figure(1)
subplot(1,2,1)
semilogy(0:(n-1),err_svd_spec,'k',...
         kvec,err_cpqr_spec,'m',...
         kvec,err_qlp_spec,'g',...
         kvec,err_rutv_spec_p0(:,1),'b',...
         kvec,err_rutv_spec_p0(:,2),'c',...
         kvec,err_rutv_spec_p0(:,3),'r',...
         kvec,err_rutv_spec(:,1),'b:',...
         kvec,err_rutv_spec(:,2),'c:',...
         kvec,err_rutv_spec(:,3),'r:',...
         [1;1]*xlim,[err_svd_spec(1);err_svd_spec(n)]*ones(1,nstep-1),'g:')
title(sprintf('Spectral norm errors: p=%d',p))
xlabel('k')
ylabel('||A - A_k||')
axis(ax)
legend('svds','col piv QR','QLP','randUTV: q=0,p=0','randUTV: q=1,p=0','randUTV: q=2,p=0',...
                           'randUTV: q=0','randUTV: q=1','randUTV: q=2',...
                           'Location','NorthEast')

subplot(1,2,2)
semilogy(0:(n-1),err_svd_frob,'k',...
         kvec,err_cpqr_frob,'m',...
         kvec,err_qlp_frob,'g',...
         kvec,err_rutv_frob_p0(:,1),'b',...
         kvec,err_rutv_frob_p0(:,2),'c',...
         kvec,err_rutv_frob_p0(:,3),'r',...
         kvec,err_rutv_frob(:,1),'b:',...
         kvec,err_rutv_frob(:,2),'c:',...
         kvec,err_rutv_frob(:,3),'r:',...
         [1;1]*xlim,[err_svd_frob(1);err_svd_frob(n)]*ones(1,nstep-1),'g:');
title(sprintf('Frobenius norm errors: p=%d',p))
xlabel('k')
ylabel('||A - A_k||')
axis(ax)
legend('svds','col piv QR','QLP','randUTV: q=0,p=0','randUTV: q=1,p=0','randUTV: q=2,p=0',...
                           'randUTV: q=0','randUTV: q=1','randUTV: q=2',...
                           'Location','NorthEast')

figure(2)
subplot(1,2,1)
max_err_spec = max(100*(err_cpqr_spec./err_svd_spec(1+kvec)-1));
plot(kvec,100*(err_cpqr_spec./err_svd_spec(1+kvec)-1),'m',...
     kvec,100*(err_qlp_spec./err_svd_spec(1+kvec)-1),'g',...
     kvec,100*(err_rutv_spec_p0(:,1)./err_svd_spec(1+kvec)-1),'b',...
     kvec,100*(err_rutv_spec_p0(:,2)./err_svd_spec(1+kvec)-1),'c',...
     kvec,100*(err_rutv_spec_p0(:,3)./err_svd_spec(1+kvec)-1),'r',...
     kvec,100*(err_rutv_spec(:,1)./err_svd_spec(1+kvec)-1),'b:',...
     kvec,100*(err_rutv_spec(:,2)./err_svd_spec(1+kvec)-1),'c:',...
     kvec,100*(err_rutv_spec(:,3)./err_svd_spec(1+kvec)-1),'r:',...
     [1;1]*xlim,[0;max_err_spec]*ones(1,nstep-1),'g:')
title('Relative spectral norm errors in percent')
xlabel('k')
ylabel('Discrepancy in rank-k approximation error in percent.')
axis([0,n,0,max_err_spec])
legend('col piv QR','QLP',...
       'randUTV: q=0,p=0',...
       'randUTV: q=1,p=0',...
       'randUTV: q=2,p=0',...
       sprintf('randUTV: q=0,p=%d',p),...
       sprintf('randUTV: q=1,p=%d',p),...
       sprintf('randUTV: q=2,p=%d',p),'Location','NorthEast')

subplot(1,2,2)
plot(kvec,100*(err_cpqr_frob./err_svd_frob(1+kvec)-1),'m',...
     kvec,100*(err_qlp_frob./err_svd_frob(1+kvec)-1),'g',...
     kvec,100*(err_rutv_frob_p0(:,1)./err_svd_frob(1+kvec)-1),'b',...
     kvec,100*(err_rutv_frob_p0(:,2)./err_svd_frob(1+kvec)-1),'c',...
     kvec,100*(err_rutv_frob_p0(:,3)./err_svd_frob(1+kvec)-1),'r',...
     kvec,100*(err_rutv_frob(:,1)./err_svd_frob(1+kvec)-1),'b:',...
     kvec,100*(err_rutv_frob(:,2)./err_svd_frob(1+kvec)-1),'c:',...
     kvec,100*(err_rutv_frob(:,3)./err_svd_frob(1+kvec)-1),'r:',...
     [1;1]*xlim,[0;max_err_spec]*ones(1,nstep-1),'g:')
title('Relative Frobenius norm errors in percent')
xlabel('k')
axis([0,n,0,max_err_spec])
legend('col piv QR','QLP',...
       'randUTV: q=0,p=0',...
       'randUTV: q=1,p=0',...
       'randUTV: q=2,p=0',...
       sprintf('randUTV: q=0,p=%d',p),...
       sprintf('randUTV: q=1,p=%d',p),...
       sprintf('randUTV: q=2,p=%d',p),'Location','NorthEast')

figure(3)
subplot(1,1,1)
semilogy(1:n,ss,'k',...
         1:n,abs(Rref_diag),'m',...
         1:n,abs(Lref_diag),'g',...
         1:n,-sort(-abs(dd(:,2))),'c',...
         1:n,-sort(-abs(dd(:,3))),'r',...
         1:n,-sort(-abs(dd_p0(:,2))),'c:',...
         1:n,-sort(-abs(dd_p0(:,3))),'r:',...
         [1;1]*xlim,[err_svd_spec(1);err_svd_spec(n)]*ones(1,nstep-1),'g')
title(sprintf('Diagonal entries of R: p=%d',p))
xlabel('k')
ylabel('|R(k,k)|')
legend('svd','col piv QR','QLP',...
       'randUTV: q=1','randUTV: q=2',...
       'randUTV: q=1,p=0','randUTV: q=2,p=0',... 
       'Location','NorthEast')

figure(4)
subplot(1,2,1)
hold off
plot([0,n,n,0,0],[0,0,m,m,0],'k',...
     [1;1]*xlim,[0;m]*ones(1,length(xlim)),'k',...
     [0;n]*ones(1,length(xlim)),[1;1]*xlim,'k')
axis equal
axis ij
axis off
hold on
for i = 1:nstep
  indi = ((i-1)*b+1):min(i*b,m);
  for j = i:nstep
    indj = ((j-1)*b+1):min(j*b,n);
    text((j-0.9)*b,(i-0.5)*b,sprintf('%7.1e',norm(T(indi,indj),'fro')));
  end
end
title('UTV: Frobenius norm of blocks of T')
subplot(1,2,2)
hold off
plot([0,n,n,0,0],[0,0,m,m,0],'k',...
     [1;1]*xlim,[0;m]*ones(1,length(xlim)),'k',...
     [0;n]*ones(1,length(xlim)),[1;1]*xlim,'k')
axis equal
axis ij
axis off
hold on
for i = 1:nstep
  indi = ((i-1)*b+1):min(i*b,m);
  for j = 1:i
    indj = ((j-1)*b+1):min(j*b,n);
    text((j-0.9)*b,(i-0.5)*b,sprintf('%7.1e',norm(Lref(indi,indj),'fro')));
  end
end
title('QLP: Frobenius norm of blocks of L')


keyboard

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This driver routine measures the speed of randUTV
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function LOCAL_timings

%%% Seed the random number generated, and set various parameters:
%rng(0)
b      = 100;            % Step size.
nvec   = 100*(2.^(2:6)); % Matrix sizes.
n_iter = 2;              % Parameter in power iteration.
p      = 0;              % Amount of oversampling to be done.

t_utv     = zeros(1,length(nvec));
t_utv_gpu = zeros(1,length(nvec));
t_qrp     = zeros(1,length(nvec));
t_svd     = zeros(1,length(nvec));
t_svd_gpu = zeros(1,length(nvec));

for i = 1:length(nvec)

   n = nvec(i);
    
   fprintf(1,'Step %2d of %2d.  n = %4d    ',i,length(nvec),n)
   %%% Create a test matrix, and compute its singular values.
   A = LOCAL_fast_decay(n,n,3*n);
   %A = LOCAL_slow_decay(n,n);
   %A = randn(n);
   %A = LOCAL_S_curve(n,n,round(0.5*n),1e-2);

   tic
   [Q,R,P] = qr(A,0);
   t_qrp(i) = toc;
   fprintf(1,'t_qrp = %7.2f    ',t_qrp(i))
   
   tic
   [U,D,V] = svd(A,'econ');
   t_svd(i) = toc;
   tic
   [U,D,V] = svd(gpuArray(A),'econ');
   U = gather(U); V = gather(V); D = gather(D);
   t_svd_gpu(i) = toc;
   fprintf(1,'t_svd = %7.2f    (%7.2f)    ',t_svd(i),t_svd_gpu(i))

   tic
   [U,D,V] = LOCAL_randUTV_econ(A,b,n_iter,p);
   t_utv(i) = toc;
   tic
   [U,D,V] = LOCAL_randUTV_gpu(A,b,n_iter,p);
   t_utv_gpu(i) = toc;
   fprintf(1,'t_utv = %7.2f    (%7.2f)\n',t_utv(i),t_utv_gpu(i))
   
end

figure(1)
subplot(1,2,1)
hold off
loglog(nvec,t_utv,'rx-',...
       nvec,t_utv_gpu,'rx:',...
       nvec,t_svd,'bx-',...
       nvec,t_svd_gpu,'bx:',...
       nvec,t_qrp,'gx-')
xlabel('n')
ylabel('time')
legend('UTV','UTVgpu','SVD','SVDgpu','QRP')
axis tight
title('Absolute times for various factorizations (q=1 in UTV)')

subplot(1,2,2)
hold off
semilogx(nvec,t_utv./(nvec.^3),'rx-',...
         nvec,t_utv_gpu./(nvec.^3),'rx:',...
         nvec,t_svd./(nvec.^3),'bx-',...
         nvec,t_svd_gpu./(nvec.^3),'bx:',...
         nvec,t_qrp./(nvec.^3),'gx-',...
         nvec(1),0)
xlabel('n')
ylabel('time/n^3')
legend('UTV','UTVgpu','SVD','SVDgpu','QRP')
axis tight
title('Times scaled by n^3 for various factorizations (q=1 in UTV)')
keyboard

fprintf(1,'    n      t_qrp      t_svd  t_svd_gpu      t_utv  t_utv_gpu\n')
for i = 1:length(nvec)
  fprintf(1,'%5d  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f\n',...
          nvec(i),t_qrp(i),t_svd(i),t_svd_gpu(i),t_utv(i),t_utv_gpu(i))
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This driver routine measures the speed of randUTV
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function LOCAL_timings_gpu

%%% Seed the random number generated, and set various parameters:
%rng(0)
b          = 100;            % Step size.
%nvec       = 100*(2.^(2:6)); % Matrix sizes.
nvec       = [500,750,1000,1250];
%nvec       = [500,1000,2000,3000,4000,6000,8000,10000,12000,14000,16000];
n_iter_max = 2;              % Parameter in power iteration.
p          = 0;              % Amount of oversampling to be done.

t_utv     = zeros(length(nvec),n_iter_max+1);
t_utv_gpu = zeros(length(nvec),n_iter_max+1);
t_qrp     = zeros(length(nvec),1);
t_svd     = zeros(length(nvec),1);
t_svd_gpu = zeros(length(nvec),1);

fprintf(1,'    n       qrp       svd      Gsvd       q=0      Gq=0       q=1      Gq=1       q=2      Gq=2\n')
for i = 1:length(nvec)

   n = nvec(i);
    
   fprintf(1,'%5d   ',n)
   %%% Create a test matrix, and compute its singular values.
   %A = LOCAL_fast_decay(n,n,3*n);
   %A = LOCAL_slow_decay(n,n);
   A = randn(n);
   %A = LOCAL_S_curve(n,n,round(0.5*n),1e-2);

   tic
   [Q,R,P] = qr(A,0);
   t_qrp(i) = toc;
   fprintf(1,'%7.2f   ',t_qrp(i))
   
   tic
   [U,D,V] = svd(A,'econ');
   t_svd(i) = toc;
   tic
   [U,D,V] = svd(gpuArray(A),'econ');
   U = gather(U); V = gather(V); D = gather(D);
   t_svd_gpu(i) = toc;
   fprintf(1,'%7.2f   %7.2f   ',t_svd(i),t_svd_gpu(i))

   for n_iter = 0:n_iter_max
     tic
     [U,D,V] = LOCAL_randUTV_econ(A,b,n_iter,p);
     t_utv(i,n_iter+1) = toc;
     tic
     [U,D,V] = LOCAL_randUTV_gpu(A,b,n_iter,p);
     t_utv_gpu(i,n_iter+1) = toc;
     fprintf(1,'%7.2f   %7.2f   ',t_utv(i,n_iter+1),t_utv_gpu(i,n_iter+1))
   end
   fprintf(1,'\n')
   
end

keyboard

figure(1)
subplot(1,2,1)
hold off
loglog(nvec,t_utv(:,1),'rx-',...
       nvec,t_utv(:,2),'r+-',...
       nvec,t_utv(:,3),'rd-',...
       nvec,t_svd,'kx-',...
       nvec,t_qrp,'bx-',...
       nvec,t_svd_gpu,'kx:',...
       nvec,t_utv_gpu(:,1),'rx:',...
       nvec,t_utv_gpu(:,2),'r+:',...
       nvec,t_utv_gpu(:,3),'rd:')
xlabel('n')
ylabel('time')
legend('UTV q=0','UTV q=1','UTV q=2','SVD','QRP','SVD (gpu)','UTV q=0 (gpu)','UTV q=1 (gpu)','UTV q=2 (gpu)','Location','NorthWest')
axis tight
title('Absolute times for various factorizations')

subplot(1,2,2)
hold off
plot(nvec,t_utv(:,1)'./(nvec.^3),'rx-',...
     nvec,t_utv(:,2)'./(nvec.^3),'r+-',...
     nvec,t_utv(:,3)'./(nvec.^3),'rd-',...
     nvec,t_svd'./(nvec.^3),'kx-',...
     nvec,t_qrp'./(nvec.^3),'bx-',...
     nvec,t_svd_gpu'./(nvec.^3),'kx:',...
     nvec,t_utv_gpu(:,1)'./(nvec.^3),'rx:',...
     nvec,t_utv_gpu(:,2)'./(nvec.^3),'r+:',...
     nvec,t_utv_gpu(:,3)'./(nvec.^3),'rd:')
xlabel('n')
ylabel('time/n^3')
legend('UTV q=0','UTV q=1','UTV q=2','SVD','QRP','SVD (gpu)','UTV q=0 (gpu)','UTV q=1 (gpu)','UTV q=2 (gpu)','Location','NorthWest')
axis tight
title('Times scaled by n^3 for various factorizations')
keyboard

figure(2)
plot(nvec,t_utv(:,1)./t_utv_gpu(:,1),'rx-',...
     nvec,t_utv(:,2)./t_utv_gpu(:,2),'r+-',...
     nvec,t_utv(:,3)./t_utv_gpu(:,3),'rd-',...
     nvec,t_svd./t_svd_gpu,'kx-')
legend('UTV q=0','UTV q=1','UTV q=2','svd','Location','NorthWest')
axis tight
title('Acceleration factors moving to GPU')

fprintf(1,'    n      t_qrp      t_svd  t_svd_gpu  |               t_utv             |            t_utv_gpu\n')
fprintf(1,'                                        |      q=0        q=1        q=2  |       q=0        q=1        q=2\n')      
for i = 1:length(nvec)
  fprintf(1,'%5d  %9.3f  %9.3f  %9.3f  |',...
          nvec(i),t_qrp(i),t_svd(i),t_svd_gpu(i))
  for n_iter = 0:n_iter_max
    fprintf(1,'%9.3f  ',t_utv(i,n_iter+1))
  end
  fprintf(1,'| ')
  for n_iter = 0:n_iter_max
    fprintf(1,'%9.3f  ',t_utv_gpu(i,n_iter+1))
  end
  fprintf(1,'\n')
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = LOCAL_fast_decay(m,n,k)

% Determine the "inner dimension"
%r = min([m,n,2000]);
r = min([m,n]);

[U,~] = qr(randn(m,r),0);
[V,~] = qr(randn(n,r),0);
alpha = (1e-15)^(1/(k-1));
ss    = alpha.^(0:(r-1));
A     = (U.*(ones(m,1)*ss))*V';

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = LOCAL_slow_decay(m,n)

% Determine the "inner dimension"
r = min([m,n]);

[U,~] = qr(randn(m,r),0);
[V,~] = qr(randn(n,r),0);
ss    = 1./(1:r);
A     = (U.*(ones(m,1)*ss))*V';

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = LOCAL_gap_decay(m,n)

% Determine the "inner dimension"
r = min([m,n]);

[U,~] = qr(randn(m,r),0);
[V,~] = qr(randn(n,r),0);
ss    = 1./(1:r);
ss(61:end) = 0.1*ss(61:end);
A     = (U.*(ones(m,1)*ss))*V';

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = LOCAL_S_curve(m,n,ktarg,acc)

% Determine the "inner dimension"
%r = min([m,n,2000]);
r = min([m,n]);

tt    = linspace(-1,1,ktarg);
uu    = [0.5*(1+tanh(5*tt)),ones(1,r-ktarg)];
ss    = 10.^(log10(acc)*uu);
[U,~] = qr(randn(m,r),0);
[V,~] = qr(randn(n,r),0);
A     = (U.*(ones(m,1)*ss))*V';

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Given an m x n matrix A, this function produces a factorization
%    A = U * T * V'
% where
%    U is an m x m orthonormal matrix
%    T is an m x n upper triangular matrix
%    V is an n x n orthonormal matrix.
% 
% It operators in blocked fashion, processing "b" columns at a time.
%
% There are two tuning parameters:
%    n_iter = parameter in the "power iteration"
%    p      = oversampling parameter 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [T,U,V] = LOCAL_randUTV(A,b,n_iter,p)

%%% Extract matrix dimensions and calculate the number of steps required.
n     = size(A,2);
m     = size(A,1);
nstep = ceil(n/b);

%%% Initialize U and V and copy A onto T.
U = eye(m);
V = eye(n);
T = A;

%%% Process all blocks, except the last.
for j = 1:(nstep-1)
   %%% Partition the index vectors as follows:
   %%%    (1:m) = [J1, J2, I3]
   %%%    (1:n) = [J1, J2, J3]
   %%% where "J2" is the block currently being processed.
   J1 = 1:((j-1)*b);
   J2 = (j-1)*b + (1:b);
   J3 = (j*b+1):n;
   I3 = (j*b+1):m;
   %%% Compute the "sampling" matrix Y.
   Aloc = T([J2,I3],[J2,J3]);
   Y    = Aloc'*randn(m-(j-1)*b,b+p);
   %%% Perform "power iteration" if requested.
   for i_iter = 1:n_iter
      Y = Aloc'*(Aloc*Y);
   end
   %%% If over-sampling is done, then reduce Y to b columns.
   if (p > 0)
      %[~,~,Jtmp] = qr(Y,0);
      %Y          = Y(:,Jtmp(1:b));
      [Utmp,~,~] = svd(Y,'econ');
      Y = Utmp(:,1:b);
   end
   %%% Construct the local transform to be applied "from the right".
   [Vloc,~]     = qr(Y);
   T(:,[J2,J3]) = T(:,[J2,J3])*Vloc;
   V(:,[J2,J3]) = V(:,[J2,J3])*Vloc;
   %%% Next determine the Householder reflectors to be applied "from the left".
   [Uloc,T22]       = qr(T([J2,I3],J2));
   U(:,[J2,I3])     = U(:,[J2,I3])*Uloc;
   T([J2,I3],J3)    = Uloc'*T([J2,I3],J3);
   T(I3,J2)         = zeros(length(I3),length(J2));
   %%% Finally, perform an SVD of the diagonal block.
   [Usvd,Dsvd,Vsvd] = svd(T22(1:b,:));
   T(J2,J2)         = Dsvd;
   T(J1,J2)         = T(J1,J2)*Vsvd;
   T(J2,J3)         = Usvd'*T(J2,J3);
   V(:,J2)          = V(:,J2)*Vsvd;
   U(:,J2)          = U(:,J2)*Usvd;
end

%%% Process the last block.
%%% At this point, no "block pivoting" is required - we simply 
%%% diagonalize the block and update U and V accordingly.
%J1               = 1:(n-b);
J1               = 1:(b*(nstep-1));
J2               = (b*(nstep-1)+1):n;
I2               = (b*(nstep-1)+1):m;
[Usvd,Dsvd,Vsvd] = svd(T(I2,J2));
T(I2,J2)         = Dsvd;
T(J1,J2)         = T(J1,J2)*Vsvd;
U( :,I2)         = U(:,I2)*Usvd;
V( :,J2)         = V(:,J2)*Vsvd;

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Given an m x n matrix A, this function produces a factorization
%    A = U * T * V'
% where
%    U is an m x m orthonormal matrix
%    T is an m x n upper triangular matrix
%    V is an n x n orthonormal matrix.
% 
% It operators in blocked fashion, processing "b" columns at a time.
%
% There are two tuning parameters:
%    n_iter = parameter in the "power iteration"
%    p      = oversampling parameter 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [T,U,V] = LOCAL_randUTV_econ(A,b,n_iter,p)

%%% Extract matrix dimensions and calculate the number of steps required.
n     = size(A,2);
m     = size(A,1);
nstep = ceil(n/b);

%%% Initialize U and V and copy A onto T.
U = eye(m);
V = eye(n);
T = A;

%%% Process all blocks, except the last.
for j = 1:(nstep-1)
   %%% Partition the index vectors as follows:
   %%%    (1:m) = [J1, J2, I3]
   %%%    (1:n) = [J1, J2, J3]
   %%% where "J2" is the block currently being processed.
   J1 = 1:((j-1)*b);
   J2 = (j-1)*b + (1:b);
   J3 = (j*b+1):n;
   I3 = (j*b+1):m;
   %%% Compute the "sampling" matrix Y.
   Aloc = T([J2,I3],[J2,J3]);
   Y    = Aloc'*randn(m-(j-1)*b,b+p);
   %%% Perform "power iteration" if requested.
   for i_iter = 1:n_iter
      Y = Aloc'*(Aloc*Y);
   end
   %%% If over-sampling is done, then reduce Y to b columns.
   if (p > 0)
      %[~,~,Jtmp] = qr(Y,0);
      %Y = Y(:,Jtmp(1:b));
      [Utmp,~,~] = svd(Y,'econ');
      Y = Utmp(:,1:b);
   end
   %%% Perform b steps of Householder QR on the Y matrix,
   %%% and then apply the Householder reflections "from the right".
   [~,Vloc_U,Vloc_TU] = LOCAL_dgeqrf_modified(Y);
   T(:,[J2,J3]) = T(:,[J2,J3]) - (T(:,[J2,J3])*Vloc_U)*Vloc_TU;
   V(:,[J2,J3]) = V(:,[J2,J3]) - (V(:,[J2,J3])*Vloc_U)*Vloc_TU;
   %%% Next determine the rotations to be applied "from the left".
   [T22,Uloc_U,Uloc_TU] = LOCAL_dgeqrf_modified(T([J2,I3],J2));
   U(:,[J2,I3])        = U(:,[J2,I3]) - (U(:,[J2,I3])*Uloc_U)*Uloc_TU;
   T([J2,I3],J3)       = T([J2,I3],J3) - Uloc_TU'*(Uloc_U'*T([J2,I3],J3));
   T(I3,J2)            = zeros(length(I3),length(J2));
   %%% Perform an SVD of the diagonal block and update T, U, V, accordingly.
   [Usvd,Dsvd,Vsvd] = svd(T22);
   T(J2,J2)         = Dsvd;
   T(J1,J2)         = T(J1,J2)*Vsvd;
   T(J2,J3)         = Usvd'*T(J2,J3);
   V(:,J2)          = V(:,J2)*Vsvd;
   U(:,J2)          = U(:,J2)*Usvd;
end

%%% Process the last block.
%%% At this point, no "block pivoting" is required - we simply 
%%% diagonalize the block and update U and V accordingly.
%J1               = 1:(n-b);
J1               = 1:(b*(nstep-1));
J2               = (b*(nstep-1)+1):n;
I2               = (b*(nstep-1)+1):m;
[Usvd,Dsvd,Vsvd] = svd(T(I2,J2));
T(I2,J2)         = Dsvd;
T(J1,J2)         = T(J1,J2)*Vsvd;
U( :,I2)         = U(:,I2)*Usvd;
V( :,J2)         = V(:,J2)*Vsvd;

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Given an m x n matrix A, this function produces a factorization
%    A = U * T * V'
% where
%    U is an m x m orthonormal matrix
%    T is an m x n upper triangular matrix
%    V is an n x n orthonormal matrix.
% 
% It operators in blocked fashion, processing "b" columns at a time.
%
% There are two tuning parameters:
%    n_iter = parameter in the "power iteration"
%    p      = oversampling parameter 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [T,U,V] = LOCAL_randUTV_gpu(A,b,n_iter,p)

%%% Extract matrix dimensions and calculate the number of steps required.
n     = size(A,2);
m     = size(A,1);
nstep = ceil(n/b);

%%% Initialize U and V and copy A onto T.
U = gpuArray(eye(m));
V = gpuArray(eye(n));
T = gpuArray(A);

%%% Process all blocks, except the last.
for j = 1:(nstep-1)
   %%% Partition the index vectors as follows:
   %%%    (1:m) = [J1, J2, I3]
   %%%    (1:n) = [J1, J2, J3]
   %%% where "J2" is the block currently being processed.
   J1 = 1:((j-1)*b);
   J2 = (j-1)*b + (1:b);
   J3 = (j*b+1):n;
   I3 = (j*b+1):m;
   %%% Compute the "sampling" matrix Y.
   Aloc = T([J2,I3],[J2,J3]);
   Y    = Aloc'*randn(m-(j-1)*b,b+p);
   %%% Perform "power iteration" if requested.
   for i_iter = 1:n_iter
      Y = Aloc'*(Aloc*Y);
   end
   %%% If over-sampling is done, then reduce Y to b columns.
   if (p > 0)
      %[~,~,Jtmp] = qr(Y,0);
      %Y = Y(:,Jtmp(1:b));
      [Utmp,~,~] = svd(Y,'econ');
      Y = Utmp(:,1:b);
   end
   %%% Perform b steps of Householder QR on the Y matrix,
   %%% and then apply the Householder reflections "from the right".
   [~,Vloc_U,Vloc_TU] = LOCAL_dgeqrf_modified_gpu(Y);
   T(:,[J2,J3]) = T(:,[J2,J3]) - (T(:,[J2,J3])*Vloc_U)*Vloc_TU;
   V(:,[J2,J3]) = V(:,[J2,J3]) - (V(:,[J2,J3])*Vloc_U)*Vloc_TU;
   %%% Next determine the rotations to be applied "from the left".
   [T22,Uloc_U,Uloc_TU] = LOCAL_dgeqrf_modified_gpu(T([J2,I3],J2));
   U(:,[J2,I3])        = U(:,[J2,I3]) - (U(:,[J2,I3])*Uloc_U)*Uloc_TU;
   T([J2,I3],J3)       = T([J2,I3],J3) - Uloc_TU'*(Uloc_U'*T([J2,I3],J3));
   T(I3,J2)            = zeros(length(I3),length(J2));
   %%% Perform an SVD of the diagonal block and update T, U, V, accordingly.
   [Usvd,Dsvd,Vsvd] = svd(T22);
   T(J2,J2)         = Dsvd;
   T(J1,J2)         = T(J1,J2)*Vsvd;
   T(J2,J3)         = Usvd'*T(J2,J3);
   V(:,J2)          = V(:,J2)*Vsvd;
   U(:,J2)          = U(:,J2)*Usvd;
end

%%% Process the last block.
%%% At this point, no "block pivoting" is required - we simply 
%%% diagonalize the block and update U and V accordingly.
%J1               = 1:(n-b);
J1               = 1:(b*(nstep-1));
J2               = (b*(nstep-1)+1):n;
I2               = (b*(nstep-1)+1):m;
[Usvd,Dsvd,Vsvd] = svd(T(I2,J2));
T(I2,J2)         = Dsvd;
T(J1,J2)         = T(J1,J2)*Vsvd;
U( :,I2)         = U(:,I2)*Usvd;
V( :,J2)         = V(:,J2)*Vsvd;

U = gather(U);
V = gather(V);
T = gather(T);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Q,R] = LOCAL_nonpiv_QR(A,k)

m = size(A,1);
n = size(A,2);
if (nargin==1)
  k = min([m-1,n]);
end

Q = eye(m);
R = A;
for i = 1:k
  alpha = -sign(R(i,i))*norm(R(i:m,i));
  r     = sqrt(0.5*alpha*(alpha - R(i,i)));
  v     = [zeros(i-1,1);(R(i,i) - alpha)/(2*r); (1/(2*r))*R((i+1):m,i)];
  R     = R - 2*v*(v'*R);
  Q     = Q - 2*(Q*v)*v';
  R((i+1):end,i) = zeros(m-i,1);
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [U,T,TUt] = LOCAL_UT_householder(X)

n = size(X,2);

%%% Build the U matrix holding the Householder vectors.
U          = tril(X,-1);
U(1:n,1:n) = U(1:n,1:n) + eye(n);

%%% Build the "T" matrix.
T = triu(U'*U,1) + diag(0.5*sum(U.*U,1));

%%% Build TUt = inv(T)*trans(U).
TUt = T\U';

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [R,U,TU] = LOCAL_dgeqrf_modified(Y)

n = size(Y,2);

%%% Apply dgeqrf to Y.
R = qr(Y);

%%% Build the U matrix holding the Householder vectors.
U          = tril(R,-1);
U(1:n,1:n) = U(1:n,1:n) + eye(n);

%%% Build the "T" matrix.
T = triu(U'*U,1) + diag(0.5*sum(U.*U,1));

%%% Build TU = inv(T)*trans(U).
TU = T\U';

R = triu(R(1:n,:));

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [R,U,TU] = LOCAL_dgeqrf_modified_gpu(Y)

n = size(Y,2);

%%% Apply dgeqrf to Y.
R = gpuArray(qr(gather(Y)));

%%% Build the U matrix holding the Householder vectors.
U          = tril(R,-1);
U(1:n,1:n) = U(1:n,1:n) + eye(n);

%%% Build the "T" matrix.
T = triu(U'*U,1) + diag(0.5*sum(U.*U,1));

%%% Build TU = inv(T)*trans(U).
TU = T\U';

R = triu(R(1:n,:));

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This driver function calls randUTV for several different parameter values
% and compares the results in a single plot.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function LOCAL_plotdiags

%%% Seed the random number generated, and set various parameters:
rng(0)
b      = 50;      % Step size.
n      = 2000;     % Matrix size.
m      = 2000;     % Matrix size.
n_iter = 0; 
p      = 0;      % Amount of oversampling to be done.

tic
%%% Fast decay
A  = LOCAL_fast_decay(m,n,3*n);
ss = svd(A);
[~,R,~] = qr(A,0);
rr = abs(diag(R));
[T,~,~] = LOCAL_randUTV_econ(A,b,n_iter,p);
tt = -sort(-abs(diag(T)));
figure(1)
subplot(2,2,1)
semilogy(1:n,ss,'k',...
         1:n,rr,'b',...
         1:n,tt,'r')
legend('Exact singular values','CPQR',sprintf('randUTV with q=%d, p=%d',n_iter,p))
xlabel('k')
ylabel('Diagonal entries')
axis tight
%title(sprintf('Fast decay. q=%d  p=%d',n_iter,p))
title(sprintf('Fast decay'))

%%% Slow decay
A  = LOCAL_slow_decay(m,n);
ss = svd(A);
[~,R,~] = qr(A,0);
rr = abs(diag(R));
[T,~,~] = LOCAL_randUTV_econ(A,b,n_iter,p);
tt = -sort(-abs(diag(T)));
figure(1)
subplot(2,2,2)
semilogy(1:n,ss,'k',...
         1:n,rr,'b',...
         1:n,tt,'r')
%legend('svd','qr','utv')
legend('Exact singular values','CPQR',sprintf('randUTV with q=%d, p=%d',n_iter,p))
xlabel('k')
ylabel('Diagonal entries')
axis tight
%title(sprintf('Slow decay. q=%d  p=%d',n_iter,p))
title(sprintf('Slow decay'))

%%% S shape
A = LOCAL_S_curve(m,n,0.5*n,1e-2);
ss = svd(A);
[~,R,~] = qr(A,0);
rr = abs(diag(R));
[T,~,~] = LOCAL_randUTV_econ(A,b,n_iter,p);
tt = -sort(-abs(diag(T)));
figure(1)
subplot(2,2,3)
semilogy(1:n,ss,'k',...
         1:n,rr,'b',...
         1:n,tt,'r')
%legend('svd','qr','utv')
legend('Exact singular values','CPQR',sprintf('randUTV with q=%d, p=%d',n_iter,p))
xlabel('k')
ylabel('Diagonal entries')
axis tight
%title(sprintf('S-shaped decay. q=%d  p=%d',n_iter,p))
title(sprintf('S-shaped decay'))

%%% Slow decay with gap
A  = LOCAL_gap_decay(m,n);
ss = svd(A);
[~,R,~] = qr(A,0);
rr = abs(diag(R));
[T,~,~] = LOCAL_randUTV_econ(A,b,n_iter,p);
tt = -sort(-abs(diag(T)));
figure(1)
subplot(2,2,4)
semilogy(1:n,ss,'k',...
         1:n,rr,'b',...
         1:n,tt,'r')
%legend('svd','qr','utv')
axis([0,100,1e-4,1])
legend('Exact singular values','CPQR',sprintf('randUTV with q=%d, p=%d',n_iter,p))
xlabel('k')
ylabel('Diagonal entries')
%title(sprintf('Slow decay with gap. q=%d  p=%d',n_iter,p))
title(sprintf('Slow decay with gap'))
toc

keyboard

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes a UTV factorization using a method proposed by
% G.W. Stewart in the 1999 SISC paper: 
%    "The QLP Approximation to the Singular Value Decomposition". 
%    http://epubs.siam.org/doi/abs/10.1137/S1064827597319519
% The idea is very simple. First compute a CPQR factorization where we
% orthonormalize the COLUMNS of A via Gram-Schmidt:
%    A = Q1*R1*trans(P1).
% Then compute a CPQR factorization of trans(R1)
%    trans(A) = Q2*R2*trans(P2).
% In other words, perform G-S on the ROWS of R1. This results in a
% factorization
%    A = Q1*P2*trans(R2)*trans(Q2)*trans(P1).
% Now define
%    U = Q1*P2,
%    V = P1*Q2,
%    L = trans(R2),
% to obtain the factorization
%    A = U*L*trans(V),
% where U and V are orthonormal, and L is lower triangular.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [U,L,V] = LOCAL_QLP(A)

[Q1,R1,J1] = qr(A,0);
[Q2,R2,J2] = qr(R1',0);
U          = Q1(:,J2);
V          = zeros(size(Q2));
V(J1,:)    = Q2;
L          = R2';

return
