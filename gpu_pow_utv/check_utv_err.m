function [] = check_utv_err()

clear

% read in matrices A,U,T,V from file "check_utv_err_dat.m"
check_utv_err_dat

n = size(A,1);

%%% Perform factorizations:
ss = svd(A);

[U2,T2,V2] = qr(A);         % Built-in LAPACK.

b = 50;
p = 0;
[U3,T3,V3] = randUTV_econ(A,b,p,q);        % randomized UTV

%%% Perform the most basic error checks:
fprintf(1,'                     gpu powerURV  matlab CPQR    matlab randUTV\n')
fprintf(1,'||A*V - U*T|| (fro) = %12.5e    %12.5e    %12.5e\n',...
        norm(A*V1 - U1*T1,'fro'),...
        norm(A*V2 - U2*T2,'fro'),...
        norm(A*V3 - U3*T3,'fro'))
fprintf(1,'||A*V - U*T|| (op)  = %12.5e    %12.5e    %12.5e\n',...
        norm(A*V1 - U1*T1),...
        norm(A*V2 - U2*T2),...
        norm(A*V3 - U3*T3))
fprintf(1,'max|trans(Q)*Q - I| = %12.5e    %12.5e    %12.5e\n',...
        max(max(abs(U1'*U1 - eye(n)))),...
        max(max(abs(U2'*U2 - eye(n)))),...
        max(max(abs(U3'*U3 - eye(n)))))

fprintf('Computing spectral norm errors\n')

%%% Compute spectral norm errors
err1_gpu_utv = [ss(1),zeros(1,n-1)];
err1_qr    = [ss(1),zeros(1,n-1)];
err1_matlab_utv = [ss(1),zeros(1,n-1)];

for i = 1:(n-1)
  if mod(i,100) == 0
      i
  end
  err1_gpu_utv(i+1) = norm(T1((i+1):n,(i+1):n));
  err1_qr(i+1)    = norm( T2((i+1):n,(i+1):n));
  err1_matlab_utv(i+1) = norm( T3((i+1):n,(i+1):n));
end

fprintf('Computing Frobenius norm errors\n')
%%% Compute Frobenius norm errors
err2_svd     = [norm(A,'fro'),zeros(1,n-1)];
err2_gpu_utv = [norm(A,'fro'),zeros(1,n-1)];
err2_qr    = [norm(A,'fro'),zeros(1,n-1)];
err2_matlab_utv = [norm(A,'fro'),zeros(1,n-1)];

for i = 1:(n-1)
  if mod(i,100) == 0
      i
  end
  err2_svd(i+1)     = norm(ss((i+1):n));
  err2_gpu_utv(i+1) = norm(T1((i+1):n,(i+1):n),'fro');
  err2_qr(i+1)    = norm( T2((i+1):n,(i+1):n),'fro');
  err2_matlab_utv(i+1) = norm( T3((i+1):n,(i+1):n),'fro');
end

%%% Plot errors
figure(1)
subplot(1,2,1)
semilogy(0:(n-1),ss,'k',...
         0:(n-1),err1_qr,'g',...
         0:(n-1),err1_gpu_utv,'r',...
         0:(n-1),err1_matlab_utv,'b');
legend('SVD','CPQR','gpu UTV','matlab UTV');
ylabel('e_k = ||A - A_k||')
xlabel('k')
title('Operator norm','FontWeight','normal');

subplot(1,2,2)
semilogy(0:(n-1),err2_svd,'k',...
         0:(n-1),err2_qr,'g',...
         0:(n-1),err2_gpu_utv,'r',...
         0:(n-1),err2_matlab_utv,'b')
legend('SVD','CPQR','gpu UTV','matlab UTV');
ylabel('e_k = ||A - A_k||')
xlabel('k')
title('Frobenius norm','FontWeight','normal');
%export_fig('fig_errors_fast.pdf','-pdf','-trans')

%%% Plot relative error
figure(2)
subplot(1,2,1)
semilogy(0:(n-1),abs(ss-err1_qr')./ss,'g',...
         0:(n-1),abs(ss-err1_gpu_utv')./ss,'r',...
         0:(n-1),abs(ss-err1_matlab_utv')./ss,'b');
legend('CPQR','gpu UTV','matlab UTV','Location','south')
ylabel('e_k = ||A - A_k||')
xlabel('k')
title('Operator norm','FontWeight','normal')
ylim([1e-16 1e2])

subplot(1,2,2)
semilogy(0:(n-1),abs(err2_svd-err2_qr)./err2_svd,'g',...
         0:(n-1),abs(err2_svd-err2_gpu_utv)./err2_svd,'r',...
         0:(n-1),abs(err2_svd-err2_matlab_utv)./err2_svd,'b')
legend('CPQR','gpu UTV','matlab UTV','Location','south')
ylabel('e_k = ||A - A_k||')
xlabel('k')
title('Frobenius norm','FontWeight','normal')
ylim([1e-16 1e2])

% save(type)

end


function [U,T,V] = randUTV_econ(A,b,p,n_iter)


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
   if j == 1 || p == 0
       Y    = Aloc'*randn(m-(j-1)*b,b+p);
   else
       %Y    = Aloc'*randn(m-(j-1)*b,b+p);
       Y    = Aloc'*randn(m-(j-1)*b,b);
   end
   
   %%% Perform "power iteration" if requested.
   for i_iter = 1:n_iter
      
       if (p > 0 && i_iter == n_iter)
           if (j > 1)
               Y = Y - Ytmp*Ytmp'*Y;
               [Y,~] = qr(Y,0);
               Y = [Ytmp Y];
           else
               [Y,~] = qr(Y,0);
           end
       end
       
       Y = Aloc'*(Aloc*Y);
   end
   
   %%% If over-sampling is done, then reduce Y to b columns.  
   if (p > 0)
       [Utmp,~,~] = svd(Y,'econ');
       Y = Utmp(:,1:b);
       Ytmp = Utmp(:,(b+1):end);
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
   
   % test: "kickstarting" next sample
   if p > 0
       % Ytmp = Vloc(:,b+1:end)'*Ytmp;
       Ytmp = Ytmp(b+1:end,:) - Vloc_TU(:,b+1:end)'*(Vloc_U'*Ytmp);
   end
   
end

%%% Process the last block.
%%% At this point, no "block pivoting" is required - we simply 
%%% diagonalize the block and update U and V accordingly.
J1               = 1:(b*(nstep-1));
J2               = (b*(nstep-1)+1):n;
I2               = (b*(nstep-1)+1):m;
[Usvd,Dsvd,Vsvd] = svd(T(I2,J2));
T(I2,J2)         = Dsvd;
T(J1,J2)         = T(J1,J2)*Vsvd;
U( :,I2)         = U(:,I2)*Usvd;
V( :,J2)         = V(:,J2)*Vsvd;

end

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

end


