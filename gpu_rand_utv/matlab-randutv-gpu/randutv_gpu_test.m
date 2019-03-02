function [] = rand_utv_test()

clear

m = 400;
n = 400; % matrix size
b = 50; % block size
p = 50; % oversampling parameter
q = 1; % power iteration parameter
type = 'fast'; % 'fast' for matrix with quickly decaying SVs
               % 'slow' for matrix with slowly decaying SVs
               % 's_curve' for matrix whose SVs decay quickly at first,
               %            then slowly at the end
               % 'hilbert' a hilbert matrix
               % 'BIE' is a PDE solution matrix
               % 'vander' is the Vandermonde matrix
               % 'gap' for matrix with an obvious gap
               

%DRIVER_error_plots(type,m,n,b,p,q);
%DRIVER_error_plots_OS_and_q(type,m,n,b,p,q);
%DRIVER_quick_error_tests(type,m,n,b,p,q);
DRIVER_time_plots(type,b,p,q);


end

function [] = DRIVER_quick_error_tests(type,m,n,b,p,q)

%%% Seed the random number generator, and set various parameters:
rng('default')
rng(0)

%%% Create a test matrix. 
switch type
    case 'fast'
        A = LOCAL_fast_decay(m,n,1e-5);
    case 'slow'
        A = LOCAL_slow_decay(m,n);
    case 's_curve'
        A = LOCAL_S_curve(m,n,round(0.5*n),1e-2);
    case 'hilbert'
        A = hilb(n);
    case 'BIE'
        A = generate_BIE(n);
    case 'vander'
        A = vander(linspace(0,1,n));
    case 'gap'
        A = LOCAL_gap(m,n);
end


%%% Normalize the test matrix so it has spectral norm 1.
ss = svd(A);
A  = A/ss(1);
ss = ss/ss(1);

%%% Perform factorizations:
[U1,T1,V1] = qr(A);         % Built-in LAPACK.
[U2,T2,V2] = randUTV_econ(A,b,0,q);
[U3,T3,V3] = randUTV_econ(A,b,p,q);        % randomized UTV

%%% Perform some accuracy checks:
fprintf('|| D - diag(T) ||_F = %e \n', norm(ss - diag(T2),'fro'));
fprintf('|| T - diag(diag(T))||_F = %e \n', norm(T2 - diag(diag(T2)),'fro') );


end

function [] = DRIVER_error_plots(type,m,n,b,p,q)

%%% Seed the random number generator, and set various parameters:
%rng('default')
%rng(0)

%%% Create a test matrix. 
switch type
    case 'fast'
        A = LOCAL_fast_decay(m,n,1e-5);
    case 'slow'
        A = LOCAL_slow_decay(m,n);
    case 's_curve'
        A = LOCAL_S_curve(m,n,round(0.5*n),1e-2);
    case 'hilbert'
        A = hilb(n);
    case 'BIE'
        A = generate_BIE(n);
    case 'vander'
        A = vander(linspace(0,1,n));
    case 'gap'
        A = LOCAL_gap(m,n);
end


%%% Normalize the test matrix so it has spectral norm 1.
ss = svd(A);
A  = A/ss(1);
ss = ss/ss(1);

%%% Perform factorizations:
[U1,T1,V1] = qr(A);         % Built-in LAPACK.

tic;
[U2,T2,V2] = randUTV_econ(A,b,p,q);        % randomized UTV
t2 = toc;

fprintf(1,'time for randUTV factorization: %.3e sec \n \n',t2);

[U3,T3,V3] = randUTV_econ(A,b,0,q);         % randomized UTV with no oversampling, no ON

[U4,T4,V4] = powerURV(A,q);                      % the power URV factorization

%%% Perform the most basic error checks:
fprintf(1,'                        LAPACK QR        randUTV      randUTVupdate    powerURV\n')
fprintf(1,'||A*V - U*T|| (fro) = %12.5e    %12.5e    %12.5e   %12.5e\n',...
        norm(A*V1 - U1*T1,'fro'),...
        norm(A*V2 - U2*T2,'fro'),...
        norm(A*V3 - U3*T3,'fro'),...
        norm(A*V4 - U4*T4,'fro'))
fprintf(1,'||A*V - U*T|| (op)  = %12.5e    %12.5e    %12.5e   %12.5e\n',...
        norm(A*V1 - U1*T1),...
        norm(A*V2 - U2*T2),...
        norm(A*V3 - U3*T3),...
        norm(A*V4 - U4*T4))
fprintf(1,'max|trans(Q)*Q - I| = %12.5e    %12.5e    %12.5e   %12.5e\n',...
        max(max(abs(U1'*U1 - eye(n)))),...
        max(max(abs(U2'*U2 - eye(n)))),...
        max(max(abs(U3'*U3 - eye(n)))),...
        max(max(abs(U4'*U4 - eye(n)))))
    
fprintf('Computing spectral norm errors\n')

%%% Compute spectral norm errors
err1_qr = [ss(1),zeros(1,n-1)];
err1_rand    = [ss(1),zeros(1,n-1)];
err1_rand_def = [ss(1),zeros(1,n-1)];
err1_pow  = [ss(1),zeros(1,n-1)];

for i = 1:(n-1)
  if mod(i,100) == 0
      i
  end
  err1_qr(i+1) = norm(T1((i+1):n,(i+1):n));
  err1_rand(i+1)    = norm( T2((i+1):n,(i+1):n));
  err1_rand_def(i+1) = norm( T3((i+1):n,(i+1):n));
  err1_pow(i+1)  = norm(T4((i+1):n,(i+1):n));
end

fprintf('Computing Frobenius norm errors\n')
%%% Compute Frobenius norm errors
err2_svd     = [norm(A,'fro'),zeros(1,n-1)];
err2_qr = [norm(A,'fro'),zeros(1,n-1)];
err2_rand    = [norm(A,'fro'),zeros(1,n-1)];
err2_rand_def = [norm(A,'fro'),zeros(1,n-1)];
err2_pow = [norm(A,'fro'),zeros(1,n-1)];

for i = 1:(n-1)
  if mod(i,100) == 0
      i
  end
  err2_svd(i+1)     = norm(ss((i+1):n));
  err2_qr(i+1) = norm(T1((i+1):n,(i+1):n),'fro');
  err2_rand(i+1)    = norm( T2((i+1):n,(i+1):n),'fro');
  err2_rand_def(i+1) = norm( T3((i+1):n,(i+1):n),'fro');
  err2_pow(i+1) = norm( T4((i+1):n,(i+1):n),'fro');
end

%%% Plot errors
figure(1)
subplot(1,2,1)
semilogy(0:(n-1),ss,'k',...
         0:(n-1),err1_qr,'g',...
         0:(n-1),err1_rand_def,'b',...
         0:(n-1),err1_rand,'b--',...
         0:(n-1),err1_pow,'r')
legend('svd','cpqr',strcat('randUTV, p=0, q=',num2str(q)),...
    strcat('randUTV, p=',num2str(p),', q=',num2str(q)),...
    strcat('powerURV, q=',num2str(q)))
ylabel('||A - A_k||')
xlabel('k')
title('Operator norm','FontWeight','normal');

subplot(1,2,2)
semilogy(0:(n-1),err2_svd,'k',...
         0:(n-1),err2_qr,'g',...
         0:(n-1),err2_rand_def,'b',...
         0:(n-1),err2_rand,'b--',...
         0:(n-1),err2_pow,'r')
legend('svd','cpqr',strcat('randUTV, p=0, q=',num2str(q)),...
    strcat('randUTV, p=',num2str(p),', q=',num2str(q)),...
    strcat('powerURV, q=',num2str(q)))
ylabel('||A - A_k||')
xlabel('k')
title('Frobenius norm','FontWeight','normal');
%export_fig('fig_errors_fast.pdf','-pdf','-trans')

%%% Plot relative error
figure(2)
subplot(1,2,1)
semilogy(0:(n-1),abs(ss-err1_qr')./ss,'g',...
         0:(n-1),abs(ss-err1_rand_def')./ss,'b',...
         0:(n-1),abs(ss-err1_rand')./ss,'b--',...
         0:(n-1),abs(ss-err1_pow')./ss,'r')
legend('cpqr',strcat('randUTV, p=0, q=',num2str(q)),strcat('randUTV, p=',num2str(p),', q=',num2str(q)),...
       strcat('powerURV, q=',num2str(q)),'Location','south')
ylabel('||A - A_k|| / ||A - A_k^{optimal}||')
xlabel('k')
title('Operator norm','FontWeight','normal')
ylim([1e-16 1e2])

subplot(1,2,2)
semilogy(0:(n-1),abs(err2_svd-err2_qr)./err2_svd,'g',...
         0:(n-1),abs(err2_svd-err2_rand_def)./err2_svd,'b',...
         0:(n-1),abs(err2_svd-err2_rand)./err2_svd,'b--',...
         0:(n-1),abs(err2_svd-err2_pow)./err2_svd,'r')
legend('cpqr',strcat('randUTV, p=0, q=',num2str(q)),strcat('randUTV, p=',num2str(p),', q=',num2str(q)),...
       strcat('powerURV, q=',num2str(q)),'Location','south')
ylabel('||A - A_k|| / ||A - A_k^{optimal}||')
xlabel('k')
title('Frobenius norm','FontWeight','normal')
ylim([1e-16 1e2])

% save(type)

end

function [] = DRIVER_error_plots_OS_and_q(type,m,n,b,p,q)

%%% Seed the random number generator, and set various parameters:
rng('default')
rng(0)

%%% Create a test matrix. 
switch type
    case 'fast'
        A = LOCAL_fast_decay(m,n,1e-5);
    case 'slow'
        A = LOCAL_slow_decay(m,n);
    case 's_curve'
        A = LOCAL_S_curve(m,n,round(0.5*n),1e-2);
    case 'hilbert'
        A = hilb(n);
    case 'BIE'
        A = generate_BIE(n);
    case 'vander'
        A = vander(linspace(0,1,n));
    case 'gap'
        A = LOCAL_gap(m,n);
end


%%% Normalize the test matrix so it has spectral norm 1.
ss = svd(A);
A  = A/ss(1);
ss = ss/ss(1);

%%% Perform factorizations:

% q = 2
q = 2;
[U3,T3,V3] = randUTV_econ(A,b,0,q);
[U4,T4,V4] = randUTV_econ(A,b,10,q);
[U5,T5,V5] = randUTV_econ(A,b,b,q);

% q = 3
q = 3;
[U6,T6,V6] = randUTV_econ(A,b,0,q);
[U7,T7,V7] = randUTV_econ(A,b,10,q);
[U8,T8,V8] = randUTV_econ(A,b,b,q);

% q = 4
q = 4;
[U9,T9,V9] = randUTV_econ(A,b,0,q);
[U10,T10,V10] = randUTV_econ(A,b,10,q);
[U11,T11,V11] = randUTV_econ(A,b,b,q);
        
fprintf('Computing spectral norm errors\n')

%%% Compute spectral norm errors
err1_rand_3 = [ss(1),zeros(1,n-1)];
err1_rand_4 = [ss(1),zeros(1,n-1)];
err1_rand_5 = [ss(1),zeros(1,n-1)];
err1_rand_6 = [ss(1),zeros(1,n-1)];
err1_rand_7 = [ss(1),zeros(1,n-1)];
err1_rand_8 = [ss(1),zeros(1,n-1)];
err1_rand_9 = [ss(1),zeros(1,n-1)];
err1_rand_10 = [ss(1),zeros(1,n-1)];
err1_rand_11 = [ss(1),zeros(1,n-1)];

for i = 1:(n-1)
  if mod(i,100) == 0
      i
  end
  err1_rand_3(i+1)    = norm( T3((i+1):n,(i+1):n));
  err1_rand_4(i+1)    = norm( T4((i+1):n,(i+1):n));
  err1_rand_5(i+1)    = norm( T5((i+1):n,(i+1):n));
  err1_rand_6(i+1)    = norm( T6((i+1):n,(i+1):n));
  err1_rand_7(i+1)    = norm( T7((i+1):n,(i+1):n));
  err1_rand_8(i+1)    = norm( T8((i+1):n,(i+1):n));
  err1_rand_9(i+1)    = norm( T9((i+1):n,(i+1):n));
  err1_rand_10(i+1)    = norm( T10((i+1):n,(i+1):n));
  err1_rand_11(i+1)    = norm( T11((i+1):n,(i+1):n));

end

fprintf('Computing Frobenius norm errors\n')
%%% Compute Frobenius norm errors
err2_svd     = [norm(A,'fro'),zeros(1,n-1)];

err2_rand_3    = [norm(A,'fro'),zeros(1,n-1)];
err2_rand_4    = [norm(A,'fro'),zeros(1,n-1)];
err2_rand_5    = [norm(A,'fro'),zeros(1,n-1)];
err2_rand_6    = [norm(A,'fro'),zeros(1,n-1)];
err2_rand_7    = [norm(A,'fro'),zeros(1,n-1)];
err2_rand_8    = [norm(A,'fro'),zeros(1,n-1)];
err2_rand_9    = [norm(A,'fro'),zeros(1,n-1)];
err2_rand_10    = [norm(A,'fro'),zeros(1,n-1)];
err2_rand_11    = [norm(A,'fro'),zeros(1,n-1)];


for i = 1:(n-1)
  if mod(i,100) == 0
      i
  end
  err2_svd(i+1)     = norm(ss((i+1):n));
  err2_rand_3(i+1)    = norm( T3((i+1):n,(i+1):n),'fro');
  err2_rand_4(i+1)    = norm( T4((i+1):n,(i+1):n),'fro');
  err2_rand_5(i+1)    = norm( T5((i+1):n,(i+1):n),'fro');
  err2_rand_6(i+1)    = norm( T6((i+1):n,(i+1):n),'fro');
  err2_rand_7(i+1)    = norm( T7((i+1):n,(i+1):n),'fro');
  err2_rand_8(i+1)    = norm( T8((i+1):n,(i+1):n),'fro');
  err2_rand_9(i+1)    = norm( T9((i+1):n,(i+1):n),'fro');
  err2_rand_10(i+1)    = norm( T10((i+1):n,(i+1):n),'fro');
  err2_rand_11(i+1)    = norm( T11((i+1):n,(i+1):n),'fro');

end

%%% Plot relative error
figure(1)
subplot(1,2,1)
semilogy(0:(n-1),abs(ss-err1_rand_3')./ss,'g',...
         0:(n-1),abs(ss-err1_rand_4')./ss,'g',...
         0:(n-1),abs(ss-err1_rand_5')./ss,'g',...
         0:(n-1),abs(ss-err1_rand_6')./ss,'b',...
         0:(n-1),abs(ss-err1_rand_7')./ss,'b',...
         0:(n-1),abs(ss-err1_rand_8')./ss,'b',...
         0:(n-1),abs(ss-err1_rand_9')./ss,'r',...
         0:(n-1),abs(ss-err1_rand_10')./ss,'r',...
         0:(n-1),abs(ss-err1_rand_11')./ss,'r')
ylabel('e_k = ||A - A_k||')
xlabel('k')
title('Operator norm','FontWeight','normal')
ylim([1e-16 1e2])

subplot(1,2,2)
semilogy(0:(n-1),abs(err2_svd-err2_rand_3)./err2_svd,'g',...
         0:(n-1),abs(err2_svd-err2_rand_4)./err2_svd,'g',...
         0:(n-1),abs(err2_svd-err2_rand_5)./err2_svd,'g',...
         0:(n-1),abs(err2_svd-err2_rand_6)./err2_svd,'b',...
         0:(n-1),abs(err2_svd-err2_rand_7)./err2_svd,'b',...
         0:(n-1),abs(err2_svd-err2_rand_8)./err2_svd,'b',...
         0:(n-1),abs(err2_svd-err2_rand_9)./err2_svd,'r',...
         0:(n-1),abs(err2_svd-err2_rand_10)./err2_svd,'r',...
         0:(n-1),abs(err2_svd-err2_rand_11)./err2_svd,'r')
ylabel('e_k = ||A - A_k||')
xlabel('k')
title('Frobenius norm','FontWeight','normal')
ylim([1e-16 1e2])

% save(type)

end

function [] = DRIVER_time_plots(type,b,p,q)

n = 1000*[1 15];%1000*[2 3 4 5 6 8 10];
cpqr_time = zeros(1,length(n));
svd_time = zeros(1,length(n));
utv_time = zeros(1,length(n));
utv_ON_time = zeros(1,length(n));
utv_os_ON_time = zeros(1,length(n));
power_urv_time = zeros(1,length(n));


for i=1:length(n)

fprintf('Step %i/%i: n=%i \n',i,length(n),n(i))    

%%% Seed the random number generator, and set various parameters:
rng('default')
rng(0)

%%% Create a test matrix. 
switch type
    case 'fast'
        A = LOCAL_fast_decay(n(i),n(i),1e-5);
    case 'slow'
        A = LOCAL_slow_decay(n(i),n(i));
    case 's_curve'
        A = LOCAL_S_curve(n(i),n(i),round(0.5*n),1e-2);
    case 'hilb'
        A = hilb(n);
    case 'BIE'
        A = generate_BIE(n);
    case 'vander'
        A = vander(linspace(0,1,n));
    case 'gap'
        A= LOCAL_gap(m,n);
end

%%% Normalize the test matrix so it has spectral norm 1.
ss = svd(A);
A  = A/ss(1);
ss = ss/ss(1);

%%% Perform factorizations:
%t = cputime; 
%[U1,T1,V1] = qr(A,0); % built-in LAPACK
%cpqr_time(i) = cputime - t; 

t = cputime; 
%[U2,T2,V2] = randUTV_econ(A,b,0,q); % randomized UTV
utv_time(i) = cputime - t;
    
t = cputime;
%[U3,T3,V3] = randUTV_econ(A,b,0,q);
utv_ON_time(i) = cputime - t;

t = cputime;
%[U3,T3,V3] = randUTV_econ(A,b,p,q);
utv_os_ON_time(i) = cputime - t;

t = cputime;
[U3,T3,V3] = powerURV(A,q);
power_urv_time(i) = cputime - t;

%t = cputime;
%[U4,T4,V4] = svd(A,'econ'); % built-in LAPACK
%svd_time(i) = cputime - t;
    
end

power_urv_time
%%% Plot times
plot(n,utv_ON_time ./ n.^3,'b^-',...
        n,power_urv_time ./ n.^3,'r^-',...
        n,utv_os_ON_time ./ n.^3,'ro-.')
legend('randUTV ON=1,p=0',strcat('powerURV '),...
    strcat('randUTV ON=1,p=',num2str(p)))
ylabel('Time [s] / n^3')
xlabel('n')
axis('tight')
title(strcat('Cost for various factorizations (q=',num2str(q),' in UTV)'));

% save('fast')

end

function [U,T,V] = randUTV_econ(A,b,p,n_iter)

gpu = gpuDevice();

Ag = gpuArray(A);

%%% Extract matrix dimensions and calculate the number of steps required.
n     = size(Ag,2);
m     = size(Ag,1);
nstep = ceil(n/b);

%%% Initialize U and V and copy A onto T.
Ug = gpuArray(eye(m));
Vg = gpuArray(eye(n));
Tg = Ag;

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
   Alocg = Tg([J2,I3],[J2,J3]);
   if j == 1 || p == 0
       Yg    = Alocg'*gpuArray(randn(m-(j-1)*b,b+p));
   else
       Yg    = Alocg'*gpuArray(randn(m-(j-1)*b,b));
   end
   
   %%% Perform "power iteration" if requested.
   for i_iter = 1:n_iter
      
       if (p > 0 && i_iter == n_iter)
           if (j > 1)
               Yg = Yg - Ytmpg*Ytmpg'*Yg;
               [Yg,~] = qr(Yg,0);
               Yg = [Ytmpg Yg];
           else
               [Yg,~] = qr(Yg,0);
           end
       end
       
       Yg = Alocg'*(Alocg*Yg);
   end
   
   %%% If over-sampling is done, then reduce Y to b columns. 
   if p > 0
       [Utmpg,~,~] = svd(Yg,'econ');
       Yg = Utmpg(:,1:b);
       Ytmpg = Utmpg(:,(b+1):end);
   end

   
   %%% Perform b steps of Householder QR on the Y matrix,
   %%% and then apply the Householder reflections "from the right".
   [~,Vloc_Ug,Vloc_TUg] = LOCAL_dgeqrf_modified(Yg);
   Tg(:,[J2,J3]) = Tg(:,[J2,J3]) - (Tg(:,[J2,J3])*Vloc_Ug)*Vloc_TUg;
   Vg(:,[J2,J3]) = Vg(:,[J2,J3]) - (Vg(:,[J2,J3])*Vloc_Ug)*Vloc_TUg;
   %%% Next determine the rotations to be applied "from the left".
   [T22g,Uloc_Ug,Uloc_TUg] = LOCAL_dgeqrf_modified(Tg([J2,I3],J2g));
   Ug(:,[J2,I3])        = Ug(:,[J2,I3]) - (Ug(:,[J2,I3])*Uloc_Ug)*Uloc_TUg;
   Tg([J2,I3],J3)       = Tg([J2,I3],J3) - Uloc_TUg'*(Uloc_Ug'*Tg([J2,I3],J3));
   Tg(I3,J2)            = gpuArray(zeros(length(I3),length(J2)));
   %%% Perform an SVD of the diagonal block and update T, U, V, accordingly.
   [Usvdg,Dsvdg,Vsvdg] = svd(T22g);
   Tg(J2,J2)         = Dsvdg;
   Tg(J1,J2)         = Tg(J1,J2)*Vsvdg;
   Tg(J2,J3)         = Usvdg'*Tg(J2,J3);
   Vg(:,J2)          = Vg(:,J2)*Vsvdg;
   Ug(:,J2)          = Ug(:,J2)*Usvdg;
   
   % "kickstarting" next sample
   if p > 0
       % Ytmp = Vloc(:,b+1:end)'*Ytmp;
       Ytmpg = Ytmpg(b+1:end,:) - Vloc_TUg(:,b+1:end)'*(Vloc_Ug'*Ytmpg);
   end
   
end

%%% Process the last block.
%%% At this point, no "block pivoting" is required - we simply 
%%% diagonalize the block and update U and V accordingly.
J1               = 1:(b*(nstep-1));
J2               = (b*(nstep-1)+1):n;
I2               = (b*(nstep-1)+1):m;
[Usvdg,Dsvdg,Vsvdg] = svd(Tg(I2,J2));
Tg(I2,J2)         = Dsvdg;
Tg(J1,J2)         = Tg(J1,J2)*Vsvdg;
Ug( :,I2)         = Ug(:,I2)*Usvdg;
Vg( :,J2)         = Vg(:,J2)*Vsvdg;

T = gather(Tg);
U = gather(Ug);
V = gather(Vg);

end

function [U,R,V] = powerURV(A,q)

Ag = gpuArray(A);

n = size(Ag,1);

Gg = gpuArray(randn(n,n));

% do random sampling
%Y = A'*G;
for j=1:q
    Yg = Ag*Gg;
    [Gg,~] = qr(Yg);
    Yg = Ag'*Gg;
    [Gg,~] = qr(Yg);
end

% form U,R
Ag = Ag*Gg;
[Ug,Rg] = qr(Ag); 

clear Yg;

R = gather(Rg);
U = gather(Ug);
V = gather(Gg);

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

function [A] = LOCAL_fast_decay(m,n,beta)

%%% Determine the "inner dimension"
r = min(m,n);
jj = 0:(r-1);

%%% Form the actual matrix
[U,~] = qr(randn(m,r),0);
[V,~] = qr(randn(n,r),0);
ss    = beta.^(jj/(r-1));
A     = (U.*(ones(m,1)*ss))*V';

end

function A = LOCAL_slow_decay(m,n)

% Determine the "inner dimension"
r = min([m,n]);

[U,~] = qr(randn(m,r),0);
[V,~] = qr(randn(n,r),0);
ss    = 1./(1:r);
A     = (U.*(ones(m,1)*ss))*V';

end

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

end

function A = LOCAL_gap(m,n)

% Determine the "inner dimension"
%r = min([m,n,2000]);
r = min([m,n]);


ss = [logspace(-1.8,-2,r/2-5) logspace(-2,-10,10) logspace(-10,-11,r/2-5)];
[U,~] = qr(randn(m,r),0);
[V,~] = qr(randn(n,r),0);
A     = (U.*(ones(m,1)*ss))*V';

end



