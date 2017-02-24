function [] = nuclear_norm_test()

clear
clc

m = 2000;
n = 2000; % matrix size
b = 64; % block size
p = 10; % oversampling parameter
q = 2; % power iteration parameter
type = 'BIE'; % 'fast' for matrix with quickly decaying SVs
               % 'slow' for matrix with slowly decaying SVs
               % 's_curve' for matrix whose SVs decay quickly at first,
               %            then slowly at the end
               % 'hilbert' a hilbert matrix
               % 'BIE' is a PDE solution matrix
               % 'vander' is the Vandermonde matrix
               

%DRIVER_error_plots(type,m,n,b,p,q);
DRIVER_time_plots(type,b,p,q);



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
end

%%% Normalize the test matrix so it has spectral norm 1.
ss = svd(A);
A  = A/ss(1);
ss = ss/ss(1);

%%% Perform calculations:
tic;
[U,T,V] = randUTV_econ(A,b,p,q);        % randomized UTV
utv_time = toc;

tic;
[s] = nuc_norm_econ(A,b,p,q);         % optimized for nuclear norm
nn_time = toc;
fprintf('full UTV: %f sec \n',utv_time);
fprintf('nuclear norm: %f sec \n',nn_time);

fprintf('relative error for full UTV: %f \n',abs((sum(ss) - sum(diag(T)))/(sum(ss))));
fprintf('relative error for optimized algorithm: %f \n',abs((sum(ss) - s)/(sum(ss))));


end

function [] = DRIVER_time_plots(type,b,p,q)

n = 1000*[2 3 4 5 6 8 10];
cpqr_time = zeros(1,length(n));
svd_time = zeros(1,length(n));
noupdate_time = zeros(1,length(n));
update_time = zeros(1,length(n));
nuc_norm_time = zeros(1,length(n));


for i=1:length(n)

fprintf('Step %i/%i: n=%i \n',i,length(n),n(i))    

%%% Seed the random number generator, and set various parameters:
%rng('default')
%rng(0)

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
end

%%% Normalize the test matrix so it has spectral norm 1.
ss = svd(A);
A  = A/ss(1);
ss = ss/ss(1);

%%% Perform factorizations:
fprintf('Working on cpqr... \n')
t = cputime; 
[U1,T1,V1] = qr(A,0); % built-in LAPACK
cpqr_time(i) = cputime - t; 

fprintf('Working on randUTV... \n')
t = cputime; 
[U2,T2,V2] = randUTV_econ(A,b,p,q); % randomized UTV
noupdate_time(i) = cputime - t;
    
fprintf('Working on randUTVupdate... \n')
t = cputime; 
[U3,T3,V3] = randUTVupdate_econ(A,b,p); % randomized UTV with downdating
update_time(i) = cputime - t;

fprintf('Working on svd... \n')
t = cputime;
[U4,T4,V4] = svd(A,'econ'); % built-in LAPACK
svd_time(i) = cputime - t;

fprintf('Working on nuclear norm... \n \n')
t = cputime;
N = nuc_norm_econ(A,b,p,q);
nuc_norm_time(i) = cputime - t;
    
end

%%% Plot times
figure(1)
subplot(1,2,1)
hold off
plot(n,svd_time,'k^-',...
         n,cpqr_time,'rs-',...
         n,noupdate_time,'go-',...
         n,update_time,'go--',...
         n,nuc_norm_time,'bx-');
legend('svd','cpqr','randUTV','randUTVupdate','nuclearNorm')
ylabel('Time [ms]')
xlabel('n')
axis('tight')
title(strcat('Absolute times for various factorizations (q=',num2str(q),' in UTV)'));

subplot(1,2,2)
hold off
loglog(n,svd_time./(n.^3),'k^-',...
         n,cpqr_time./(n.^3),'rs-',...
         n,noupdate_time./(n.^3),'go-',...
         n,update_time./(n.^3),'go--',...
         n,nuc_norm_time./(n.^3),'bx-');
legend('svd','cpqr','randUTV','randUTVupdate','nuclearNorm')
ylabel('Time / n^3 [ms]')
xlabel('n')
axis('tight')
title(strcat('Times scaled by n^3 for various factorizations (q=',num2str(q),' in UTV)'));

% save('fast')

end

function [s] = nuc_norm(A,b,p,q)
% given an m x n matrix A, computes an approximation of the nuclear norm of
% A, the sum of the singular values

% inputs:
% A: input matrix of dimensions m x n, with m >= n
% b: block size b; typically b = 16,32,64
% q: parameter for power iteration; typically q = 0,1,2

s = 0;
T = A;

for i=1:ceil(size(A,2)/b)
    
    I2 = (b*(i-1)+1):(b*i);
    I3 = (b*i+1):size(A,1);
    J2 = (b*(i-1)+1):(b*i);
    J3 = (b*i+1):size(A,2);
    
    if ~isempty(J3)
        [TT,s_part] = step_nuc_norm(T([I2 I3],[J2 J3]),b,p,q);
        s = s + s_part;
        T(I3,J3) = TT;
    else
        [~,D,~] = svd(T(I2,J2));
        s = s + sum(diag(D));
    end
    
end



end

function [T,s_part] = step_nuc_norm(A,b,p,q)
% given an m x n matrix A, and parameters b, q, performs the bulk of the
% work of a single step of the UTV factorization

G = randn(size(A,1),b+p);

Y = A'*G;

for i=1:q
    Y = A'*(A*Y);
end

[V,~] = qr(Y);

[U,D,~] = svd(A*V(:,1:b));

s_part = sum(diag(D));

T = U(:,b+1:end)'*A*V(:,(b+1):end);

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
J1               = 1:(n-b);
J2               = (b*(nstep-1)+1):n;
I2               = (b*(nstep-1)+1):m;
[Usvd,Dsvd,Vsvd] = svd(T(I2,J2));
T(I2,J2)         = Dsvd;
T(J1,J2)         = T(J1,J2)*Vsvd;
U( :,I2)         = U(:,I2)*Usvd;
V( :,J2)         = V(:,J2)*Vsvd;

end

function [U,T,V] = randUTVupdate_econ(A,b,p)

%%% Extract matrix dimensions and calculate the number of steps required.
n     = size(A,2);
m     = size(A,1);
nstep = ceil(n/b);

%%% Initialize U, V, G, Y and copy A onto T;
U = eye(m);
V = eye(n);
T = A;

G = randn(m,b+p);
Y = A'*G;

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
   %%% downdate the sampling matrix Y and the random matrix G
   if j < (nstep-1)
       UG = G - Uloc_TU'*(Uloc_U'*G); % U'*G
       UG(1:b,:) = Usvd'*UG(1:b,:);
       Y = -T(J2,J3)'*UG(1:b,:);
       G =  UG((b+1):end,:);
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

function [s] = nuc_norm_econ(A,b,p,n_iter)

%%% Extract matrix dimensions and calculate the number of steps required.
n     = size(A,2);
m     = size(A,1);
nstep = ceil(n/b);

%%% copy A onto T; initialize nuclear norm estimate
T = A;
s = 0;

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
   T([J2,I3],[J2,J3]) = T([J2,I3],[J2,J3]) - (T([J2,I3],[J2,J3])*Vloc_U)*Vloc_TU;
   %%% Next determine the rotations to be applied "from the left".
   [T22,Uloc_U,Uloc_TU] = LOCAL_dgeqrf_modified(T([J2,I3],J2));
   T([J2,I3],J3)       = T([J2,I3],J3) - Uloc_TU'*(Uloc_U'*T([J2,I3],J3));
   %%% Perform an SVD of the diagonal block and update T, U, V, accordingly.
   Dsvd = svd(T22);
   s = s + sum(Dsvd);
   
end

%%% Process the last block.
%%% At this point, no "block pivoting" is required - we simply 
%%% diagonalize the block and update U and V accordingly.
J1               = 1:(n-b);
J2               = (b*(nstep-1)+1):n;
I2               = (b*(nstep-1)+1):m;
Dsvd = svd(T(I2,J2)); % could replace this with a version that
                            % doesn't compute U,V

s = s + sum(Dsvd);

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



