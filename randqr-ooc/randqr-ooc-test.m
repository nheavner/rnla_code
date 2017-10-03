function [] = randqr_ooc_test()

DRIVER_fast

end

function DRIVER_fast

%%% Seed the random number generator, and set various parameters:
rng('default')
rng(0)
n = 100;  % Matrix size.
b =  8;   % Block size.
p =  1;   % Over-sampling parameter.

%%% Create a test matrix. 
fname = LOCAL_fast_decay_ooc(n,n,1e-5);

% the built-in factorizations need A to be stored in RAM
fid = fopen(fname,'r');
A = fread(fid,'double');
A = vec2mat(A,n);
A = A';

%%% Normalize the test matrix so it has spectral norm 1.
ss = svd(A);
A  = A/ss(1);
ss = ss/ss(1);

%%% Perform three versions of QR factorization:
[Q1,R1,J1] = qr(A,'vector');         % Built-in LAPACK.
[Q2,R2,J2] = qr_right(A,b,p);% randqr_ooc();           % randqr out of core 

% done with the out of core file
fclose(fid);

%%% Perform the most basic error checks:
fprintf(1,'                        LAPACK          randQR ooc \n')
fprintf(1,'||A*P - Q*R|| (fro) = %12.5e    %12.5e \n',...
        norm(A(:,J1) - Q1*R1,'fro'),...
        norm(A(:,J2) - Q2*R2,'fro'))
fprintf(1,'||A*P - Q*R|| (op)  = %12.5e    %12.5e \n',...
        norm(A(:,J1) - Q1*R1),...
        norm(A(:,J2) - Q2*R2))
fprintf(1,'max|trans(Q)*Q - I| = %12.5e    %12.5e \n',...
        max(max(abs(Q1'*Q1 - eye(n)))),...
        max(max(abs(Q2'*Q2 - eye(n)))))

fprintf('Computing spectral norm errors')
%%% Compute spectral norm errors
err1_classic = [ss(1),zeros(1,n-1)];
err1_rand    = [ss(1),zeros(1,n-1)];
for i = 1:(n-1)
  if mod(i,100) == 0
      i
  end
  err1_classic(i+1) = norm(R1((i+1):n,(i+1):n));
  err1_rand(i+1)    = norm( R2((i+1):n,(i+1):n));
end

fprintf('Computing Frobenius norm errors')
%%% Compute Frobenius norm errors
err2_svd     = [norm(A,'fro'),zeros(1,n-1)];
err2_classic = [norm(A,'fro'),zeros(1,n-1)];
err2_rand    = [norm(A,'fro'),zeros(1,n-1)];
for i = 1:(n-1)
  if mod(i,100) == 0
      i
  end
  err2_svd(i+1)     = norm(ss((i+1):n));
  err2_classic(i+1) = norm(R1((i+1):n,(i+1):n),'fro');
  err2_rand(i+1)    = norm( R2((i+1):n,(i+1):n),'fro');
end

%%% Plot errors
figure(1)
subplot(1,2,1)
semilogy(0:(n-1),ss,'k',...
         0:(n-1),err1_classic,'r',...
         0:(n-1),err1_rand,'g')
legend('optimal','HQRP','HQRRP ooc')
%legend('svd','cpqr','randQR','randQRupdate')
ylabel('e_k = ||A - A_k||')
xlabel('k')
title('Operator norm','FontWeight','normal');

subplot(1,2,2)
semilogy(0:(n-1),err2_svd,'k',...
         0:(n-1),err2_classic,'r',...
         0:(n-1),err2_rand,'g')
legend('optimal','HQRP','HQRRP ooc')
%legend('svd','cpqr','randQR','randQRupdate')
ylabel('e_k = ||A - A_k||')
xlabel('k')
title('Frobenius norm','FontWeight','normal');
%export_fig('fig_errors_fast.pdf','-pdf','-trans')

%%% Plot diagonal entries.
figure(2)
subplot(1,1,1)
semilogy(1:n,ss,'k',...
         1:n,abs(diag(R1)),'r',...
         1:n,abs(diag(R2)),'g')
legend('svd','HQRP','HQRRP ooc')
% legend('svd','cpqr','randQR right','randQR left')
xlabel('k')
ylabel('|R(k,k)|')
%title('Magnitude of diagonal entries','FontWeight','normal')
title('Matrix 1 ("fast")','FontWeight','normal')
% %export_fig('fig_diag_fast.pdf','-pdf','-trans')
% 
keyboard

end

function [Q,R,J] = qr_left(fname,A,b,p)

n     = size(A,2);
nstep = ceil(n/b);
Q     = eye(n);
J     = 1:n;

%%% We will create an upper triangular matrix "R" by applying a sequence
%%% of ON maps to the matrix "A". For now, we simply copy A to R, and we 
%%% will then perform the various transforms (overwriting R at each step).
R = A;

%%% Draw a Gaussian matrix, and build the associated sampling matrix. Note
%%% that this is the only time we generate a random matrix, and the only
%%% time we multiple the random matrix by a "large" matrix.
G = randn(b+p,n);
Y = G*A;

%%% Process all blocks, except the last.
%%% At each step, we partition the index vector as follows:
%%%    (1:n) = [I1, I2, I3]
%%% where "I2" is the block currently being processed.
for j  = 1:nstep

  %%% Construct the index vectors that partition the matrix.  
  I1 = 1:((j-1)*b);
  I2 = (j-1)*b + (1:min(b,n-b*(j-1)));
  I3 = (j*b+1):n;

  %%% Find b good pivot columns from [I2 I3] using the randomized sampling
  %%% procedure and move them to the I2 column.
  %%% (We don't do this at the last step, when I3 is empty.)
  if (j < nstep)
    %%% Determine pivots using the randomized strategy.
    [~,~,Jloc]   = pivoted_QR(Y(:,[I2,I3]),b);
    %%% Permute the columns in the [I2,I3] block as dictated by Jloc:
    I23          = [I2 I3];
    R(I1,[I2,I3]) = R(I1,I23(Jloc));
    Y(:,[I2,I3]) = Y(:,I23(Jloc));
    J([I2,I3])   = J(I23(Jloc));
  end

  %%% update lower middle column with Q, P
  R([I2 I3],I2) = Q(:,[I2 I3])'*A(:,J(I2));
  
  %%% Perform QR on the middle column, and update accordingly
  [Qloc,Rloc,Jloc] = qr(R([I2 I3],I2),'vector');
  R(I1,I2)         = R(I1,I2(Jloc));
  R([I2,I3],I2)    = Rloc;
  Q(:,[I2,I3])     = Q(:,[I2,I3])*Qloc;
  
  R(I2,I3)         = Q(:,I2)'*A(:,J(I3));
  Y(:,I2)          = Y(:,I2(Jloc));
  Y                = [zeros(b+p,length(I1)),Y(:,I23) - (G*Q(:,I2))*R(I2,I23)];
  J(I2)            = J(I2(Jloc));
%  Rtmp             = [zeros(n-length(I3),n);zeros(length(I3),n-length(I3)),R(I3,I3)];
%  fprintf(1,'||A*P - Q*R|| = %12.5e    ||Y - G*Q*[0,0;0,R33]|| = %12.5e\n',...
%          norm(A(:,J) - Q*R),norm(G*Q*Rtmp - Y))

end

end

function [Q,R,J] = qr_right(A,b,p)

n     = size(A,2);
nstep = ceil(n/b);
Q     = eye(n);
J     = 1:n;

%%% We will create an upper triangular matrix "R" by applying a sequence
%%% of ON maps to the matrix "A". For now, we simply copy A to R, and we 
%%% will then perform the various transforms (overwriting R at each step).
R = A;

%%% Draw a Gaussian matrix, and build the associated sampling matrix. Note
%%% that this is the only time we generate a random matrix, and the only
%%% time we multiple the random matrix by a "large" matrix.
G = randn(b+p,n);
Y = G*A;

%%% Process all blocks, except the last.
%%% At each step, we partition the index vector as follows:
%%%    (1:n) = [I1, I2, I3]
%%% where "I2" is the block currently being processed.
for j  = 1:nstep

  %%% Construct the index vectors that partition the matrix.  
  I1 = 1:((j-1)*b);
  I2 = (j-1)*b + (1:min(b,n-b*(j-1)));
  I3 = (j*b+1):n;

  %%% Find b good pivot columns from [I2 I3] using the randomized sampling
  %%% procedure and move them to the I2 column.
  %%% (We don't do this at the last step, when I3 is empty.)
  if (j < nstep)
    %%% Determine pivots using the randomized strategy.
    [~,~,Jloc]   = pivoted_QR(Y(:,[I2,I3]),b);
    %%% Permute the columns in the [I2,I3] block as dictated by Jloc:
    I23          = [I2 I3];
    R(:,[I2,I3]) = R(:,I23(Jloc));
    Y(:,[I2,I3]) = Y(:,I23(Jloc));
    J([I2,I3])   = J(I23(Jloc));  
  end

  %%% Perform QR on the "middle" column, and update accordingly.
  [Qloc,Rloc,Jloc] = qr(R([I2,I3],I2),'vector');
  R(I1,I2)         = R(I1,I2(Jloc));
  R([I2,I3],I2)    = Rloc;
  R([I2,I3],I3)    = Qloc'*R([I2,I3],I3);
  Q(:,[I2,I3])     = Q(:,[I2,I3])*Qloc;
  Y(:,I2)          = Y(:,I2(Jloc));
  Y                = [zeros(b+p,length(I1)),Y(:,I23) - (G*Q(:,I2))*R(I2,I23)];
  J(I2)            = J(I2(Jloc));
%  Rtmp             = [zeros(n-length(I3),n);zeros(length(I3),n-length(I3)),R(I3,I3)];
%  fprintf(1,'||A*P - Q*R|| = %12.5e    ||Y - G*Q*[0,0;0,R33]|| = %12.5e\n',...
%          norm(A(:,J) - Q*R),norm(G*Q*Rtmp - Y))

end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function performs classical column pivoted QR.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Q,R,ind] = pivoted_QR(A,k)

% This function orthogonalizes the COLUMNS of A
% It uses a modified Gram-Schmidt with column pivoting

m = size(A,1);
n = size(A,2);

R = zeros(min(m,n),n);
Q = A;
ind = 1:n;

for j = 1:k
    [~, j_max] = max(sum(Q(:,j:n).*Q(:,j:n)));
    j_max          = j_max + j - 1;
    Q(:,[j,j_max]) = Q(:,[j_max,j]);
    R(:,[j,j_max]) = R(:,[j_max,j]);
    ind([j,j_max]) = ind([j_max,j]);
    r_jj   = norm(Q(:,j));
    Q(:,j) = Q(:,j)/r_jj;
    Q(:,j) = Q(:,j) - Q(:,1:(j-1))*(Q(:,1:(j-1))'*Q(:,j));
    Q(:,j) = Q(:,j)/norm(Q(:,j));
    R(j,j) = r_jj;
    rr     = Q(:,j)'*Q(:,(j+1):n);
    R(j,(j+1):n) = rr;
    Q(:,(j+1):n) = Q(:,(j+1):n) - Q(:,j)*rr;
end

Q = Q(:, 1:min(m,n));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = LOCAL_fast_decay(m,n,beta)

%%% Determine the "inner dimension"
r = min(m,n);
jj = 0:(r-1);

%%% Form the actual matrix
[U,~] = qr(randn(m,r),0);
[V,~] = qr(randn(n,r),0);
ss    = beta.^(jj/(r-1));
A     = (U.*(ones(m,1)*ss))*V';

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Q,R,J] = qr_ooc(fp,A,b,p)
% fp is the file pointer to the matrix A stored out of core


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fname = LOCAL_fast_decay_ooc(m,n,beta)

%%% Determine the "inner dimension"
r = min(m,n);
jj = 0:(r-1);

%%% Form the actual matrix
[U,~] = qr(randn(m,r),0);
[V,~] = qr(randn(n,r),0);
ss    = beta.^(jj/(r-1));
A     = (U.*(ones(m,1)*ss))*V'

fname = 'mat_ooc';
fid = fopen(fname,'w');

% write to binary file in column major order
fwrite(fid,A,'double');

fclose(fid);

end