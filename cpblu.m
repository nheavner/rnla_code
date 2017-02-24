function [L,U,P,Q] = cpblu(A,b,k)
%Given an m x n matrix A a block size b, and a sampling parameter k, 
% this function uses Gaussian elimination with complete pivoting to compute a lower
%triangular matrix L, upper triangular matrix U, row permutation matrix P,
%and column permutation matrix Q such that P*A*Q = L*U

opts.LT = true; % For use later when performing triangular linsolves 

[m,n] = size(A);
i=1;

L = eye(m,n);
P = eye(m,m);
Q = eye(n,n);

while b*i <= min(m,n)
    I1 = 1:b*(i-1);
    I2 = b*(i-1)+1:b*i; % These are the indices for the current block
    I3R = b*i+1:m;
    I3C = b*i+1:n;
    
    % compute sampling matrix and determine pivot columns
    Omega = normrnd(0,1,k,m-b*(i-1));
    Y = Omega*A([I2 I3R],[I2 I3C]); % Omega*A(b*(i-1)+1:m,b*(i-1)+1:n);
    [~,~,PCtil] = gsqr(Y);
    
    % update cols of A
    A([I1 I2 I3R],[I2 I3C]) = A([I1 I2 I3R],[I2 I3C])*PCtil;
    
    % perform pivoted Gaussian elimination
    [Lhat,U22,PRhat] = plu(A([I2 I3R],I2));
    L22 = Lhat(1:b,:);
    L32 = Lhat(b+1:end,:);
    
    % store multipliers in bottom part of A
    A([I2 I3R],I2) = A([I2 I3R],I2) - tril(A([I2 I3R],I2),-1) + tril(Lhat,-1);
    
    % permute remaining part of rows
    if i > 1
        A([I2 I3R],I1) = PRhat*A([I2 I3R],I1);
    end
    
    A([I2 I3R],I3C) = PRhat*A([I2 I3R],I3C);
    
    % multiple RHS solve to update pivot rows of A
    U23 = linsolve(L22,A(I2,I3C),opts);
    
    % update final block
    A(I3R,I3C) = A(I3R,I3C) - L32*U23;
    
    % update PR, PC, A, L
    placeholder = eye(m,m);
    placeholder([I2 I3R],[I2 I3R]) = PRhat;
    P = placeholder*P;
    
    placeholder = eye(n,n);
    placeholder([I2 I3C],[I2 I3C]) = PCtil;
    Q = Q*placeholder;
    
    A(I2,[I2 I3C]) = A(I2,[I2 I3C]) - triu(A(I2,[I2 I3C])) + triu([U22 U23]);
    L([I2 I3R],I2) = [L22; L32];
    
    
    i = i + 1;
end

% process final lower-right block of A
if b*(i-1)+1 <= n
    I1 = 1:b*(i-1);
    I2R = b*(i-1)+1:m; % These are the indices for the current block rows
    I2C = b*(i-1)+1:n; % Indices for current block columns
    
    A22 = A(I2R,I2C);
    [L22,U22,PRhat] = plu(A22);

    % permute rest of row of A
    A(I2R,I1) = PRhat*A(I2R,I1);
    
    A(I2C,I2C) = triu(U22); % Note the shorter dimensions on the
                            % row index of A because the LU fact is
                            % possibly rectangular
    A(I2R,I2C) = A(I2R,I2C) - tril(A(I2R,I2C),-1) + tril(L22,-1);
    placeholder = eye(m,m);
    placeholder(I2R,I2R) = PRhat;
    P = placeholder*P;
end
U = triu(A);
U = U(1:n,:);
L = eye(m,n) + tril(A,-1);
end

