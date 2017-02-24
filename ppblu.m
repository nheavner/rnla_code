function [L,U,P] = ppblu(A,b)
%Given an m x n matrix A, m >= n, and a block size b, 
% this function uses a blocked partial pivoting algorithm to compute a lower
%triangular matrix L, upper triangular matrix U and a row permutation matrix P,
% such that P*A = L*U

opts.LT = true; % For use later when performing triangular linsolves 

[m,n] = size(A);
i=1;

L = eye(m,n);
P = eye(m,m);

while b*i <= min(m,n)
    I1 = 1:b*(i-1);
    I2 = b*(i-1)+1:b*i; % These are the indices for the current block
    I3R = b*i+1:m; % The indices for the rows in the final block
    I3C = b*i+1:n; % The indices for the columns in the final block
    
    % perform pivoted Gaussian elimination
    [Lhat,U22,PRhat] = plu(A([I2 I3R],I2));
    L22 = Lhat(1:b,:);
    L32 = Lhat(b+1:end,:);
    
    % store multipliers in bottom part of A
    A([I2 I3R],I2) = A([I2 I3R],I2) - tril(A([I2 I3R],I2),-1) + tril(Lhat,-1);
    
    % permute remaining part of rows (the parts not in the I2 block)
    if i > 1
        A([I2 I3R],I1) = PRhat*A([I2 I3R],I1);
    end
    A([I2 I3R],I3C) = PRhat*A([I2 I3R],I3C);
    
    % multiple RHS solve to update pivot rows of A
    U23 = linsolve(L22,A(I2,I3C),opts);
    
    % update final block
    A(I3R,I3C) = A(I3R,I3C) - L32*U23;
    
    % update P, A, L
    placeholder = eye(m,m);
    placeholder([I2 I3R],[I2 I3R]) = PRhat;
    P = placeholder*P;
    
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

