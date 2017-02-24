function [L,U] = blu(A,r)
% given an n x n matrix A and a block size r, this matrix returns a lower triangular matrix L
% and upper triangular matrix U such that A = L*U; the algorithm is the
% nonrecursive block LU factorization in Golub and van Loan's "Matrix
% Computations"

% NOTE: matrix A must be square for this algorithm!

% NOTE 2: This algorithm does not utilize pivoting and is therefore
% unstable and may even fail outright!!

opts.LT = true; % For use later when performing triangular linsolves 

[n,~] = size(A);

L = zeros(n,n);
U = zeros(n,n);

N = floor(n/r);

for k=1:N
    kblockind = (k-1)*r+1:k*r;     % Contains the indices that correspond to the current block (note that 
    kblockend = (k-1)*r+1:n;       % each block is square); kblockind starts and ends at current block;
                                   % kblockend starts at current block and
                                   % goes to the end of the matrix
                                   
    % find the LU factorization down all the rows and one block's width of columns                               
    if k < N
        [L(kblockend,kblockind),U(kblockind,kblockind)] = mylu(A(kblockend,kblockind));
    else
        [L(kblockend,kblockend),U(kblockend,kblockend)] = mylu(A(kblockend,kblockend));
    end
    
    % for the rows that were just zeroed out, determine the rest of the row
    % all the way across so we can update the other rows with the
    % multipliers
    U(kblockind,kblockend) = linsolve(L(kblockind,kblockind),A(kblockind,kblockend),opts);
   
    % level three updates to A
    for i=k+1:N % When k == N, MATLAB skips these loops, which is what we want
        iblockind = (i-1)*r+1:i*r;
        iblockend = (i-1)*r+1:n;
        for j=k+1:N
            jblockind = (j-1)*r+1:j*r;
            jblockend = (j-1)*r+1:n;
            if (i < N) && (j < N)
                A(iblockind,jblockind) = A(iblockind,jblockind) - L(iblockind,kblockind)*U(kblockind,jblockind);
            elseif (i < N) && (j == N)
                A(iblockind,jblockend) = A(iblockind,jblockend) - L(iblockind,kblockind)*U(kblockind,jblockend);
            elseif (i == N) && (j < N)
                A(iblockend,jblockind) = A(iblockend,jblockind) - L(iblockend,kblockind)*U(kblockind,jblockind);
            else    % i == N and j == N
                A(iblockend,jblockend) = A(iblockend,jblockend) - L(iblockend,kblockind)*U(kblockind,jblockend);
            end
        end
    end
end

end

