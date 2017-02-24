function [A] = permcol(A,P)

% given a matrix A and permutation vector P (P(i) = j means: "move the j-th
% column of A to the i-th position"), computes the permuted matrix "A*P"

A = A(:,P);

end

