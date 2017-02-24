function [A] = permrow(A,P)
% given a matrix A and permutation vector P (P(i) = j means: "move the j-th
% row of A to the i-th row"), computes the permuted matrix "P*A"

A = A(P,:);

end

