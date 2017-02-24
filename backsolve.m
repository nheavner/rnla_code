function [x] = backsolve(A,b)
%Given an n x n matrix lower or upper triangular matrix A and an n x 1
%vector b, outputs an n x 1 vector x s.t. Ax = b by performing back or
%forward substitution

x = b;
[n,~] = size(A); % A must be square!

if norm(A(1,2:end)) > 0
    for i=n:-1:1    % back-substitute
        for j=n:-1:i
            if j > i
                x(i) = x(i) - A(i,j)*x(j);
            else
                x(i) = x(i) / A(i,j);
            end
        end
    end
else
    for i=1:n % forward-substitute
        for j=1:i
            if j < i
                x(i) = x(i) - A(i,j)*x(j);
            else
                x(i) = x(i) / A(i,j);
            end
        end
    end
end

end

