function [L,U,P,Q] = cplu(A)
%Given an n x n invertible matrix A, this function uses Gaussian
%elimination with complete pivoting to compute n x n matrices L and U and
%n x n permutation matrices P and Q s.t. P*A*Q = L*U

% initialize matrices
[n,~] = size(A);
P = zeros(n,1);
for i=1:n
    P(i) = i;
end

Q = zeros(n,1);
for i=1:n
    Q(i) = i;
end

for i=1:n-1

    mind = zeros(2,1);
    Asmall = A(i:n,i:n);
    [~,absind] = max(abs(Asmall(:)));
    if mod(absind,n-i+1) ~= 0
        mind(1) = mod(absind,n-i+1);
    else
        mind(1) = n-i+1;
    end
    mind(2) = floor((absind-1)/(n-i+1))+1;
    
    mind = mind + (i-1);    % need to add on (i-1) to refer to the correct absolute row and column of A
    
    % Record which rows are swapped
    temp = P(mind(1));
    P(mind(1)) = P(i);
    P(i) = temp;
    
    % Record which cols are swapped
    temp = Q(mind(2));
    Q(mind(2)) = Q(i);
    Q(i) = temp;
    
    % Swap the rows of A
    temp = A(mind(1),:);
    A(mind(1),:) = A(i,:);
    A(i,:) = temp;
    
    % Swap the cols of A
    temp = A(:,mind(2));
    A(:,mind(2)) = A(:,i);
    A(:,i) = temp;
    
    if A(i,i) ~= 0
        if i < n
            rho = i+1:n;               
            % perform the pivot
            A(rho,i) = A(rho,i)/A(i,i); % determines multipliers and stores them in lower part of A
            A(rho,i+1:end) = A(rho,i+1:end) - A(rho,i)*A(i,i+1:end); % updates rest of rows
        else
            % in this case, you are in the bottom right corner, so do
            % nothing
        end
        
    end  

end

L = eye(n,n) + tril(A,-1);
U = triu(A);
U = U(1:n,:);
P = permrow(eye(n,n),P);
Q = permcol(eye(n,n),Q);

end

