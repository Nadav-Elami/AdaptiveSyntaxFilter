function J = logSoftmaxJacobianMC(x, n)
    J = zeros(length(x), length(x));
    % For the first n entries
    J(1:n, 1:n) = logSoftmaxJacobian(x(1:n));
    % For the next n^2 entries
    for i = 1:n
        J(n + (i-1)*n + 1 : n + i*n, n + (i-1)*n + 1 : n + i*n) = logSoftmaxJacobian(x(n + (i-1)*n + 1 : n + i*n));
    end
end

function J = logSoftmaxJacobian(z)
    s = softmax(z);
    K = length(z);
    J = zeros(K, K);
    for i = 1:K
        for j = 1:K
            if i == j
                J(i, j) = 1 - s(j);
            else
                J(i, j) = -s(j);
            end
        end
    end
end