function H = logSoftmaxHessianMC(x, n)
    H = zeros(length(x), length(x), length(x));
    % For the first n entries
    H(1:n, 1:n, 1:n) = logSoftmaxHessian(x(1:n));
    % For the next n^2 entries
    for i = 1:n
        H(n + (i-1)*n + 1 : n + i*n, n + (i-1)*n + 1 : n + i*n, n + (i-1)*n + 1 : n + i*n) = logSoftmaxHessian(x(n + (i-1)*n + 1 : n + i*n));
    end
end

function H = logSoftmaxHessian(z)
    s = softmax(z);
    K = length(z);
    H = zeros(K, K, K);
    for i = 1:K
        for j = 1:K
            for k = 1:K
                H(i, j, k) = -s(j) * (kroneckerDelta(j, k) - s(k));
            end
        end
    end
end

function d = kroneckerDelta(i, j)
    if i == j
        d = 1;
    else
        d = 0;
    end
end
