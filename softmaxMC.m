function s = softmaxMC(x, n)
    s = zeros(size(x));
    % For the first n entries
    s(1:n) = softmax(x(1:n));
    % For the next n^2 entries
    for i = 1:n
        s(n + (i-1)*n + 1 : n + i*n) = softmax(x(n + (i-1)*n + 1 : n + i*n));
    end
end