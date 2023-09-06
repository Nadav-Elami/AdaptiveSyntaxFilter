function ls = logSoftmaxMC(x, n)
    ls = zeros(size(x));
    % For the first n entries
    ls(1:n) = logSoftmax(x(1:n));
    % For the next n^2 entries
    for i = 1:n
        ls(n + (i-1)*n + 1 : n + i*n) = logSoftmax(x(n + (i-1)*n + 1 : n + i*n));
    end
end

function ls = logSoftmax(z)
    ls = z - log(sum(exp(z)));
end
