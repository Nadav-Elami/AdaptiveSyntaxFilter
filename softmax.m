function s = softmax(z)
    exp_z = exp(z - max(z)); % Subtracting max(z) for numerical stability
    s = exp_z / sum(exp_z);
end
