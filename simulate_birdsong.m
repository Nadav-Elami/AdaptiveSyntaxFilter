function [sequences, x_values] = simulate_birdsong(num_sequences, seq_range, x_init, Sigma_init, p) % take p off

% Define the alphabet
alphabet = {'a', 'b', 'c'};
num_phrases = length(alphabet);

% If x_init and Sigma_init are not provided, initialize them randomly
if nargin < 3
    x = randn(num_phrases + num_phrases^2, 1);
    Sigma = diag(rand(num_phrases + num_phrases^2, 1));
else
    x = x_init;
    Sigma = Sigma_init;
end

sequences = cell(num_sequences, 1);
x_values = zeros(length(x), num_sequences);

for seq_idx = 1:num_sequences
    sequence = '';
    
    % If seq_length is not provided, choose randomly between 3 and 8
    if nargin < 2
        seq_length = randi([3, 8]);
    else
        seq_length = randi(seq_range);
    end

    % Start with the first phrase
    probs = softmax(x(1:num_phrases));
    sequence = [sequence, alphabet{find(rand < cumsum(probs), 1)}];
    
    % Generate the rest of the sequence based on the Markov chain
    for i = 2:seq_length
        prev_phrase_idx = find(strcmp(sequence(end), alphabet));
        start_idx = num_phrases + (prev_phrase_idx - 1) * num_phrases + 1;
        end_idx = start_idx + num_phrases - 1;
        probs = softmax(x(start_idx:end_idx));
        sequence = [sequence, alphabet{find(rand < cumsum(probs), 1)}];
    end
    
    sequences{seq_idx} = sequence;
    x_values(:, seq_idx) = x;
    
    % Add Gaussian noise to x for the next sequence
    if p == 1
        x = x + 0*mvnrnd(zeros(size(x)), Sigma)';
    elseif p == 4
        SigmaNew = (1 - seq_idx/num_sequences) * Sigma;
        x = x + mvnrnd(zeros(size(x)), SigmaNew)';
    else
        x = x + mvnrnd(zeros(size(x)), Sigma)'; % this is the only line that should be left
    end
end

end