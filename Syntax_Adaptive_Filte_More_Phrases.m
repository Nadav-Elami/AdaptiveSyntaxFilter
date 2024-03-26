%% === Generate Data for the Simulation === %%

rng(0) % Fixed seed for reproducibility

% Define the alphabet
alphabet = {'a', 'b', 'c', 'd', 'e', 'f'};
num_phrases = length(alphabet);

% Define Simulation Parameters
seq_range = [2, 12];                                     % Desired sequence length
n_sequences = 4*randi([59 608]);                         % Desired num sequences
x_init = randi(15,num_phrases + num_phrases^2,1);        % Initial x values
SF = 0.5*10^2;                                           % Scaling Factor for Sigma_init
Sigma_init = diag(rand(num_phrases + num_phrases^2, 1));

[sequences, x_values, A] = simulate_birdsong(n_sequences, seq_range, x_init, Sigma_init, alphabet);

%% === Initialization of EM Algorithm === %%

% Number of sequences
K = length(sequences);

% Maximum sequence length
max_seq_length = max(cellfun(@length, sequences));

% Number of phrases in the alphabet
num_phrases = length(alphabet); % {'a', 'b', 'c'}

% Total size of the parameter vector x
R = num_phrases + num_phrases^2;

% Initial parameter vector
x_0 = ones(R,1)/num_phrases; %randn(R, 1);

% Random noise matrix
Sigma = diag(ones(R,1))/SF; %diag(rand(R, 1))/SF;

% Initialize other matrices and vectors
x_k_k_1     = zeros(R, K);      % State estimate at time k given observations up to k
x_k_k       = zeros(R, K);      % Updated state estimate at time k given observations up to k
x_k_K       = zeros(R, K);      % State estimate at time k given all observations
W_k_k_1     = zeros(R, R, K);   % State covariance at time k given observations up to k
W_k_k       = zeros(R, R, K);   % Updated state covariance at time k given observations up to k
W_k_K       = zeros(R, R, K);   % State covariance at time k given all observations
M_k         = zeros(R, R, K);   % Smoothing gain

% Additional matrices for the smoothing step
x_k_K_2     = zeros(R, R, K);   % Second moment of the state estimate at time k given all observations
W_k_u_K     = zeros(R, R, K, K);% Cross-covariance between states at different times given all observations

% Initial values
x_k_k_1(:, 1)       = x_0;
W_k_k_1(:, :, 1)    = Sigma;

% Maximum number of iterations for the EM algorithm
MAX_ITERATIONS = 100;

% Convergence limit for the EM algorithm
convergence_measure = zeros(1,MAX_ITERATIONS);
EM_CONV_LIMIT = 1e-3;

% Current iteration of the EM algorithm
Current_EM_iter = 1;

% Stop criterion for the EM algorithm (0 means continue, 1 means stop)
EM_stop_criterion = 0;

%% === EM Simulation === %%

% Create a figure outside the loop
figure("Position",[21.67,42,727.33,605.33]);


% % Define video object
% % Generate the video filename
% videoFile = sprintf('Video_Name.avi');
% videoObj = VideoWriter(videoFile);
% open(videoObj);


while (Current_EM_iter <= MAX_ITERATIONS) && (EM_stop_criterion == 0)
    %% === E Step - Forward Filter === %%

    % The E step involves estimating the expected value of the log-likelihood function
    % given the observed data and the current estimate of the parameters.
    % The forward filter is used to compute the expected sufficient statistics
    % of the hidden states given the observed data and current parameter estimates.
    %
    % Procedure:
    % 1. Initialize with random x_(0|0) and set W_(0|0) to zero.
    % 2. Update x_(k|k-1) and W_(k|k-1) using previous values and Sigma.
    % 3. For each sequence, compute:
    %    (*) Update x_(k|k) using logSoftmaxJacobian.
    %    (**) Update W_(k|k) using logSoftmaxHessian.

    % 1. Initialization
    x_k_k       = zeros(R, K);
    x_k_k_1     = zeros(R, K);
    W_k_k       = zeros(R, R, K);
    W_k_k_1     = zeros(R, R, K);

    % 2. Forward filter
    for k = 1:K
        if k == 1
            % for k == 1
            x_k_k_1(:, 1) = x_0;
            W_k_k_1(:, :, 1) = Sigma;
        else
            % Update x_k_k_1 and W_k_k_1 for the current k
            x_k_k_1(:, k) = x_k_k(:, k-1);
            W_k_k_1(:, :, k) = W_k_k(:, :, k-1) + Sigma;
        end

        % Initialize sum terms for (*) and (**)
        sum_term1 = zeros(R, 1);
        sum_term2 = zeros(R, R);

        % Compute Jacobian and Hessian outside the loop for efficiency
        Jacobian = logSoftmaxJacobianMC(x_k_k_1(:, k), num_phrases);
        Hessian  = logSoftmaxHessianMC(x_k_k_1(:, k), num_phrases);

        % For each phrase in the sequence
        for m = 1:length(sequences{k})
            y_m = sequences{k}(m);
            if m == 1
                y_m_prev = ''; % No previous phrase for the first phrase
            else
                y_m_prev = sequences{k}(m-1);
            end

            % Compute indices for x based on y_m and y_m_prev
            if isempty(y_m_prev)
                idx = find(strcmp(alphabet, y_m));
            else
                idx_prev = find(strcmp(alphabet, y_m_prev));
                idx = num_phrases + (idx_prev - 1) * num_phrases + find(strcmp(alphabet, y_m));
            end

            % Update sum terms using Jacobian and Hessian
            sum_term1 = sum_term1 + Jacobian(idx, :).';
            sum_term2 = sum_term2 + Hessian(:, :, idx);
        end

        % (**) Compute W_k_k using the updated sum terms
        epsilon = 1e-5; % Small regularization term
        W_k_k(:, :, k) = inv(inv(W_k_k_1(:, :, k) + epsilon * eye(size(W_k_k_1, 1))) + sum_term2);

        % (*) Compute x_k_k using the updated sum terms
        x_k_k(:, k) = x_k_k_1(:, k) + W_k_k(:, :, k) * sum_term1;
    end

    %% === E Step - Smoothing === %%

    % In the smoothing step of the E-Step, we refine our estimates of the states
    % by incorporating information from all the sequences, both past and future.
    % This contrasts with the filtering step, where estimates are based only on
    % past and current sequences.

    % 1. Initialization:
    % Begin with the estimates from the forward filtering algorithm for the last sequence.
    x_k_K(:, end) = x_k_k(:, end);
    W_k_K(:, :, end) = W_k_k(:, :, end);

    % 2. Backward Pass:
    % For each sequence k from K-1 down to 1:
    for k = K-1:-1:1
        % Compute the smoothing gain M_k
        epsilon = 1e-5; % Small regularization term
        M_k(:, :, k) = W_k_k(:, :, k) / (W_k_k_1(:, :, k+1) + epsilon * eye(size(W_k_k_1, 1)));

        % Update the smoothed state estimate x_k_K
        x_k_K(:, k) = x_k_k(:, k) + M_k(:, :, k) * (x_k_K(:, k+1) - x_k_k_1(:, k+1));

        % Update the smoothed state covariance W_k_K
        W_k_K(:, :, k) = W_k_k(:, :, k) + M_k(:, :, k) * (W_k_K(:, :, k+1) - W_k_k_1(:, :, k+1)) * M_k(:, :, k).';

        % Update the Second moment of the state estimate
        x_k_K_2(:, :, k) = W_k_K(:, :, k) + x_k_K(:, k)' * x_k_K(:, k);
    end

    % Outcome:
    % At the end of the smoothing step, we have refined estimates of the states
    % for each sequence, x_k_K and W_k_K, that incorporate information from all sequences.

    % Compute W_(k,u|K) and expected log-softmax values
    W_k_u_K = zeros(R, R, K, K);
    expected_log_softmax = zeros(R, K);

    for k = 1:K-1
        % Compute W_(k,k+1|K)
        W_k_u_K(:, :, k, k+1) = M_k(:, :, k) * W_k_K(:, :, k+1);

        % Compute expected log-softmax value for x_k_K
        f_val = softmaxMC(x_k_K(:, k), num_phrases);
        expected_log_softmax(:, k) = log(f_val) - 0.5 * f_val .* (1 - f_val) .* diag(W_k_K(:, :, k)).^(-0.5);
    end

    %% === M-Step === %%

    % In the M-step, we maximize the expected complete-data log-likelihood with respect to the parameters.
    % This involves updating the parameters to values that are most likely given the observed data and the current state estimates.
    % 1. The covariance matrix Sigma is updated based on the expected squared differences between consecutive smoothed states.
    % 2. The initial state x_0 is set to the expected value of the first smoothed state.

    % 1. Update Sigma - My Way
    sum_term = zeros(R, R);
    for k = 2:K
        delta_x = x_k_K(:, k) - x_k_K(:, k-1);
        sum_term = sum_term + delta_x * delta_x';
    end
    SigmaNew = diag(diag(sum_term)) / K;

    % Introduce a Learning Rate
    alpha = 1/(10000*MAX_ITERATIONS); 
    % Sigma = (1 - alpha) * Sigma + alpha * SigmaNew;

    % % 1. Update Sigma - Yarden's Way
    % SigmaNew = x_k_K_2(:,:,1) - x_0 * x_k_K(:,1) - x_k_K(:,1) * x_0' + x_0 * x_0';
    %
    % for k = 2:K
    %     SigmaNew = SigmaNew + x_k_K_2(:,:,k) + x_k_K_2(:,:,k-1) ...
    %         - W_k_u_K(:,:,k,k-1) - x_k_K(:,k) * x_k_K(:,k-1)' ...
    %         - W_k_u_K(:,:,k-1,k) - x_k_K(:,k-1) * x_k_K(:,k)';
    % end
    %
    % SigmaNew = SigmaNew / K;
    %
    % % Ensure positive definiteness
    % [v, d] = eig(SigmaNew);
    % Sigma = v * diag(max(0, diag(d))) * v';

    % 2. Update x_0
    x_0 = x_k_K(:, 1);

    % 3. convergence measure
    % convergence_measure(Current_EM_iter) = max(abs(softmaxMC(x_0, num_phrases) ...
    %     - softmaxMC(x_init, num_phrases)));
    convergence_measure(Current_EM_iter) = JSDiv(softmaxMC(x_0, num_phrases)', ...
        softmaxMC(x_init, num_phrases)');
    % convergence_measure(Current_EM_iter) = max(abs(x_k_k(:, k) - x_k_k_1(:, k)));

    %% === Visualization === %%

    % Clear the current figure content
    clf;

    sgtitle({"Current Iteration: " + num2str(Current_EM_iter), ...
        "Dynamic Learning Process : x^{k} = x^{k-1} + A, K = "+num2str(K)+ ...
        ", Sequence Range = ["+num2str(seq_range(1))+", "+num2str(seq_range(2))+"]"})

    % Calculate Probabilities from Logits x
    probs_sim        = softmaxMC(x_k_K(:, 1), num_phrases);
    probs_real       = softmaxMC(x_init, num_phrases);
    probs_final_real = softmaxMC(x_init + n_sequences * A, num_phrases);

    % Subplot 1: Simulated x_0 probabilities
    subplot(4, 2, 1);
    bar(probs_sim(1:num_phrases));
    title('Simulated Probabilities - Softmax(x_0(a,b,c))');
    ylim([0 1]);

    % Subplot 2: Real x_init probabilities
    subplot(4, 2, 2);
    bar(probs_real(1:num_phrases));
    title('Real Probabilities - Softmax(x_{init}(a,b,c))');
    ylim([0 1]);

    % Subplot 3: Simulated transition matrix
    subplot(4, 2, 3);
    imagesc(reshape(probs_sim(num_phrases + 1:end), [num_phrases, num_phrases])');
    title('Simulated Transition Matrix - Softmax(x_0(a|a,...,c|c))');
    clim([0 1])
    colorbar;

    % Subplot 4: Real transition matrix
    subplot(4, 2, 4);
    imagesc(reshape(probs_real(num_phrases + 1:end), [num_phrases, num_phrases])');
    title('Real Transition Matrix - Softmax(x_{init}(a|a,...,c|c))');
    clim([0 1])
    colorbar;

    % Subplot 5: Simulated Sigma
    subplot(4, 2, 5);
    imagesc(Sigma);
    title('Simulated Sigma');
    colorbar;

    % Subplot 6: Real Sigma
    subplot(4, 2, 6);
    imagesc(reshape(probs_final_real(num_phrases + 1:end), [num_phrases, num_phrases])');
    title('Real Transition Matrix - Softmax(x_{final}(a|a,...,c|c))');
    clim([0 1])
    colorbar;

    % Subplot 7+8: Convergence measure
    subplot(4, 2, [7, 8]);
    plot(1:Current_EM_iter, convergence_measure(1:Current_EM_iter));
    title('Convergence Measure vs. Iteration');
    xlabel('Iteration');
    ylabel('CM');

    % Pause to create an animation effect
    pause(0.005);

    % % Capture the current plot as a frame
    % currentFrame = getframe(gcf);
    % writeVideo(videoObj, currentFrame);

    % Update iteration counter
    Current_EM_iter = Current_EM_iter + 1;

    % Check for convergence
    if convergence_measure(Current_EM_iter - 1) < EM_CONV_LIMIT
        break;
    end
end

%% Close the video object
% close(videoObj);
close all;