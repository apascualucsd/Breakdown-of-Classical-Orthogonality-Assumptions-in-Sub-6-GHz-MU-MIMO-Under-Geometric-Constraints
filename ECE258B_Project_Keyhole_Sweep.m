%% Benchmark model: Keyhole

%% Sweep Parameters
M = [8 16 32 64]; % Number of BS antennas
K = [2 3 4 5]; % Number of users % Number of users
MC = 500; % Monte Carlo 

%% SNR
SNR_dB = 10; % SNR (dB)
SNR = 10^(SNR_dB / 10); % SNR (linear)
Es = 1; % Symbol Energy                    
N0 = Es / SNR; % Noise Variance

%% Initialize
cond_keyhole = zeros(length(M), length(K), MC);
rate_keyhole = zeros(length(M), length(K), MC);
rank_keyhole = zeros(length(M), length(K), MC);
SINR_keyhole = zeros(length(M), length(K), MC, max(K));
Noise_keyhole = zeros(length(M), length(K), MC);
sigma_min_keyhole = zeros(length(M), length(K), MC);

%% Rayleigh Parameter Sweep
for i = 1:length(M) % Sweep through BS antennas
    m = M(i);
    for ii = 1:length(K)
        k = K(ii);
        for mc = 1:MC % Monte Carlo Trials
            a = (randn(m,1) + 1j*randn(m,1))/sqrt(2);
            b = (randn(1,k) + 1j*randn(1,k))/sqrt(2);

             H = a * b;   % Rank-1
    
            cond_keyhole(i,ii,mc) = cond(H);
            rate_keyhole(i,ii,mc) = ZF_sumrate(H, N0);
            s = svd(H);
            rank_keyhole(i,ii,mc) = sum(s > 1e-3 * max(s));
            sigma_min_keyhole(i,ii,mc) = min(s);
            SINR_keyhole(i,ii,mc,1:k) = ZF_SINR(H,N0);
            Noise_keyhole(i,ii,mc) = ZF_noise_amp(H);
        end
    end
end

% Average results over Monte Carlo
avg_cond_Keyhole = mean(cond_keyhole,3);
avg_rate_Keyhole = mean(rate_keyhole,3);
avg_rank_Keyhole = mean(rank_keyhole,3);

avg_noise_Keyhole = mean(Noise_keyhole,3);
avg_sigma_min_Keyhole= mean(sigma_min_keyhole,3);

avg_sinr_Rayleigh = zeros(length(M), length(K));

for i = 1:length(M)
    for ii = 1:length(K)
        tmp = squeeze(SINR_keyhole(i,ii,:,:));
        tmp = tmp(:);
        tmp = tmp(tmp > 0);
        avg_sinr_Keyhole(i,ii) = mean(tmp);
    end
end

% Display average results
disp('Average Condition Numbers:');
disp(avg_cond_Keyhole);
disp('Average Rates:');
disp(avg_rate_Keyhole);
disp('Average Rank:');
disp(avg_rank_Keyhole);

%% ZF Sum Rate Function

function [R] = ZF_sumrate(H,N0)
    K = size(H,2);
    G = H'*H;
    Ginv = pinv(G);

    % Check rank
    if rank(H) < K
        R = 0;   % ZF not feasible
        return;
    end

    snr = zeros(K,1);

    for j = 1:K % Calculate SNR for each user
        snr(j) = 1/(N0*real(Ginv(j,j)));
    end

    R = sum(log2(1+snr));
end

function [sinr] = ZF_SINR(H,N0)
    K = size(H,2);
    G = H' * H;

    % Handle ill-conditioning safely
    if rank(H) < K
        sinr = zeros(K,1);
        return;
    end

    Ginv = pinv(G); 

    sinr = zeros(K,1);
    for j = 1:K
        sinr(j) = 1 / (N0 * real(Ginv(j,j)));
    end
end

function [noise_amp_avg] = ZF_noise_amp(H)
    K = size(H,2);
    G = H' * H;

    % Handle rank deficiency
    if rank(H) < K
        noise_amp_avg = inf;
        return;
    end

    Ginv = pinv(G);

    noise_amp = real(diag(Ginv));   % per-user
    noise_amp_avg = mean(noise_amp); % scalar summary
end


%% Plot Everything

% Rank Plot
figure;
hold on; grid on;

for i = 1:length(M)
    plot(K, avg_rank_Keyhole(i,:), '-o', 'LineWidth',2);
end

xlabel('Number of Users (K)');
ylabel('Average Channel Rank');
title('Keyhole Channel: Rank vs Number of Users');
legend('M=8','M=16','M=32','M=64','Location','best');


xlim([min(K)-0.5 max(K)+0.5])
ylim([0 max(avg_rank_Keyhole(:))*1.1])


% ZF Sum Rate Plot
figure;
hold on; grid on;

for i = 1:length(M)
    plot(K, real(avg_rate_Keyhole(i,:)), '-o', 'LineWidth',2);
end

xlabel('Number of Users (K)');
ylabel('Average ZF Sum-Rate (bits/s/Hz)');
title('Keyhole Channel: ZF Sum-Rate vs Number of Users');
legend('M=8','M=16','M=32','M=64','Location','best');


% Condition Number Plot
figure;
hold on; grid on;

for i = 1:length(M)
    semilogy(K, avg_cond_Keyhole(i,:), '-o', 'LineWidth',2);
end

xlabel('Number of Users (K)');
ylabel('Average Condition Number');
title('Keyhole Channel: Condition Number vs Number of Users');
legend('M=8','M=16','M=32','M=64','Location','best');


xlim([min(K)-0.5 max(K)+0.5])

% Noise Amplification
figure;
hold on; grid on;

for i = 1:length(M)
    plot(K, avg_noise_Keyhole(i,:), '-o', 'LineWidth',2);
end

xlabel('Number of Users (K)');
ylabel('Noise Amplification');
title('Keyhole Channel: Noise Amplification vs Users');
legend('M=8','M=16','M=32','M=64','Location','best');

% SINR
figure;
hold on; grid on;

for i = 1:length(M)
    plot(K, 10*log10(avg_sinr_Keyhole(i,:)), '-o', 'LineWidth',2);
end

xlabel('Number of Users (K)');
ylabel('Average SINR (dB)');
title('Keyhole Channel: SINR vs Users');
legend('M=8','M=16','M=32','M=64','Location','best');

% Minimum Singular Value 
figure;
hold on; grid on;

for i = 1:length(M)
    semilogy(K, avg_sigma_min_Keyhole(i,:), '-o', 'LineWidth',2);
end

xlabel('Number of Users (K)');
ylabel('Min Singular Value');
title('Keyhole Channel: σ_{min} vs Users');
legend('M=8','M=16','M=32','M=64','Location','best');