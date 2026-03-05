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
            rank_keyhole(i,ii,mc) = sum(svd(H) > 1e-4);
        end
    end
end

% Average results over Monte Carlo
avg_cond_Keyhole = mean(cond_keyhole,3);
avg_rate_Keyhole = mean(rate_keyhole,3);
avg_rank_Keyhole = mean(rank_keyhole,3);

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
    Ginv = inv(G);

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
set(gca, 'XDir','reverse');

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
set(gca, 'XDir','reverse');

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
set(gca, 'XDir','reverse');

xlim([min(K)-0.5 max(K)+0.5])
