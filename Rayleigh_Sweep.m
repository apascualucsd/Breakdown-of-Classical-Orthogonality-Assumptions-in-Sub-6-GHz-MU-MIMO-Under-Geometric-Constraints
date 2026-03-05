%% Baseline Model: Rayleigh Fading

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
cond_rayleigh = zeros(length(M), length(K), MC);
rate_rayleigh = zeros(length(M), length(K), MC);
rank_rayleigh= zeros(length(M), length(K), MC);

%% Rayleigh Parameter Sweep
for i = 1:length(M) % Sweep through BS antennas
    m = M(i);
    for ii = 1:length(K)
        k = K(ii);
        for mc = 1:MC % Monte Carlo Trials
            H = (randn(m,k) + 1j*randn(m,k))/sqrt(2);
            cond_rayleigh(i,ii,mc) = cond(H);
            rate_rayleigh(i,ii,mc) = ZF_sumrate(H,N0);
            rank_rayleigh(i,ii,mc) = sum(svd(H) > 1e-4);
        end
    end
end

% Average results over Monte Carlo
avg_cond_Rayleigh = mean(cond_rayleigh,3);
avg_rate_Rayleigh = mean(rate_rayleigh,3);
avg_rank_Rayleigh = mean(rank_rayleigh,3);

% Display average results
disp('Average Condition Numbers:');
disp(avg_cond_Rayleigh);
disp('Average Rates:');
disp(avg_rate_Rayleigh);
disp('Average Rank:');
disp(avg_rank_Rayleigh);

%% ZF Sum Rate Function

function [R] = ZF_sumrate(H,N0)
    K = size(H,2);
    G = H'*H;
    Ginv = inv(G);

    snr = zeros(K,1);

    for j = 1:K % Calculate SNR for each user
        snr(j) = 1/(N0*Ginv(j,j));
    end

    R = sum(log2(1+snr));
end
