%% Geometry-Structured Model: ULA
%% Sweep Parameters
M = [8 16 32 64]; % Number of BS antennas
K = [2 3 4 5]; % Number of users

angular_sep = linspace(1e-6, 0.1, 100); % user angular separation
%% Physical Parameters
fc = 1e8; % carrier frequency
c = 3e8;  % speed of light
lambda = c/fc; % wavelength
k = 2*pi/lambda; % wavenumber
d = lambda/2; % distance 
MC = 500; % Monte Carlo Trials

%% SNR
SNR_dB = 10; % SNR (dB)
SNR = 10^(SNR_dB / 10); % SNR (linear)
Es = 1; % Symbol Energy                    
N0 = Es / SNR; % Noise Variance


% Initialize
cond_ULA = zeros(length(M), length(angular_sep), length(K), MC);
rate_ULA = zeros(length(M), length(angular_sep), length(K), MC);
rank_ULA = zeros(length(M), length(angular_sep), length(K),MC);

for i = 1:length(M) % Sweep through BS antennas
    m = M(i);
    for ii = 1:length(K)
        kk = K(ii);
        for n = 1:length(angular_sep) % Sweep through angular separation between users
            delta = deg2rad(angular_sep(n));
            for mc = 1:MC % Monte Carlo Trials
                user_angles = (0:kk-1)*delta;
                H = zeros(m,kk);
                for user = 1:kk % Sweep through users
                     phi = k*d*sin(user_angles(user));
                     H(:,user) = exp(1j*(0:m-1)'*phi);  
                end
                H = H./sqrt(m);
                cond_ULA(i,n,ii,mc) = cond(H);
                rate_ULA(i,n,ii,mc) = ZF_sumrate(H,N0);
                rank_ULA(i,n,ii,mc) = sum(svd(H) > 1e-4);
            end
        end
    end
end

avg_cond_ULA = mean(cond_ULA,4);
avg_rate_ULA = mean(rate_ULA,4);
avg_rank_ULA = mean(rank_ULA,4);

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

%% Convert radians to degrees
angular_deg = angular_sep * 180/pi;

% Condition Number vs Angular Separation
figure;
for nK = 1:length(K)
    subplot(2,2,nK);
    hold on; grid on;
    for i = 1:length(M)
        semilogy(angular_deg, squeeze(avg_cond_ULA(i,:,nK)), 'LineWidth', 2);
    end
    xlabel('User Angular Separation (degrees)');
    ylabel('Avg Condition Number');
    title(['K = ', num2str(K(nK))]);
    set(gca, 'XDir','reverse');
    legend('M=8','M=16','M=32','M=64', 'Location','best');
end
sgtitle('ULA MU-MIMO: Condition Number vs Angular Separation');

% ZF Sum-Rate vs Angular Separation
figure;
for nK = 1:length(K)
    subplot(2,2,nK);
    hold on; grid on;
    for i = 1:length(M)
        plot(angular_deg, squeeze(avg_rate_ULA(i,:,nK)), 'LineWidth', 2);
    end
    xlabel('User Angular Separation (degrees)');
    ylabel('Avg ZF Sum-Rate (bits/s/Hz)');
    title(['K = ', num2str(K(nK))]);
    set(gca, 'XDir','reverse');
    legend('M=8','M=16','M=32','M=64', 'Location','best');
    yline(0, ':', 'LineWidth', 1.5);
end
sgtitle('ULA MU-MIMO: ZF Sum-Rate vs Angular Separation');

% Average Rank vs Angular Separation
figure;
for nK = 1:length(K)
    subplot(2,2,nK);
    hold on; grid on;
    for i = 1:length(M)
        plot(angular_deg, squeeze(avg_rank_ULA(i,:,nK)), 'LineWidth', 2);
    end
    xlabel('User Angular Separation (degrees)');
    ylabel('Avg Rank');
    title(['K = ', num2str(K(nK))]);
    set(gca, 'XDir','reverse');
    legend('M=8','M=16','M=32','M=64', 'Location','best');
    yline(1, ':', 'LineWidth', 1.5);
end
sgtitle('ULA MU-MIMO: Rank vs Angular Separation');
