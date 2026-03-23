%% Geometry-Structured Model: ULA
%% Sweep Parameters
M = [8 16 32 64]; % Number of BS antennas
K = [2 3 4 5]; % Number of users
angular_sep = linspace(1e-6, 0.05, 100); % user angular separation

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
SINR_ULA = zeros(length(M), length(angular_sep), length(K),MC, max(K));
Noise_ULA = zeros(length(M), length(angular_sep), length(K), MC);
sigma_min_ULA = zeros(length(M), length(angular_sep), length(K), MC);

for i = 1:length(M) % Sweep through BS antennas
    m = M(i);
    for ii = 1:length(K)
        kk = K(ii);
        for n = 1:length(angular_sep) % Sweep through angular separation between users
            delta = angular_sep(n);
            for mc = 1:MC % Monte Carlo Trials
                user_angles = (0:kk-1)*delta;
                H = zeros(m,kk);
                for user = 1:kk % Sweep through users
                     phi = k*d*sin(user_angles(user));
                     H(:,user) = exp(1j*(0:m-1)'*phi) .* exp(1j*2*pi*rand);
                end
                H = H./sqrt(m);
                cond_ULA(i,n,ii,mc) = cond(H+1e-10*randn(size(H)));
                rate_ULA(i,n,ii,mc) = ZF_sumrate(H,N0);
                s = svd(H);
                rank_ULA(i,n,ii,mc) = sum(s > 1e-3 * max(s));
                SINR_ULA(i,n,ii,mc, 1:kk) = ZF_SINR(H, N0);
                Noise_ULA(i,n,ii,mc) = ZF_noise_amp(H);
                sigma_min_ULA(i,n,ii,mc) = min(svd(H));
            end
        end
    end
end

avg_cond_ULA = mean(cond_ULA,4);
avg_rate_ULA = mean(rate_ULA,4);
avg_rank_ULA = mean(rank_ULA,4);
avg_noise_ULA = mean(Noise_ULA,4);
avg_sigma_min_ULA = mean(sigma_min_ULA,4);

avg_sinr_ULA = zeros(length(M), length(angular_sep), length(K));

for i = 1:length(M)
    for n = 1:length(angular_sep)
        for ii = 1:length(K)
            tmp = squeeze(SINR_ULA(i,n,ii,:,:));
            tmp = tmp(:);
            tmp = tmp(tmp > 0);
            avg_sinr_ULA(i,n,ii) = mean(tmp);
        end
    end
end

%% Functions

function [R] = ZF_sumrate(H,N0)
    K = size(H,2);
    G = H'*H;
    Ginv = pinv(G);

    snr = zeros(K,1);

    for j = 1:K % Calculate SNR for each user
        snr(j) = 1/(N0*Ginv(j,j));
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

%% Threshold
sigma_thresh = 1e-2;        % near-zero singular value
cond_thresh  = 1e2;         % ill-conditioned
sinr_thresh  = 0;           % low sinr (dB)
rate_frac    = 0.2;         % 20% of ULA baseline

angular_deg = angular_sep * 180/pi;

%% Threshold Extraction
threshold_sigma = zeros(length(M), length(K));
threshold_cond  = zeros(length(M), length(K));
threshold_rate  = zeros(length(M), length(K));
threshold_noise = zeros(length(M), length(K));
threshold_sinr  = zeros(length(M), length(K));

for i = 1:length(M)
    for nK = 1:length(K)

        % Extract curves
        sigma_curve = squeeze(avg_sigma_min_ULA(i,:,nK));
        cond_curve  = squeeze(avg_cond_ULA(i,:,nK));
        rate_curve  = squeeze(avg_rate_ULA(i,:,nK));
        noise_curve = squeeze(avg_noise_ULA(i,:,nK));
        sinr_curve  = 10*log10(squeeze(avg_sinr_ULA(i,:,nK)));

        % ULA baseline (same M, K)
        Rayleigh_rate = avg_rate_Rayleigh(i,nK);

        % Minimum singular value threshold
        idx = find(sigma_curve < sigma_thresh, 1, 'first');
        if ~isempty(idx)
            threshold_sigma(i,nK) = angular_deg(idx);
        else
            threshold_sigma(i,nK) = NaN;
        end

        % Condition number threshold 
        idx = find(cond_curve > cond_thresh, 1, 'first');
        if ~isempty(idx)
            threshold_cond(i,nK) = angular_deg(idx);
        else
            threshold_cond(i,nK) = NaN;
        end

        % Rate degradation threshold 
        idx = find(rate_curve < rate_frac * Rayleigh_rate, 1, 'first');
        if ~isempty(idx)
            threshold_rate(i,nK) = angular_deg(idx);
        else
            threshold_rate(i,nK) = NaN;
        end

        % Noise amplification threshold 
        noise_thresh_dyn = 0.4*max(noise_curve);   
        idx = find(noise_curve > noise_thresh_dyn, 1, 'first');
        if ~isempty(idx)
            threshold_noise(i,nK) = angular_deg(idx);
        else
            threshold_noise(i,nK) = NaN;
        end

        % SINR threshold
        idx = find(sinr_curve < sinr_thresh, 1, 'first');
        if ~isempty(idx)
            threshold_sinr(i,nK) = angular_deg(idx);
        else
            threshold_sinr(i,nK) = NaN;
        end

    end
end

%% Graph Everything (with Threshold Points)
colors = lines(length(M));

% Condition Number
figure;
for nK = 1:length(K)
    subplot(2,2,nK);
    hold on; grid on;

    h = gobjects(length(M),1);

    for i = 1:length(M)
        y = squeeze(avg_cond_ULA(i,:,nK));
        h(i) = semilogy(angular_deg, y, 'LineWidth', 2, 'Color', colors(i,:));

        % Threshold marker
        if ~isnan(threshold_cond(i,nK))
            [~, idx] = min(abs(angular_deg - threshold_cond(i,nK)));
            semilogy(angular_deg(idx), y(idx), 'o', ...
                'MarkerSize', 8, ...
                'MarkerFaceColor', colors(i,:), ...
                'MarkerEdgeColor', 'k');
        end
    end

    xlabel('User Angular Separation (degrees)');
    ylabel('Avg Condition Number');
    title(['K = ', num2str(K(nK))]);
    set(gca, 'XDir','reverse');

    legend(h, {'M=8','M=16','M=32','M=64'}, 'Location','best');
end
sgtitle('ULA MU-MIMO: Condition Number vs Angular Separation');

% ZF Sum Rate
figure;
for nK = 1:length(K)
    subplot(2,2,nK);
    hold on; grid on;

    h = gobjects(length(M),1);

    for i = 1:length(M)
        y = squeeze(avg_rate_ULA(i,:,nK));
        h(i) = plot(angular_deg, y, 'LineWidth', 2, 'Color', colors(i,:));

        % Threshold marker
        if ~isnan(threshold_rate(i,nK))
            [~, idx] = min(abs(angular_deg - threshold_rate(i,nK)));
            plot(angular_deg(idx), y(idx), 'o', ...
                'MarkerSize', 8, ...
                'MarkerFaceColor', colors(i,:), ...
                'MarkerEdgeColor', 'k');
        end
    end

    yline(0, ':', 'HandleVisibility','off');

    xlabel('User Angular Separation (degrees)');
    ylabel('Avg ZF Sum-Rate (bits/s/Hz)');
    title(['K = ', num2str(K(nK))]);
    set(gca, 'XDir','reverse');

    legend(h, {'M=8','M=16','M=32','M=64'}, 'Location','best');
end
sgtitle('ULA MU-MIMO: ZF Sum-Rate vs Angular Separation');


% Rank
figure;
for nK = 1:length(K)
    subplot(2,2,nK);
    hold on; grid on;

    h = gobjects(length(M),1);

    for i = 1:length(M)
        y = squeeze(avg_rank_ULA(i,:,nK));
        h(i) = plot(angular_deg, y, 'LineWidth', 2, 'Color', colors(i,:));

        % Threshold marker (σ_min-based)
        if ~isnan(threshold_sigma(i,nK))
            [~, idx] = min(abs(angular_deg - threshold_sigma(i,nK)));
            plot(angular_deg(idx), y(idx), 'o', ...
                'MarkerSize', 8, ...
                'MarkerFaceColor', colors(i,:), ...
                'MarkerEdgeColor', 'k');
        end
    end

    yline(1, ':', 'HandleVisibility','off');

    xlabel('User Angular Separation (degrees)');
    ylabel('Avg Rank');
    title(['K = ', num2str(K(nK))]);
    set(gca, 'XDir','reverse');

    legend(h, {'M=8','M=16','M=32','M=64'}, 'Location','best');
end
sgtitle('ULA MU-MIMO: Rank vs Angular Separation');

% Noise Amplification
figure;
for nK = 1:length(K)
    subplot(2,2,nK);
    hold on; grid on;

    h = gobjects(length(M),1);

    for i = 1:length(M)
        y = squeeze(avg_noise_ULA(i,:,nK));
        h(i) = plot(angular_deg, y, 'LineWidth', 2, 'Color', colors(i,:));

        % Threshold marker
        if ~isnan(threshold_noise(i,nK))
            [~, idx] = min(abs(angular_deg - threshold_noise(i,nK)));
            plot(angular_deg(idx), y(idx), 'o', ...
                'MarkerSize', 8, ...
                'MarkerFaceColor', colors(i,:), ...
                'MarkerEdgeColor', 'k');
        end
    end

    xlabel('Angular Separation (degrees)');
    ylabel('Noise Amplification');
    title(['K = ', num2str(K(nK))]);
    set(gca,'XDir','reverse');

    legend(h, {'M=8','M=16','M=32','M=64'}, 'Location','best');
end
sgtitle('Noise Amplification vs Angular Separation');


% SINR
figure;
for nK = 1:length(K)
    subplot(2,2,nK);
    hold on; grid on;

    h = gobjects(length(M),1);

    for i = 1:length(M)
        y = 10*log10(squeeze(avg_sinr_ULA(i,:,nK)));
        h(i) = plot(angular_deg, y, 'LineWidth', 2, 'Color', colors(i,:));

        % Threshold marker
        if ~isnan(threshold_sinr(i,nK))
            [~, idx] = min(abs(angular_deg - threshold_sinr(i,nK)));
            plot(angular_deg(idx), y(idx), 'o', ...
                'MarkerSize', 8, ...
                'MarkerFaceColor', colors(i,:), ...
                'MarkerEdgeColor', 'k');
        end
    end

    xlabel('Angular Separation (degrees)');
    ylabel('Avg SINR (dB)');
    title(['K = ', num2str(K(nK))]);
    set(gca,'XDir','reverse');
    yline(-100, ':', 'HandleVisibility','off');

    legend(h, {'M=8','M=16','M=32','M=64'}, 'Location','best');
end
sgtitle('Average SINR vs Angular Separation');


% Minimum Singular Value
figure;
for nK = 1:length(K)
    subplot(2,2,nK);
    hold on; grid on;

    h = gobjects(length(M),1);

    for i = 1:length(M)
        y = squeeze(avg_sigma_min_ULA(i,:,nK));
        h(i) = semilogy(angular_deg, y, 'LineWidth', 2, 'Color', colors(i,:));

        % Threshold marker
        if ~isnan(threshold_sigma(i,nK))
            [~, idx] = min(abs(angular_deg - threshold_sigma(i,nK)));
            semilogy(angular_deg(idx), y(idx), 'o', ...
                'MarkerSize', 8, ...
                'MarkerFaceColor', colors(i,:), ...
                'MarkerEdgeColor', 'k');
        end
    end

    xlabel('User Angular Separation (degrees)');
    ylabel('Min Singular Value');
    title(['K = ', num2str(K(nK))]);
    set(gca,'XDir','reverse');
    yline(0, ':', 'HandleVisibility','off');
    legend(h, {'M=8','M=16','M=32','M=64'}, 'Location','best');
end
sgtitle('Minimum Singular Value vs Angular Separation');
