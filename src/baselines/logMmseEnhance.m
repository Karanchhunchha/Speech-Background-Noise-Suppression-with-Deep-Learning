function enhanced = logMmseEnhance(noisy, fs)
%LOGMMSEENHANCE Log-MMSE spectral amplitude estimation (Ephraim-Malah).
%   enhanced = LOGMMSEENHANCE(noisy, fs) enhances the noisy signal with
%   the log-spectral amplitude estimator of Ephraim & Malah (1985),
%   the canonical strong classical reference in the Loizou benchmark
%   tradition. Where the MMSE-STSA gain (see WIENERENHANCE) minimises
%   the squared error of the spectral amplitude, the log-MMSE gain
%   minimises the squared error of the LOG amplitude, matching the
%   ear's roughly logarithmic loudness perception:
%
%       G = (xi / (1 + xi)) * exp( 0.5 * E1(nu) ),
%       nu = gamma * xi / (1 + xi),
%
%   with a posteriori SNR gamma = |Y|^2 / noisePower, a priori SNR xi
%   from the same decision-directed recursion (smoothing 0.98) the
%   Wiener baseline uses, the same NOISEESTIMATE noise floor, and E1
%   the exponential integral (base-MATLAB EXPINT). The practical
%   effect versus MMSE-STSA is less aggressive suppression at low
%   SNR: less speech distortion and less musical noise, at the cost
%   of slightly more residual noise. fs is validated (positive
%   scalar) for interface consistency but otherwise unused — the
%   framing is defined in samples.

    if ~(isscalar(fs) && fs > 0)
        error('logMmseEnhance:BadFs', 'fs must be a positive scalar.');
    end
    winLen = 512;
    hop = 256;
    S = stft(noisy, winLen, hop);
    mag = abs(S);
    phase = angle(S);
    noisePower = noiseEstimate(mag) .^ 2;
    noisyPower = mag .^ 2;
    numFrames = size(S, 2);
    gain = zeros(size(S));
    posteriorPrev = zeros(size(S, 1), 1);
    smoothing = 0.98;
    for m = 1:numFrames
        posterior = noisyPower(:, m) ./ noisePower;
        if m == 1
            apriori = max(posterior - 1, 0);
        else
            dd = (gain(:, m - 1) .^ 2) .* posteriorPrev;
            apriori = smoothing * dd + (1 - smoothing) * max(posterior - 1, 0);
        end
        nu = posterior .* apriori ./ (1 + apriori);
        % E1(nu) diverges as nu -> 0; flooring nu caps the gain at a
        % finite value, and the gain is additionally capped at 1 (the
        % estimator never amplifies).
        g = (apriori ./ (1 + apriori)) .* exp(0.5 * expint(max(nu, 1e-6)));
        gain(:, m) = min(g, 1);
        posteriorPrev = posterior;
    end
    enhanced = istft((gain .* mag) .* exp(1i * phase), winLen, hop, numel(noisy));
end
