function enhanced = wienerEnhance(noisy, fs)
%WIENERENHANCE Decision-directed Wiener filtering (Ephraim-Malah style).
%   enhanced = WIENERENHANCE(noisy, fs) enhances the noisy signal with a
%   Wiener gain driven by the decision-directed a priori SNR estimate
%   of Ephraim & Malah (1984): the a priori SNR follows the previous
%   frame's posterior SNR with smoothing factor 0.98, which suppresses
%   the musical noise that plain spectral subtraction leaves behind.
%   The noise power comes from NOISEESTIMATE. fs is accepted for
%   interface consistency and is currently unused.

    if ~(isscalar(fs) && fs > 0)
        error('wienerEnhance:BadFs', 'fs must be a positive scalar.');
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
        g = apriori ./ (1 + apriori);
        gain(:, m) = g;
        posteriorPrev = posterior;
    end
    enhanced = istft((gain .* mag) .* exp(1i * phase), winLen, hop, numel(noisy));
end
