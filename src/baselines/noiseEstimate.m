function noiseMag = noiseEstimate(magSpectrogram)
%NOISEESTIMATE Per-frequency noise magnitude estimate from a mixture.
%   noiseMag = NOISEESTIMATE(magSpectrogram) estimates the noise level
%   in each frequency bin as the 10th percentile of that bin's magnitude
%   over time. Speech is sparse in time-frequency, so a low percentile
%   tracks the noise floor without needing a noise-only segment at the
%   start of the file (VoiceBank-DEMAND mixtures do not guarantee one).
%   The estimate is lightly smoothed across frequency.

    % 10th percentile over time, computed directly (no toolbox needed).
    sortedMag = sort(magSpectrogram, 2);
    idx = max(1, ceil(0.10 * size(magSpectrogram, 2)));
    p = sortedMag(:, idx);
    % Smooth over neighbouring bins (3-tap) to reduce variance.
    noiseMag = p;
    if numel(p) > 2
        noiseMag(2:end - 1) = (p(1:end - 2) + 2 * p(2:end - 1) + p(3:end)) / 4;
        noiseMag(1) = (3 * p(1) + p(2)) / 4;
        noiseMag(end) = (p(end - 1) + 3 * p(end)) / 4;
    end
    noiseMag = max(noiseMag, 1e-12);
end
