function d = stoiScore(clean, est, fs)
%STOISCORE Short-time objective intelligibility (STOI), Taal et al. 2011.
%   d = STOISCORE(clean, est, fs) computes the STOI of the degraded
%   signal est relative to the clean reference clean, following
%   C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen, "An Algorithm
%   for Intelligibility Prediction of Time-Frequency Weighted Noisy
%   Speech", IEEE Trans. Audio Speech Lang. Process. 19(7), 2011, and
%   mirroring the reference implementation of that algorithm:
%
%   1. Both signals are resampled to 10 kHz.
%   2. Silent frames are removed in the time domain: 256-sample
%      Hann-windowed frames (hop 128) whose energy is more than 40 dB
%      below the loudest clean frame are dropped, and the kept frames
%      are overlap-added back into continuous signals.
%   3. STFT with a 256-sample symmetric Hann window, 512-point FFT,
%      hop 128.
%   4. DFT bins are grouped into 15 one-third-octave bands (band edges
%      snapped to the nearest bin); band envelopes are the root-sum of
%      squares per band.
%   5. For every band and every 30-frame segment, the degraded envelope
%      is energy-normalised to the clean envelope, clipped at a -15 dB
%      signal-to-distortion bound, and Pearson-correlated with it.
%      STOI is the mean correlation — at most 1, and in [0, 1] for
%      degraded speech in practice; STOI(x, x) = 1.
%
%   Clean-room implementation from the paper, included so the project
%   needs no toolbox. It was cross-validated against pystoi, a
%   reference implementation of Taal et al. (2011), on the committed
%   sample pairs: agreement within 0.0005 on every file (e.g. noisy
%   p232_001: 0.8976 reference, 0.8971 here). An earlier draft of this function used a 512-sample
%   analysis window, which over-smooths the envelopes and inflates the
%   score (0.9997 on the same file); the window length is load-bearing
%   and is covered by the unit tests' range checks.

    clean = clean(:);
    est = est(:);
    n = min(numel(clean), numel(est));
    clean = clean(1:n);
    est = est(1:n);
    if fs ~= 10000
        clean = resampleFFT(clean, fs, 10000);
        est = resampleFFT(est, fs, 10000);
    end

    [clean, est] = removeSilentFrames(clean, est);
    S = abs(localStft(clean));
    E = abs(localStft(est));
    numFrames = min(size(S, 2), size(E, 2));
    S = S(:, 1:numFrames);
    E = E(:, 1:numFrames);

    % One-third-octave band envelopes (15 bands from 150 Hz, band edges
    % snapped to the nearest DFT bin, as in the reference).
    numBands = 15;
    binFreqs = (0:256)' * (10000 / 512);
    envS = zeros(numBands, numFrames);
    envE = zeros(numBands, numFrames);
    for j = 1:numBands
        fLow = 150 * 2 ^ ((2 * (j - 1) - 1) / 6);
        fHigh = 150 * 2 ^ ((2 * (j - 1) + 1) / 6);
        [~, fl] = min((binFreqs - fLow) .^ 2);
        [~, fh] = min((binFreqs - fHigh) .^ 2);
        rows = fl:(fh - 1);
        envS(j, :) = sqrt(sum(S(rows, :) .^ 2, 1));
        envE(j, :) = sqrt(sum(E(rows, :) .^ 2, 1));
    end

    % Segment-wise envelope correlation with normalisation and clipping.
    segLen = 30;
    clipFactor = 1 + 10 ^ (15 / 20);   % -15 dB SDR clipping bound
    if numFrames < segLen
        d = NaN;
        return;
    end
    numSegs = numFrames - segLen + 1;
    total = 0;
    count = 0;
    for j = 1:numBands
        for m = 1:numSegs
            x = envS(j, m:m + segLen - 1)';
            y = envE(j, m:m + segLen - 1)';
            nx = norm(x);
            ny = norm(y);
            if nx == 0
                continue;
            end
            if ny > 0
                y = y * (nx / ny);
            end
            y = min(y, clipFactor * x);
            xc = x - mean(x);
            yc = y - mean(y);
            denom = norm(xc) * norm(yc);
            if denom == 0
                continue;
            end
            total = total + (xc' * yc) / denom;
            count = count + 1;
        end
    end
    if count == 0
        d = NaN;
    else
        d = total / count;
    end
end

function [xOut, yOut] = removeSilentFrames(x, y)
%REMOVESILENTFRAMES Drop low-energy frames and overlap-add the rest.
%   Mirrors the reference implementation: frame energies come from
%   Hann-windowed 256-sample frames (hop 128) of the clean signal;
%   frames more than 40 dB below the loudest frame are removed from
%   both signals, and the kept (windowed) frames are overlap-added
%   without envelope normalisation.
    winLen = 256;
    hop = 128;
    w = hannSymmetric(winLen);
    starts = 1:hop:(numel(x) - winLen);
    numFrames = numel(starts);
    xFrames = zeros(winLen, numFrames);
    yFrames = zeros(winLen, numFrames);
    energies = zeros(1, numFrames);
    for i = 1:numFrames
        idx = starts(i):(starts(i) + winLen - 1);
        xf = x(idx) .* w;
        xFrames(:, i) = xf;
        yFrames(:, i) = y(idx) .* w;
        energies(i) = 20 * log10(norm(xf) + eps);
    end
    keep = (max(energies) - 40 - energies) < 0;
    xOut = overlapAdd(xFrames(:, keep), hop, winLen);
    yOut = overlapAdd(yFrames(:, keep), hop, winLen);
end

function buf = overlapAdd(frames, hop, winLen)
    numFrames = size(frames, 2);
    if numFrames == 0
        buf = zeros(0, 1);
        return;
    end
    buf = zeros((numFrames - 1) * hop + winLen, 1);
    for i = 1:numFrames
        idx = (i - 1) * hop + (1:winLen);
        buf(idx) = buf(idx) + frames(:, i);
    end
end

function S = localStft(x)
%LOCALSTFT 256-sample symmetric Hann window, 512-point FFT, hop 128.
    winLen = 256;
    hop = 128;
    w = hannSymmetric(winLen);
    starts = 1:hop:(numel(x) - winLen);
    numFrames = numel(starts);
    S = zeros(257, numFrames);
    for i = 1:numFrames
        idx = starts(i):(starts(i) + winLen - 1);
        F = fft(x(idx) .* w, 512);
        S(:, i) = F(1:257);
    end
end

function w = hannSymmetric(n)
%HANNSYMMETRIC Symmetric Hann window (as used by the STOI reference).
    if n == 1
        w = 1;
        return;
    end
    k = (0:n - 1)';
    w = 0.5 - 0.5 * cos(2 * pi * k / (n - 1));
end
