function d = stoiScore(clean, est, fs)
%STOISCORE Short-time objective intelligibility (STOI), Taal et al. 2011.
%   d = STOISCORE(clean, est, fs) estimates the intelligibility of the
%   degraded signal est relative to the clean reference clean, following
%   C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen, "An Algorithm
%   for Intelligibility Prediction of Time-Frequency Weighted Noisy
%   Speech", IEEE Trans. Audio Speech Lang. Process. 19(7), 2011.
%
%   Method: both signals are resampled to 10 kHz (the rate STOI is
%   defined at), transformed with a 512-point Hann STFT (hop 256), and
%   grouped into 15 one-third-octave bands. Silent frames (clean-frame
%   energy more than 40 dB below the loudest frame) are excluded. For
%   every band and every 30-frame segment, the degraded temporal
%   envelope is energy-normalised to the clean envelope, clipped at a
%   -15 dB signal-to-distortion ratio, and correlated with it; STOI is
%   the mean correlation. The score lies in [0, 1]; STOI(x, x) = 1.
%
%   This is a clean-room implementation from the paper, included so the
%   project needs no toolbox; it is validated by unit tests (identity,
%   monotonicity in SNR) rather than against the reference code.

    n = min(numel(clean), numel(est));
    clean = clean(1:n);
    est = est(1:n);
    if fs ~= 10000
        clean = resampleFFT(clean, fs, 10000);
        est = resampleFFT(est, fs, 10000);
        fs = 10000;
    end

    winLen = 512;
    hop = 256;
    S = stft(clean, winLen, hop);
    E = stft(est, winLen, hop);
    numFrames = min(size(S, 2), size(E, 2));
    S = abs(S(:, 1:numFrames));
    E = abs(E(:, 1:numFrames));

    % Exclude silent frames (clean energy 40 dB below the loudest frame).
    frameEnergy = sum(S .^ 2, 1);
    active = frameEnergy >= max(frameEnergy) * 1e-4;
    S = S(:, active);
    E = E(:, active);
    numFrames = size(S, 2);

    % One-third-octave band envelopes (15 bands from 150 Hz).
    numBands = 15;
    binFreqs = (0:winLen / 2)' * (fs / winLen);
    envS = zeros(numBands, numFrames);
    envE = zeros(numBands, numFrames);
    for j = 1:numBands
        fc = 150 * 2 ^ ((j - 1) / 3);
        lo = fc * 2 ^ (-1 / 6);
        hi = fc * 2 ^ (1 / 6);
        bins = binFreqs >= lo & binFreqs < hi;
        envS(j, :) = sqrt(sum(S(bins, :) .^ 2, 1));
        envE(j, :) = sqrt(sum(E(bins, :) .^ 2, 1));
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
