function v = segSnrDb(clean, est)
%SEGSNRDB Segmental SNR (dB) of an estimate against clean speech.
%   v = SEGSNRDB(clean, est) averages the per-frame SNR over
%   non-overlapping 512-sample frames, clipping each frame's SNR to
%   [-10, 35] dB (the standard segmental-SNR convention of Hansen &
%   Quackenbush). Frames in which the clean signal is silent (energy
%   below 1e-10 of the loudest frame) are excluded so that pauses do
%   not dominate the score.

    n = min(numel(clean), numel(est));
    c = clean(1:n);
    e = est(1:n);
    frameLen = 512;
    numFrames = floor(n / frameLen);
    if numFrames == 0
        v = snrDb(c, e);
        return;
    end
    C = reshape(c(1:numFrames * frameLen), frameLen, numFrames);
    E = reshape(e(1:numFrames * frameLen), frameLen, numFrames);
    sigPow = sum(C .^ 2, 1);
    errPow = sum((C - E) .^ 2, 1);
    active = sigPow > 1e-10 * max(sigPow);
    if ~any(active)
        v = snrDb(c, e);
        return;
    end
    frameSnr = 10 * log10(sigPow(active) ./ max(errPow(active), eps));
    frameSnr = min(max(frameSnr, -10), 35);
    v = mean(frameSnr);
end
