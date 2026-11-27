function x = istft(S, winLen, hop, outLen)
%ISTFT Inverse short-time Fourier transform (toolbox-free).
%   x = ISTFT(S, winLen, hop, outLen) reconstructs a real time signal of
%   length outLen from the one-sided STFT S produced by STFT, using
%   weighted overlap-add (the periodic Hann window with hop = winLen/2
%   or winLen/4 satisfies the constant-overlap-add condition; the
%   window-envelope normalisation makes the reconstruction exact for
%   any hop that divides winLen).

    if mod(winLen, 2) ~= 0 || winLen < 2
        error('istft:BadWinLen', 'winLen must be a positive even integer.');
    end
    if hop < 1 || hop > winLen
        error('istft:BadHop', 'hop must be between 1 and winLen.');
    end
    numFrames = size(S, 2);
    win = hannPeriodic(winLen);
    pad = winLen / 2;
    bufLen = (numFrames - 1) * hop + winLen;
    buf = zeros(bufLen, 1);
    env = zeros(bufLen, 1);
    full = zeros(winLen, 1);
    for m = 1:numFrames
        half = S(:, m);
        full(1:winLen / 2 + 1) = half;
        full(winLen / 2 + 2:end) = conj(half(winLen / 2:-1:2));
        frame = real(ifft(full, winLen)) .* win;
        idx = (m - 1) * hop + (1:winLen);
        buf(idx) = buf(idx) + frame;
        env(idx) = env(idx) + win .^ 2;
    end
    nz = env > 1e-12;
    buf(nz) = buf(nz) ./ env(nz);
    if pad + outLen > bufLen
        error('istft:BadLength', ...
            'outLen %d exceeds the reconstructed buffer (%d samples).', outLen, bufLen - pad);
    end
    % Remove the centre-padding added by STFT and restore the length.
    x = buf(pad + 1:pad + outLen);
end
