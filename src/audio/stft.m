function [S, numFrames] = stft(x, winLen, hop)
%STFT Short-time Fourier transform (toolbox-free).
%   S = STFT(x, winLen, hop) computes the complex STFT of the real signal
%   x using a periodic Hann window of length winLen and the given hop.
%   The signal is padded with winLen/2 zeros on both sides so that the
%   first frame is centred on the first sample; ISTFT removes the
%   padding again, so STFT/ISTFT round-trip exactly.
%
%   S has size (winLen/2+1) x numFrames and contains the one-sided
%   spectrum. Only base MATLAB (fft) is used.

    x = x(:);
    win = hannPeriodic(winLen);
    pad = winLen / 2;
    xp = [zeros(pad, 1); x; zeros(pad, 1)];
    numFrames = floor((numel(xp) - winLen) / hop) + 1;
    S = zeros(winLen / 2 + 1, numFrames);
    for m = 1:numFrames
        idx = (m - 1) * hop + (1:winLen);
        frame = xp(idx) .* win;
        F = fft(frame, winLen);
        S(:, m) = F(1:winLen / 2 + 1);
    end
end
