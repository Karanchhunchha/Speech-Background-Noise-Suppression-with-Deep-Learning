function v = lsdDb(clean, est)
%LSDDB Log-spectral distance (dB) between clean and estimated speech.
%   v = LSDDB(clean, est) computes the mean over STFT frames of the
%   root-mean-square difference between the log power spectra of the
%   two signals (512-point Hann STFT, hop 256). Lower is better; a
%   perfect estimate returns 0.

    winLen = 512;
    hop = 256;
    S = stft(clean, winLen, hop);
    E = stft(est, winLen, hop);
    numFrames = min(size(S, 2), size(E, 2));
    logDiff = 10 * log10((abs(S(:, 1:numFrames)) .^ 2 + 1e-12) ./ ...
                         (abs(E(:, 1:numFrames)) .^ 2 + 1e-12));
    v = mean(sqrt(mean(logDiff .^ 2, 1)));
end
