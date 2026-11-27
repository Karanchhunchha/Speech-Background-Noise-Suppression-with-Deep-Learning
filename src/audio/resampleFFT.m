function y = resampleFFT(x, fsIn, fsOut)
%RESAMPLEFFT Band-limited resampling by spectrum truncation/padding.
%   y = RESAMPLEFFT(x, fsIn, fsOut) resamples the signal x from fsIn to
%   fsOut by truncating or zero-padding its DFT. This is exact for
%   band-limited signals and needs no toolbox. It is used to bring
%   48 kHz VoiceBank-DEMAND audio to 16 kHz and inside the STOI metric
%   (which is defined at 10 kHz).

    x = x(:);
    if fsIn == fsOut
        y = x;
        return;
    end
    nIn = numel(x);
    nOut = round(nIn * fsOut / fsIn);
    X = fft(x);
    Y = zeros(nOut, 1);
    if fsOut <= fsIn
        % Downsample: keep the low-frequency part of the spectrum.
        keep = floor(nOut / 2);
        Y(1:keep + 1) = X(1:keep + 1);
        Y(nOut - keep + 1:nOut) = X(nIn - keep + 1:nIn);
        if mod(nOut, 2) == 0
            % Nyquist bin is shared between positive/negative sides.
            Y(nOut / 2 + 1) = X(nOut / 2 + 1) + X(nIn - nOut / 2 + 1);
        end
    else
        % Upsample: embed the spectrum in a longer zero spectrum.
        keep = floor(nIn / 2);
        Y(1:keep + 1) = X(1:keep + 1);
        Y(nOut - keep + 1:nOut) = X(nIn - keep + 1:nIn);
        if mod(nIn, 2) == 0
            Y(nIn / 2 + 1) = X(nIn / 2 + 1) / 2;
            Y(nOut - nIn / 2 + 1) = X(nIn / 2 + 1) / 2;
        end
    end
    y = real(ifft(Y)) * (nOut / nIn);
end
