function enhanced = spectralSubtraction(noisy, fs)
%SPECTRALSUBTRACTION Boll spectral subtraction speech enhancement.
%   enhanced = SPECTRALSUBTRACTION(noisy, fs) enhances the noisy signal
%   with the classic power spectral subtraction of S. F. Boll (1979):
%   the estimated noise power spectrum is subtracted (with
%   over-subtraction factor alpha = 2 and spectral floor beta = 0.01 of
%   the noisy power, the common operating point) and the result is
%   resynthesised with the noisy phase. fs is validated (positive
%   scalar) for interface consistency but otherwise unused — the
%   framing is defined in samples.

    if ~(isscalar(fs) && fs > 0)
        error('spectralSubtraction:BadFs', 'fs must be a positive scalar.');
    end
    winLen = 512;
    hop = 256;
    S = stft(noisy, winLen, hop);
    mag = abs(S);
    phase = angle(S);
    noiseMag = noiseEstimate(mag);
    alpha = 2;
    beta = 0.01;
    powerEst = mag .^ 2 - alpha * (noiseMag .^ 2);
    floorPower = beta * (mag .^ 2);
    powerEst = max(powerEst, floorPower);
    enhanced = istft(sqrt(powerEst) .* exp(1i * phase), winLen, hop, numel(noisy));
end
