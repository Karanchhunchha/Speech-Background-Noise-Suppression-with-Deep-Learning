function mask = computeIRM(cleanMag, noiseMag)
%COMPUTEIRM Ideal ratio mask (IRM) from clean and noise magnitudes.
%   mask = COMPUTEIRM(cleanMag, noiseMag) returns the ideal ratio mask
%   with exponent 0.5, (|S|^2 / (|S|^2 + |N|^2))^0.5, the training
%   target for the denoising network. Values lie in [0, 1].

    ratio = (cleanMag .^ 2) ./ (cleanMag .^ 2 + noiseMag .^ 2 + eps);
    mask = sqrt(ratio);
    mask = min(max(mask, 0), 1);
end
