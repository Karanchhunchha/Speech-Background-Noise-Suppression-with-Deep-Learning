function enhanced = denoiseWithNet(noisy, fs, model, varargin)
%DENOISEWITHNET Enhance a noisy signal with the trained CNN mask model.
%   enhanced = DENOISEWITHNET(noisy, fs, model) applies the network in
%   MODEL (see LOADDENOISEMODEL) to the noisy signal: the magnitude
%   spectrogram is processed in overlapping segments of model.segLen
%   frames, the predicted ideal-ratio masks are averaged over the
%   overlaps and clipped to [0, 1], and the signal is resynthesised
%   with the noisy phase.
%
%   Name-value options (defaults reproduce the original behaviour
%   exactly; they exist for the Phase-1 ablation in reports/ablation.md):
%     'MaskFloor'    - lower bound applied to the averaged mask before
%                      use (default 0). A floor limits over-suppression
%                      of low-energy speech, a known STOI lever.
%     'MaskExponent' - exponent applied to the mask (default 1).
%                      Values below 1 soften the suppression.
%
%   Models in this project are trained on 16 kHz audio (the training
%   pipeline enforces that rate), so fs must be 16000: applying the
%   mask statistics learned at one rate to audio sampled at another
%   would silently produce wrong results, and is rejected instead.

    p = inputParser;
    p.addParameter('MaskFloor', 0, @(x) isscalar(x) && x >= 0 && x < 1);
    p.addParameter('MaskExponent', 1, @(x) isscalar(x) && x > 0);
    p.parse(varargin{:});
    maskFloor = p.Results.MaskFloor;
    maskExponent = p.Results.MaskExponent;

    if ~(isscalar(fs) && fs > 0)
        error('denoiseWithNet:BadFs', 'fs must be a positive scalar.');
    end
    if fs ~= 16000
        error('denoiseWithNet:BadFs', ...
            'This model was trained at 16 kHz; got fs = %g Hz.', fs);
    end
    winLen = model.winLen;
    hop = model.hop;
    segLen = model.segLen;
    S = stft(noisy, winLen, hop);
    mag = abs(S);
    phase = angle(S);
    numFrames = size(S, 2);
    if numFrames < segLen
        % Very short file: zero-pad the spectrogram to one full segment.
        mag = [mag, zeros(size(mag, 1), segLen - numFrames)];
        phase = [phase, zeros(size(phase, 1), segLen - numFrames)];
        numFrames = segLen;
    end

    logMag = log1p(mag);
    maskSum = zeros(size(mag));
    maskCount = zeros(size(mag));
    segHop = segLen / 2;
    starts = 1:segHop:(numFrames - segLen + 1);
    if isempty(starts)
        starts = 1;
    end
    if starts(end) ~= numFrames - segLen + 1
        starts = [starts, numFrames - segLen + 1];
    end
    for s = starts
        cols = s:(s + segLen - 1);
        patch = (logMag(:, cols) - model.mu) / model.sigma;
        pred = predict(model.net, reshape(single(patch), [size(patch, 1), size(patch, 2), 1, 1]));
        pred = double(reshape(pred, size(patch)));
        maskSum(:, cols) = maskSum(:, cols) + pred;
        maskCount(:, cols) = maskCount(:, cols) + 1;
    end
    mask = maskSum ./ max(maskCount, 1);
    mask = min(max(mask, 0), 1);
    if maskExponent ~= 1
        mask = mask .^ maskExponent;
    end
    if maskFloor > 0
        mask = max(mask, maskFloor);
    end
    enhanced = istft((mask .* mag) .* exp(1i * phase), winLen, hop, numel(noisy));
end
