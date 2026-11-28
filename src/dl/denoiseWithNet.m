function enhanced = denoiseWithNet(noisy, fs, model, varargin)
%DENOISEWITHNET Enhance a noisy signal with the trained CNN mask model.
%   enhanced = DENOISEWITHNET(noisy, fs, model) applies the network in
%   MODEL (see LOADDENOISEMODEL) to the noisy signal: the magnitude
%   spectrogram is processed in overlapping segments of model.segLen
%   frames, the predicted ideal-ratio masks are averaged over the
%   overlaps and clipped to [0, 1], and the signal is resynthesised
%   with the noisy phase.
%
%   Name-value options:
%     'MaskFloor'    - lower bound applied to the averaged mask before
%                      use (default 0.10). A floor limits
%                      over-suppression of low-energy speech.
%     'MaskExponent' - exponent applied to the mask (default 0.7).
%                      Values below 1 soften the suppression.
%   The defaults are the shipped operating point, chosen by the
%   speaker-disjoint held-out ablation in reports/ablation.md
%   (pooled p236+p254 validation, 145 files): with the R1 model this
%   setting keeps segmental SNR within 0.1 dB of the grid maximum
%   while lifting STOI above the noisy reference — the raw-mask
%   setting (floor 0, exponent 1) does not. Pass 'MaskFloor', 0,
%   'MaskExponent', 1 to reproduce the raw predicted mask.
%
%   Models in this project are trained on 16 kHz audio (the training
%   pipeline enforces that rate), so fs must be 16000: applying the
%   mask statistics learned at one rate to audio sampled at another
%   would silently produce wrong results, and is rejected instead.

    p = inputParser;
    p.addParameter('MaskFloor', 0.10, @(x) isscalar(x) && x >= 0 && x < 1);
    p.addParameter('MaskExponent', 0.7, @(x) isscalar(x) && x > 0);
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
    if isfield(model, 'kind') && strcmp(model.kind, 'crm')
        % Complex-mask models correct the phase as well as the
        % magnitude; the MaskFloor/MaskExponent operating point does
        % not apply to them (see BUILDENOISENETV3 / TRAIN_MODEL_V3).
        enhanced = denoiseCRM(noisy, model);
        return;
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

function enhanced = denoiseCRM(noisy, model)
%DENOISECRM Apply a complex-ratio-mask model (kind 'crm').
%   The network consumes overlapping segments of three channels
%   (noisy real / specScale, noisy imaginary / specScale,
%   standardised log-magnitude) and emits a raw two-channel mask;
%   the bounded complex mask maskBound * tanh(output) is averaged
%   over the segment overlaps and applied to the full noisy complex
%   spectrogram, correcting magnitude and phase together.

    winLen = model.winLen;
    hop = model.hop;
    segLen = model.segLen;
    S = stft(noisy, winLen, hop);
    mag = abs(S);
    numFrames = size(S, 2);
    if numFrames < segLen
        S = [S, zeros(size(S, 1), segLen - numFrames)];
        mag = abs(S);
        numFrames = segLen;
    end
    logMag = log1p(mag);
    maskBound = 2;
    if isfield(model, 'maskBound')
        maskBound = model.maskBound;
    end
    mReSum = zeros(size(S));
    mImSum = zeros(size(S));
    maskCount = zeros(size(S));
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
        patch = zeros(size(S, 1), segLen, 3, 'single');
        patch(:, :, 1) = single(real(S(:, cols)) / model.specScale);
        patch(:, :, 2) = single(imag(S(:, cols)) / model.specScale);
        patch(:, :, 3) = single((logMag(:, cols) - model.mu) / model.sigma);
        pred = predict(model.net, reshape(patch, [size(patch, 1), size(patch, 2), 3, 1]));
        pred = double(reshape(pred, [size(S, 1), segLen, 2]));
        mask = maskBound * tanh(pred);
        mReSum(:, cols) = mReSum(:, cols) + mask(:, :, 1);
        mImSum(:, cols) = mImSum(:, cols) + mask(:, :, 2);
        maskCount(:, cols) = maskCount(:, cols) + 1;
    end
    mRe = mReSum ./ max(maskCount, 1);
    mIm = mImSum ./ max(maskCount, 1);
    estSpec = (mRe + 1i * mIm) .* S;
    enhanced = istft(estSpec, winLen, hop, numel(noisy));
end
