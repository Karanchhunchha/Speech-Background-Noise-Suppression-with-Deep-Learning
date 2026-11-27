function stats = prepare_training_data(varargin)
%PREPARE_TRAINING_DATA Build spectrogram training patches from VoiceBank pairs.
%   stats = PREPARE_TRAINING_DATA() reads clean/noisy training pairs
%   from data/voicebank (run DOWNLOAD_DATA first), computes STFT
%   magnitude spectrograms, and crops random segments of 64 frames.
%   Inputs are log1p magnitudes standardised by the training-set mean
%   and standard deviation; targets are ideal ratio masks computed
%   from the clean and noise spectrograms (the noise signal is the
%   difference between the noisy and clean waveforms, which the paired
%   corpus makes available exactly). The result is saved to
%   data/training_patches.mat.
%
%   Name-value options:
%     'NumUtterances'  pairs to use (default 300)
%     'CropsPerUtterance' random 64-frame crops per pair (default 6)
%     'Seed'           RNG seed for reproducibility (default 42)

    p = inputParser;
    addParameter(p, 'NumUtterances', 300);
    addParameter(p, 'CropsPerUtterance', 6);
    addParameter(p, 'Seed', 42);
    parse(p, varargin{:});
    numUtt = p.Results.NumUtterances;
    cropsPerUtt = p.Results.CropsPerUtterance;

    rootDir = setup();
    winLen = 512;
    hop = 256;
    segLen = 64;
    numFreq = winLen / 2 + 1;

    cleanDir = fullfile(rootDir, 'data', 'voicebank', 'clean_train');
    noisyDir = fullfile(rootDir, 'data', 'voicebank', 'noisy_train');
    pairs = listPairs(cleanDir, noisyDir);
    if isempty(pairs)
        error('prepare_training_data:NoData', ...
            'No training pairs found in %s. Run download_data() first.', cleanDir);
    end
    numUtt = min(numUtt, numel(pairs));
    pairs = pairs(1:numUtt);

    rng(p.Results.Seed);
    total = numUtt * cropsPerUtt;
    XTrain = zeros(numFreq, segLen, 1, total, 'single');
    YTrain = zeros(numFreq, segLen, 1, total, 'single');
    logSum = 0;
    logSqSum = 0;
    logCount = 0;
    out = 0;
    for i = 1:numUtt
        [clean, fsC] = audioread(pairs(i).cleanFile);
        [noisy, fsN] = audioread(pairs(i).noisyFile);
        if fsC ~= 16000 || fsN ~= 16000
            error('prepare_training_data:BadRate', ...
                'Expected 16 kHz audio, got %d/%d Hz for %s.', fsC, fsN, pairs(i).id);
        end
        n = min(numel(clean), numel(noisy));
        clean = clean(1:n);
        noisy = noisy(1:n);
        noise = noisy - clean;
        magClean = abs(stft(clean, winLen, hop));
        magNoisy = abs(stft(noisy, winLen, hop));
        magNoise = abs(stft(noise, winLen, hop));
        logMag = log1p(magNoisy);
        irm = computeIRM(magClean, magNoise);
        numFrames = size(logMag, 2);
        if numFrames < segLen
            continue;
        end
        logSum = logSum + sum(logMag(:));
        logSqSum = logSqSum + sum(logMag(:) .^ 2);
        logCount = logCount + numel(logMag);
        for c = 1:cropsPerUtt
            start = randi(numFrames - segLen + 1);
            cols = start:(start + segLen - 1);
            out = out + 1;
            XTrain(:, :, 1, out) = single(logMag(:, cols));
            YTrain(:, :, 1, out) = single(irm(:, cols));
        end
    end
    XTrain = XTrain(:, :, :, 1:out);
    YTrain = YTrain(:, :, :, 1:out);
    mu = logSum / logCount;
    sigma = sqrt(logSqSum / logCount - mu ^ 2);
    XTrain = (XTrain - single(mu)) ./ single(sigma);

    stats = struct('NumPatches', out, 'NumUtterances', numUtt, 'Mu', mu, 'Sigma', sigma);
    save(fullfile(rootDir, 'data', 'training_patches.mat'), ...
        'XTrain', 'YTrain', 'mu', 'sigma', 'winLen', 'hop', 'segLen', '-v7.3');
    fprintf('Prepared %d patches from %d utterances (mu=%.4f, sigma=%.4f).\n', ...
        out, numUtt, mu, sigma);
end
