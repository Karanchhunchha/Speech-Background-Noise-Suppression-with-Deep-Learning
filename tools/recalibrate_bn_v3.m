function model = recalibrate_bn_v3(varargin)
%RECALIBRATE_BN_V3 Recompute the batch-norm state of the v3 candidate.
%   model = RECALIBRATE_BN_V3() repairs the v3 complex-mask candidate
%   trained on 28 Nov 2026. That run's custom training loop dropped
%   the batch-norm state produced by FORWARD inside the loss (the
%   pre-fix MODELLOSSCRM returned only loss and gradients), so the
%   saved dlnetwork kept the INITIAL trailing averages (mean 0,
%   variance 1). Training looked healthy (loss 2.80 -> 0.15) because
%   FORWARD normalises with per-batch statistics, but PREDICT uses
%   the stored trailing averages and exploded: validation SegSNR
%   -9.23 dB against +3.73 dB for the shipped IRM model, with the
%   mask saturated at its bound on ~100% of bins.
%
%   This tool replays training-style batches (same crop construction
%   and speaker exclusions as TRAIN_MODEL_V3) through FORWARD,
%   carrying the state after every batch, so the trailing averages
%   converge to the population statistics the weights were trained
%   against. It writes a NEW artifact (default
%   models/denoiseNetV3_recal.mat); the original candidate and the
%   shipped models/denoiseNet.mat are never touched. Whether the
%   recalibrated candidate ships is decided by TOOLS/VALIDATE_MODEL
%   on the held-out speakers, exactly like every other candidate.
%
%   Name-value options:
%     'InFile'          candidate to repair
%                       (default 'models/denoiseNetV3.mat')
%     'OutFile'         repaired artifact
%                       (default 'models/denoiseNetV3_recal.mat')
%     'Passes'          full passes over the training pairs (default 2)
%     'MiniBatchSize'   patches per forward call (default 32)
%     'ExcludeSpeakers' speakers held out, as in training
%                       (default {'p236','p254'})

    p = inputParser;
    addParameter(p, 'InFile', fullfile('models', 'denoiseNetV3.mat'));
    addParameter(p, 'OutFile', fullfile('models', 'denoiseNetV3_recal.mat'));
    addParameter(p, 'Passes', 2, @(x) isscalar(x) && x >= 1);
    addParameter(p, 'MiniBatchSize', 32, @(x) isscalar(x) && x >= 1);
    addParameter(p, 'ExcludeSpeakers', {'p236', 'p254'});
    parse(p, varargin{:});
    cfg = p.Results;

    rootDir = setup();
    inPath = fullfile(rootDir, cfg.InFile);
    if exist(inPath, 'file') ~= 2
        error('recalibrate_bn_v3:NoCandidate', ...
            'Candidate not found: %s', cfg.InFile);
    end
    loaded = load(inPath, 'model');
    model = loaded.model;
    if ~isfield(model, 'kind') || ~strcmp(model.kind, 'crm')
        error('recalibrate_bn_v3:NotCrm', ...
            '%s is not a complex-mask (crm) artifact.', cfg.InFile);
    end
    net = model.net;
    mu = model.mu; sigma = model.sigma; specScale = model.specScale;
    winLen = model.winLen; hop = model.hop; segLen = model.segLen;
    numFreqBins = 257;

    pairs = listPairs(fullfile(rootDir, 'data', 'voicebank', 'clean_train'), ...
                      fullfile(rootDir, 'data', 'voicebank', 'noisy_train'));
    excluded = string(cellstr(cfg.ExcludeSpeakers));
    keep = true(numel(pairs), 1);
    for i = 1:numel(pairs)
        if any(string(extractBefore(pairs(i).id, '_')) == excluded)
            keep(i) = false;
        end
    end
    pairs = pairs(keep);
    fprintf('recalibrate_bn_v3: %d training utterances, %d passes\n', ...
        numel(pairs), cfg.Passes);

    updates = 0;
    for pass = 1:cfg.Passes
        rng(7000 + pass, 'twister');
        order = randperm(numel(pairs));
        X = zeros(numFreqBins, segLen, 3, cfg.MiniBatchSize, 'single');
        fill = 0;
        for k = 1:numel(order)
            pr = pairs(order(k));
            [noisy, fsN] = audioread(pr.noisyFile);
            if fsN ~= 16000
                continue;
            end
            Sn = stft(noisy, winLen, hop);
            numFrames = size(Sn, 2);
            if numFrames < segLen
                continue;
            end
            start = randi(numFrames - segLen + 1);
            cols = start:(start + segLen - 1);
            fill = fill + 1;
            X(:, :, 1, fill) = single(real(Sn(:, cols)) / specScale);
            X(:, :, 2, fill) = single(imag(Sn(:, cols)) / specScale);
            X(:, :, 3, fill) = single((log1p(abs(Sn(:, cols))) - mu) / sigma);
            if fill == cfg.MiniBatchSize || k == numel(order)
                [~, state] = forward(net, dlarray(X(:, :, :, 1:fill), 'SSCB'));
                net.State = state;
                updates = updates + 1;
                fill = 0;
            end
        end
        fprintf('recalibrate_bn_v3: pass %d done (%d state updates total)\n', ...
            pass, updates);
    end

    model.net = net;
    outPath = fullfile(rootDir, cfg.OutFile);
    save(outPath, 'model', '-v7.3');
    fprintf('recalibrate_bn_v3: recalibrated candidate -> %s\n', cfg.OutFile);
end
