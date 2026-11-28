function model = train_model_v3(varargin)
%TRAIN_MODEL_V3 Train the complex-mask model (custom training loop).
%   model = TRAIN_MODEL_V3() trains the residual dilated CNN of
%   BUILDENOISENETV3 to predict a complex ratio mask, on the
%   VoiceBank-DEMAND training pairs (download with DOWNLOAD_DATA).
%   Unlike TRAIN_MODEL (trainNetwork + precomputed patches), this
%   trainer uses a custom dlnetwork loop with MODELLOSSCRM and
%   extracts one fresh random 64-frame crop per utterance per epoch
%   on the fly — so the full training corpus (11k+ utterances)
%   provides the diversity, no multi-GB patch file is needed, and
%   every epoch checkpoints to disk: an interrupted run resumes
%   where it stopped instead of starting over.
%
%   The candidate is written to models/denoiseNetV3.mat by default
%   and NEVER overwrites the shipped models/denoiseNet.mat; whether
%   it ships is a separate decision made on speaker-held-out
%   validation (see tools/validate_model.m and reports/ablation.md).
%
%   Name-value options:
%     'Epochs'          epochs to train in total (default 6)
%     'MiniBatchSize'   patches per update (default 32)
%     'LearnRate'       Adam learning rate (default 1e-3)
%     'MaxUtterances'   cap on training utterances (default Inf)
%     'ExcludeSpeakers' speakers held out of training
%                       (default {'p236','p254'} — the validation
%                       speakers of reports/ablation.md)
%     'OutFile'         candidate artifact (default
%                       'models/denoiseNetV3.mat')
%     'CheckpointFile'  resume checkpoint (default
%                       'models/denoiseNetV3_checkpoint.mat')
%     'Resume'          resume from the checkpoint when one exists
%                       (default true)

    p = inputParser;
    addParameter(p, 'Epochs', 6, @(x) isscalar(x) && x >= 1);
    addParameter(p, 'MiniBatchSize', 32, @(x) isscalar(x) && x >= 1);
    addParameter(p, 'LearnRate', 1e-3, @(x) isscalar(x) && x > 0);
    addParameter(p, 'MaxUtterances', Inf, @(x) isscalar(x) && x >= 1);
    addParameter(p, 'ExcludeSpeakers', {'p236', 'p254'});
    addParameter(p, 'OutFile', fullfile('models', 'denoiseNetV3.mat'));
    addParameter(p, 'CheckpointFile', fullfile('models', 'denoiseNetV3_checkpoint.mat'));
    addParameter(p, 'Resume', true, @(x) islogical(x) && isscalar(x));
    parse(p, varargin{:});
    cfg = p.Results;

    rootDir = setup();
    winLen = 512; hop = 256; segLen = 64; numFreqBins = 257;
    pairs = listPairs(fullfile(rootDir, 'data', 'voicebank', 'clean_train'), ...
                      fullfile(rootDir, 'data', 'voicebank', 'noisy_train'));
    if isempty(pairs)
        error('train_model_v3:NoData', ...
            'No training pairs found. Run download_data() first.');
    end
    excluded = string(cellstr(cfg.ExcludeSpeakers));
    keep = true(numel(pairs), 1);
    for i = 1:numel(pairs)
        if any(string(extractBefore(pairs(i).id, '_')) == excluded)
            keep(i) = false;
        end
    end
    pairs = pairs(keep);
    if numel(pairs) > cfg.MaxUtterances
        pairs = pairs(1:cfg.MaxUtterances);
    end
    fprintf('train_model_v3: %d training utterances (excluded speakers: %s)\n', ...
        numel(pairs), strjoin(cellstr(excluded), ', '));

    checkpointPath = fullfile(rootDir, cfg.CheckpointFile);
    if cfg.Resume && exist(checkpointPath, 'file') == 2
        state = load(checkpointPath);
        net = state.net;
        averageGrad = state.averageGrad;
        averageSqGrad = state.averageSqGrad;
        iteration = state.iteration;
        startEpoch = state.epoch + 1;
        mu = state.mu; sigma = state.sigma; specScale = state.specScale;
        fprintf('train_model_v3: resuming from checkpoint at epoch %d (iteration %d)\n', ...
            state.epoch, iteration);
    else
        [mu, sigma, specScale] = computeStats(pairs, winLen, hop);
        net = dlnetwork(buildDenoiseNetV3(numFreqBins, segLen));
        averageGrad = [];
        averageSqGrad = [];
        iteration = 0;
        startEpoch = 1;
        fprintf('train_model_v3: fresh start. mu=%.4f sigma=%.4f specScale=%.4f\n', ...
            mu, sigma, specScale);
    end

    numParams = 0;
    lv = net.Learnables.Value;
    for i = 1:numel(lv)
        numParams = numParams + numel(lv{i});
    end
    fprintf('train_model_v3: %.0f learnable parameters\n', numParams);

    for epoch = startEpoch:cfg.Epochs
        rng(1000 + epoch, 'twister');
        order = randperm(numel(pairs));
        X = zeros(numFreqBins, segLen, 3, cfg.MiniBatchSize, 'single');
        T = zeros(numFreqBins, segLen, 2, cfg.MiniBatchSize, 'single');
        fill = 0;
        lossSum = 0; lossCount = 0; skipped = 0;
        for k = 1:numel(order)
            pr = pairs(order(k));
            [clean, fsC] = audioread(pr.cleanFile);
            [noisy, fsN] = audioread(pr.noisyFile);
            if fsC ~= 16000 || fsN ~= 16000
                skipped = skipped + 1;
                continue;
            end
            n = min(numel(clean), numel(noisy));
            Sc = stft(clean(1:n), winLen, hop);
            Sn = stft(noisy(1:n), winLen, hop);
            numFrames = min(size(Sc, 2), size(Sn, 2));
            if numFrames < segLen
                skipped = skipped + 1;
                continue;
            end
            rng(31 * epoch + k, 'twister');
            start = randi(numFrames - segLen + 1);
            cols = start:(start + segLen - 1);
            fill = fill + 1;
            X(:, :, 1, fill) = single(real(Sn(:, cols)) / specScale);
            X(:, :, 2, fill) = single(imag(Sn(:, cols)) / specScale);
            X(:, :, 3, fill) = single((log1p(abs(Sn(:, cols))) - mu) / sigma);
            T(:, :, 1, fill) = single(real(Sc(:, cols)) / specScale);
            T(:, :, 2, fill) = single(imag(Sc(:, cols)) / specScale);
            if fill == cfg.MiniBatchSize || k == numel(order)
                [loss, gradients, state] = dlfeval(@modelLossCRM, net, ...
                    dlarray(X(:, :, :, 1:fill), 'SSCB'), ...
                    dlarray(T(:, :, :, 1:fill), 'SSCB'));
                % Carry the batch-norm state forward. Without this the
                % saved network keeps the INITIAL trailing averages and
                % predict() explodes at inference (the bug that sank
                % the first v3 run, 28 Nov 2026 — see reports/ablation.md).
                net.State = state;
                gradNorm = sqrt(sum(cellfun( ...
                    @(v) sum(double(extractdata(v)) .^ 2, 'all'), gradients.Value)));
                if gradNorm > 5
                    for gi = 1:numel(gradients.Value)
                        gradients.Value{gi} = gradients.Value{gi} * (5 / gradNorm);
                    end
                end
                iteration = iteration + 1;
                [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, ...
                    averageGrad, averageSqGrad, iteration, cfg.LearnRate);
                lossSum = lossSum + double(extractdata(loss));
                lossCount = lossCount + 1;
                fill = 0;
            end
            if mod(k, 2000) == 0
                fprintf('  epoch %d: %d / %d utterances, mean loss %.4f\n', ...
                    epoch, k, numel(order), lossSum / max(lossCount, 1));
            end
        end
        fprintf('train_model_v3: epoch %d done. mean loss %.4f (%d updates, %d skipped)\n', ...
            epoch, lossSum / max(lossCount, 1), lossCount, skipped);

        model = struct('net', net, 'mu', mu, 'sigma', sigma, ...
            'specScale', specScale, 'maskBound', 2, 'kind', 'crm', ...
            'winLen', winLen, 'hop', hop, 'segLen', segLen);
        trainConfig = cfg;
        save(checkpointPath, 'model', 'net', 'averageGrad', 'averageSqGrad', ...
            'iteration', 'epoch', 'mu', 'sigma', 'specScale', 'trainConfig', '-v7.3');
        save(fullfile(rootDir, cfg.OutFile), 'model', '-v7.3');
        fprintf('train_model_v3: checkpoint + candidate saved (epoch %d)\n', epoch);
    end
    fprintf('train_model_v3: training complete -> %s\n', cfg.OutFile);
end

function [mu, sigma, specScale] = computeStats(pairs, winLen, hop)
%COMPUTEStats Standardisation constants from a sample of utterances.
    sampleIdx = unique(round(linspace(1, numel(pairs), min(400, numel(pairs)))));
    logSum = 0; logSqSum = 0; logCount = 0;
    powSum = 0; powCount = 0;
    for i = 1:numel(sampleIdx)
        [noisy, ~] = audioread(pairs(sampleIdx(i)).noisyFile);
        S = stft(noisy, winLen, hop);
        lm = log1p(abs(S));
        logSum = logSum + sum(lm(:));
        logSqSum = logSqSum + sum(lm(:) .^ 2);
        logCount = logCount + numel(lm);
        powSum = powSum + sum(abs(S(:)) .^ 2);
        powCount = powCount + numel(S);
    end
    mu = logSum / logCount;
    sigma = sqrt(max(logSqSum / logCount - mu ^ 2, 1e-8));
    specScale = sqrt(powSum / powCount);
end
