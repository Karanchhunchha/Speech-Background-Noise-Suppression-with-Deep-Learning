function model = train_model(varargin)
%TRAIN_MODEL Train the CNN mask-estimation network.
%   model = TRAIN_MODEL() trains the network from BUILDENOISENET on the
%   patches built by PREPARE_TRAINING_DATA (building them first if the
%   patch file is missing) and saves the artifact to
%   models/denoiseNet.mat: the trained network plus the log-magnitude
%   standardisation constants and STFT configuration that
%   DENOISEWITHNET needs.
%
%   Name-value options:
%     'Epochs'   training epochs (default 24)
%     'Quick'    true for a short smoke run: 60 utterances, 4 crops,
%                3 epochs (default false)
%     'NumUtterances' / 'CropsPerUtterance'
%                patch configuration for a full run (defaults 800 / 6)
%     'Seed'     RNG seed for the training shuffle (default 42), so a
%                repeated run on the same patches is reproducible
%     'ExcludeSpeakers' speakers to hold out of training (default {};
%                forwarded to PREPARE_TRAINING_DATA)
%     'ModelFile' file name under models/ to save to (default
%                'denoiseNet.mat' — the shipped artifact; Phase-1
%                candidates use their own names so the shipped model
%                is never overwritten by an experiment)
%   The defaults are the shipped model's configuration: 800 utterances,
%   4,800 patches, 24 epochs (about 55 minutes on CPU in MATLAB Online).

    p = inputParser;
    addParameter(p, 'Epochs', 24);
    addParameter(p, 'Quick', false);
    addParameter(p, 'NumUtterances', 800);
    addParameter(p, 'CropsPerUtterance', 6);
    addParameter(p, 'Seed', 42);
    addParameter(p, 'ExcludeSpeakers', {});
    addParameter(p, 'ModelFile', 'denoiseNet.mat');
    parse(p, varargin{:});
    excludeSpeakers = sort(cellstr(p.Results.ExcludeSpeakers));

    rootDir = setup();
    patchFile = fullfile(rootDir, 'data', 'training_patches.mat');
    % Reuse the patch file only when it was built with the configuration
    % this run needs — a leftover quick-mode patch file must never
    % silently shrink a full training run (or the reverse).
    if p.Results.Quick
        wantUtt = 60;
        wantCrops = 4;
    else
        wantUtt = p.Results.NumUtterances;
        wantCrops = p.Results.CropsPerUtterance;
    end
    reuse = false;
    if exist(patchFile, 'file')
        info = whos('-file', patchFile, 'patchConfig');
        if ~isempty(info)
            saved = load(patchFile, 'patchConfig');
            % Compare against the REQUESTED utterance count when the
            % patch file records it: the stored NumUtterances is the
            % actual (possibly corpus-clamped) count, and comparing
            % against that would rebuild the patches on every call
            % whenever the corpus holds fewer pairs than requested.
            if isfield(saved.patchConfig, 'RequestedUtterances')
                savedWant = saved.patchConfig.RequestedUtterances;
            else
                savedWant = saved.patchConfig.NumUtterances;
            end
            savedExclude = {};
            if isfield(saved.patchConfig, 'ExcludeSpeakers')
                savedExclude = sort(cellstr(saved.patchConfig.ExcludeSpeakers));
            end
            reuse = savedWant == wantUtt && ...
                saved.patchConfig.CropsPerUtterance == wantCrops && ...
                isequal(savedExclude, excludeSpeakers);
        end
    end
    if ~reuse
        prepare_training_data('NumUtterances', wantUtt, 'CropsPerUtterance', wantCrops, ...
            'ExcludeSpeakers', excludeSpeakers);
    end
    data = load(patchFile, 'XTrain', 'YTrain', 'mu', 'sigma', 'winLen', 'hop', 'segLen');

    epochs = p.Results.Epochs;
    if p.Results.Quick
        epochs = 3;
    end
    layers = buildDenoiseNet(data.winLen / 2 + 1, data.segLen);
    options = trainingOptions('adam', ...
        'MaxEpochs', epochs, ...
        'MiniBatchSize', 32, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 8, ...
        'LearnRateDropFactor', 0.5, ...
        'Shuffle', 'every-epoch', ...
        'ExecutionEnvironment', 'cpu', ...
        'Verbose', true, ...
        'Plots', 'none');
    fprintf('Training on %d patches for %d epochs...\n', size(data.XTrain, 4), epochs);
    rng(p.Results.Seed);
    net = trainNetwork(data.XTrain, data.YTrain, layers, options);

    model = struct('net', net, 'mu', data.mu, 'sigma', data.sigma, ...
        'winLen', data.winLen, 'hop', data.hop, 'segLen', data.segLen, ...
        'fs', 16000);
    modelDir = fullfile(rootDir, 'models');
    if ~exist(modelDir, 'dir'); mkdir(modelDir); end
    modelPath = fullfile(modelDir, p.Results.ModelFile);
    save(modelPath, 'model', '-v7.3');
    fprintf('Saved model to %s\n', modelPath);
end
