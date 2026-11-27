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
%     'Epochs'   training epochs (default 12)
%     'Quick'    true for a short smoke run: 60 utterances, 4 crops,
%                3 epochs (default false)
%     'NumUtterances' / 'CropsPerUtterance'
%                patch configuration for a full run (defaults 300 / 6)

    p = inputParser;
    addParameter(p, 'Epochs', 12);
    addParameter(p, 'Quick', false);
    addParameter(p, 'NumUtterances', 300);
    addParameter(p, 'CropsPerUtterance', 6);
    parse(p, varargin{:});

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
            reuse = saved.patchConfig.NumUtterances == wantUtt && ...
                saved.patchConfig.CropsPerUtterance == wantCrops;
        end
    end
    if ~reuse
        prepare_training_data('NumUtterances', wantUtt, 'CropsPerUtterance', wantCrops);
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
    net = trainNetwork(data.XTrain, data.YTrain, layers, options);

    model = struct('net', net, 'mu', data.mu, 'sigma', data.sigma, ...
        'winLen', data.winLen, 'hop', data.hop, 'segLen', data.segLen);
    modelDir = fullfile(rootDir, 'models');
    if ~exist(modelDir, 'dir'); mkdir(modelDir); end
    save(fullfile(modelDir, 'denoiseNet.mat'), 'model', '-v7.3');
    fprintf('Saved model to %s\n', fullfile(modelDir, 'denoiseNet.mat'));
end
