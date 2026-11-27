function summary = evaluate_models(varargin)
%EVALUATE_MODELS Objective evaluation of all enhancers on the test set.
%   summary = EVALUATE_MODELS() evaluates the noisy reference, spectral
%   subtraction, the decision-directed Wiener filter and (when a
%   trained model exists in models/denoiseNet.mat) the CNN mask model
%   on the VoiceBank-DEMAND test pairs in data/voicebank (falling back
%   to the committed sample in data/sample when the full test set has
%   not been downloaded). Metrics per file: segmental SNR (dB),
%   log-spectral distance (dB) and STOI. Results are written to
%   reports/metrics_per_file.csv, reports/metrics_summary.csv and
%   reports/EVALUATION.md, and the summary table is returned.
%
%   Name-value options:
%     'MaxFiles'  evaluate at most this many pairs (default: all)

    p = inputParser;
    addParameter(p, 'MaxFiles', Inf);
    parse(p, varargin{:});

    rootDir = setup();
    testClean = fullfile(rootDir, 'data', 'voicebank', 'clean_test');
    testNoisy = fullfile(rootDir, 'data', 'voicebank', 'noisy_test');
    pairs = listPairs(testClean, testNoisy);
    dataNote = 'VoiceBank-DEMAND standard test set (data/voicebank)';
    if isempty(pairs)
        pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                          fullfile(rootDir, 'data', 'sample', 'noisy'));
        dataNote = 'committed sample only (data/sample) - run download_data() for the full test set';
    end
    if isempty(pairs)
        error('evaluate_models:NoData', 'No test pairs found.');
    end
    pairs = pairs(1:min(numel(pairs), p.Results.MaxFiles));
    fprintf('Evaluating %d pairs from %s\n', numel(pairs), dataNote);

    modelFile = fullfile(rootDir, 'models', 'denoiseNet.mat');
    haveModel = exist(modelFile, 'file') == 2;
    if haveModel
        model = loadDenoiseModel(modelFile);
        methodNames = {'Noisy', 'SpectralSubtraction', 'Wiener', 'CNN'};
    else
        methodNames = {'Noisy', 'SpectralSubtraction', 'Wiener'};
        fprintf('No trained model found; skipping the CNN method.\n');
    end
    numMethods = numel(methodNames);

    seg = zeros(numel(pairs), numMethods);
    lsd = zeros(numel(pairs), numMethods);
    stoi = zeros(numel(pairs), numMethods);
    for i = 1:numel(pairs)
        [clean, fs] = audioread(pairs(i).cleanFile);
        [noisy, ~] = audioread(pairs(i).noisyFile);
        n = min(numel(clean), numel(noisy));
        clean = clean(1:n);
        noisy = noisy(1:n);
        enhanced = cell(1, numMethods);
        enhanced{1} = noisy;
        enhanced{2} = spectralSubtraction(noisy, fs);
        enhanced{3} = wienerEnhance(noisy, fs);
        if haveModel
            enhanced{4} = denoiseWithNet(noisy, fs, model);
        end
        for m = 1:numMethods
            seg(i, m) = segSnrDb(clean, enhanced{m});
            lsd(i, m) = lsdDb(clean, enhanced{m});
            stoi(i, m) = stoiScore(clean, enhanced{m}, fs);
        end
        if mod(i, 100) == 0 || i == numel(pairs)
            fprintf('  %d / %d files\n', i, numel(pairs));
        end
    end

    reportDir = fullfile(rootDir, 'reports');
    if ~exist(reportDir, 'dir'); mkdir(reportDir); end

    perFile = table();
    perFile.File = string({pairs.id})';
    for m = 1:numMethods
        perFile.(['SegSNR_' methodNames{m}]) = seg(:, m);
        perFile.(['LSD_' methodNames{m}]) = lsd(:, m);
        perFile.(['STOI_' methodNames{m}]) = stoi(:, m);
    end
    writetable(perFile, fullfile(reportDir, 'metrics_per_file.csv'));

    summary = table();
    summary.Method = string(methodNames)';
    summary.SegSNR_dB = mean(seg, 1)';
    summary.SegSNR_Std = std(seg, 0, 1)';
    summary.LSD_dB = mean(lsd, 1)';
    summary.STOI = mean(stoi, 1)';
    summary.STOI_Std = std(stoi, 0, 1)';
    writetable(summary, fullfile(reportDir, 'metrics_summary.csv'));

    fid = fopen(fullfile(reportDir, 'EVALUATION.md'), 'w');
    fprintf(fid, '# Objective evaluation\n\n');
    fprintf(fid, 'Data: %s. Files evaluated: %d. Metrics are means over files ', dataNote, numel(pairs));
    fprintf(fid, '(segmental SNR and STOI: higher is better; LSD: lower is better).\n\n');
    fprintf(fid, '| Method | Segmental SNR (dB) | LSD (dB) | STOI |\n');
    fprintf(fid, '|---|---|---|---|\n');
    for m = 1:numMethods
        fprintf(fid, '| %s | %.2f | %.2f | %.3f |\n', methodNames{m}, ...
            summary.SegSNR_dB(m), summary.LSD_dB(m), summary.STOI(m));
    end
    fprintf(fid, '\nSegmental-SNR improvement of the CNN over the noisy input: %.2f dB.\n', ...
        summary.SegSNR_dB(end) - summary.SegSNR_dB(1));
    fclose(fid);

    disp(summary);
end
