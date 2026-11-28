function results = validate_model(varargin)
%VALIDATE_MODEL Head-to-head model comparison on held-out speakers.
%   results = VALIDATE_MODEL() evaluates one or more model artifacts
%   on the speaker-held-out slice of the TRAINING corpus (default
%   speakers p236 + p254, whose utterances no candidate in this
%   project is trained on) — never on the test set — and prints a
%   markdown table of SegSNR / LSD / STOI / SI-SDR means next to the
%   noisy reference. This is the instrument for ship decisions: a
%   candidate replaces the shipped model only if it wins here (see
%   reports/ablation.md for the protocol).
%
%   Name-value options:
%     'ModelFiles' - one artifact path or a cell array of paths
%                    (default 'models/denoiseNet.mat')
%     'Speaker'    - validation speaker id or cell array of ids
%                    (default {'p236','p254'})
%     'MaxFiles'   - cap on validation files (default 200)
%     'DataRoot'   - corpus root (default 'data/voicebank')

    p = inputParser;
    p.addParameter('ModelFiles', fullfile('models', 'denoiseNet.mat'));
    p.addParameter('Speaker', {'p236', 'p254'});
    p.addParameter('MaxFiles', 200, @(x) isscalar(x) && x > 0);
    p.addParameter('DataRoot', fullfile('data', 'voicebank'), @ischar);
    p.parse(varargin{:});
    modelFiles = cellstr(p.Results.ModelFiles);
    speakers = cellstr(p.Results.Speaker);

    rootDir = setup();
    cleanDir = fullfile(rootDir, p.Results.DataRoot, 'clean_train');
    noisyDir = fullfile(rootDir, p.Results.DataRoot, 'noisy_train');
    names = {};
    for s = 1:numel(speakers)
        d = dir(fullfile(cleanDir, [speakers{s} '_*.wav']));
        names = [names, {d.name}]; %#ok<AGROW>
    end
    if isempty(names)
        error('validate_model:NoValidationFiles', ...
            'No training files for speaker(s) %s under %s.', ...
            strjoin(speakers, '+'), cleanDir);
    end
    names = sort(names);
    names = names(1:min(p.Results.MaxFiles, numel(names)));
    fprintf('Validation: %d files, speaker(s) %s (training-corpus holdout)\n', ...
        numel(names), strjoin(speakers, '+'));

    models = cell(numel(modelFiles), 1);
    labels = cell(numel(modelFiles), 1);
    for m = 1:numel(modelFiles)
        models{m} = loadDenoiseModel(fullfile(rootDir, modelFiles{m}));
        [~, base, ~] = fileparts(modelFiles{m});
        labels{m} = sprintf('%s (%s)', base, models{m}.kind);
    end

    nRows = numel(models) + 1;
    seg = zeros(nRows, 1); lsd = zeros(nRows, 1);
    stoi = zeros(nRows, 1); sisdr = zeros(nRows, 1);
    for i = 1:numel(names)
        [clean, fs] = audioread(fullfile(cleanDir, names{i}));
        [noisy, ~] = audioread(fullfile(noisyDir, names{i}));
        clean = clean(:, 1); noisy = noisy(:, 1);
        n = min(numel(clean), numel(noisy));
        clean = clean(1:n); noisy = noisy(1:n);
        seg(1) = seg(1) + segSnrDb(clean, noisy);
        lsd(1) = lsd(1) + lsdDb(clean, noisy);
        stoi(1) = stoi(1) + stoiScore(clean, noisy, fs);
        sisdr(1) = sisdr(1) + sisdrDb(clean, noisy);
        for m = 1:numel(models)
            enh = denoiseWithNet(noisy, fs, models{m});
            seg(m + 1) = seg(m + 1) + segSnrDb(clean, enh);
            lsd(m + 1) = lsd(m + 1) + lsdDb(clean, enh);
            stoi(m + 1) = stoi(m + 1) + stoiScore(clean, enh, fs);
            sisdr(m + 1) = sisdr(m + 1) + sisdrDb(clean, enh);
        end
        if mod(i, 50) == 0 || i == numel(names)
            fprintf('  %d / %d files\n', i, numel(names));
        end
    end
    nf = numel(names);
    rowLabels = [{'Noisy (reference)'}, labels{:}]; %#ok<CCAT>
    results = table(string(rowLabels)', seg / nf, lsd / nf, stoi / nf, sisdr / nf, ...
        'VariableNames', {'Model', 'SegSNR_dB', 'LSD_dB', 'STOI', 'SISDR_dB'});
    fprintf('\n| Model | SegSNR (dB) | LSD (dB) | STOI | SI-SDR (dB) |\n');
    fprintf('|---|---|---|---|---|\n');
    for r = 1:nRows
        fprintf('| %s | %.2f | %.2f | %.3f | %.2f |\n', rowLabels{r}, ...
            results.SegSNR_dB(r), results.LSD_dB(r), ...
            results.STOI(r), results.SISDR_dB(r));
    end
end
