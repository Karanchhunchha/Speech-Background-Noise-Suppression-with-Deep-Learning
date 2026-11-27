function results = ablate_mask(varargin)
%ABLATE_MASK Phase-1 ablation: inference-time mask options on validation.
%   results = ABLATE_MASK() evaluates the shipped CNN model with a grid
%   of 'MaskFloor' / 'MaskExponent' settings (see DENOISEWITHNET) on a
%   speaker-held-out slice of the TRAINING corpus — never on the test
%   set — and prints a markdown table for reports/ablation.md.
%
%   The validation speaker's files were seen in training by the shipped
%   model, so absolute values are optimistic; the table is used only to
%   RANK configurations. The chosen configuration is afterwards
%   evaluated exactly once on the full test set.
%
%   Name-value options:
%     'Speaker'  - validation speaker id (default 'p228')
%     'MaxFiles' - cap on validation files (default 100)
%     'DataRoot' - corpus root (default 'data/voicebank')

    p = inputParser;
    p.addParameter('Speaker', 'p228', @(x) ischar(x) || isstring(x));
    p.addParameter('MaxFiles', 100, @(x) isscalar(x) && x > 0);
    p.addParameter('DataRoot', fullfile('data', 'voicebank'), @ischar);
    p.parse(varargin{:});
    speaker = char(p.Results.Speaker);
    maxFiles = p.Results.MaxFiles;
    root = p.Results.DataRoot;

    setup();
    model = loadDenoiseModel(fullfile('models', 'denoiseNet.mat'));

    cleanDir = fullfile(root, 'clean_train');
    noisyDir = fullfile(root, 'noisy_train');
    all = dir(fullfile(cleanDir, [speaker '_*.wav']));
    if isempty(all)
        error('ablate_mask:NoValidationFiles', ...
            'No training files for speaker %s under %s.', speaker, cleanDir);
    end
    names = sort({all.name});
    names = names(1:min(maxFiles, numel(names)));
    fprintf('Validation: %d files, speaker %s (training-corpus holdout)\n', ...
        numel(names), speaker);

    floors = [0, 0.05, 0.1, 0.2];
    exponents = [1, 0.7];
    nCfg = numel(floors) * numel(exponents) + 1;
    labels = cell(nCfg, 1);
    seg = zeros(nCfg, 1); lsd = zeros(nCfg, 1); stoi = zeros(nCfg, 1);
    labels{1} = 'noisy (reference)';
    cfg = 1;
    grids = cell(nCfg, 2);
    grids{1, 1} = NaN; grids{1, 2} = NaN;
    for f = floors
        for e = exponents
            cfg = cfg + 1;
            labels{cfg} = sprintf('floor %.2f, exp %.1f', f, e);
            grids{cfg, 1} = f; grids{cfg, 2} = e;
        end
    end

    for i = 1:numel(names)
        [clean, fs] = audioread(fullfile(cleanDir, names{i}));
        [noisy, ~] = audioread(fullfile(noisyDir, names{i}));
        clean = clean(:, 1); noisy = noisy(:, 1);
        seg(1) = seg(1) + segSnrDb(clean, noisy);
        lsd(1) = lsd(1) + lsdDb(clean, noisy);
        stoi(1) = stoi(1) + stoiScore(clean, noisy, fs);
        for c = 2:nCfg
            enh = denoiseWithNet(noisy, fs, model, ...
                'MaskFloor', grids{c, 1}, 'MaskExponent', grids{c, 2});
            seg(c) = seg(c) + segSnrDb(clean, enh);
            lsd(c) = lsd(c) + lsdDb(clean, enh);
            stoi(c) = stoi(c) + stoiScore(clean, enh, fs);
        end
    end
    n = numel(names);
    results = table(labels, seg / n, lsd / n, stoi / n, ...
        'VariableNames', {'Config', 'SegSNR_dB', 'LSD_dB', 'STOI'});
    disp(results);
end
