function results = pesq_native(varargin)
%PESQ_NATIVE PESQ scores from a MATLAB `pesq` function, if one exists.
%   results = PESQ_NATIVE() would score the noisy input and every
%   enhancer on the VoiceBank-DEMAND test pairs with a `pesq`
%   function (ITU-T P.862) and write a long-format CSV with one row
%   per file and method.
%
%   VERIFIED FINDING (28 Nov 2026, MATLAB Online R2026b, Audio
%   Toolbox licensed and resolving - audioDatastore works): MATLAB
%   ships NO `pesq` function at all. `which('pesq','-all')` reports
%   "not found" and `exist('pesq')` is 0. PESQ's reference
%   implementation is not part of MATLAB or the Audio Toolbox, so
%   in-MATLAB PESQ is not possible; this project's PESQ numbers
%   come from the offline route (tools/python_metrics.py, see
%   reports/pesq_subset.md). This function is kept as the record of
%   that check: it detects the absence, says so, and returns
%   without writing anything - scores are never approximated or
%   invented. If a future MATLAB release ships `pesq`, this tool
%   will start working unchanged.
%
%   The core pipeline needs only MATLAB + Deep Learning Toolbox
%   (see reports/required_products.txt) and never calls this
%   function.
%
%   Name-value options:
%     'MaxFiles'    evaluate at most this many pairs (default: all)
%     'StartIndex'  start at this pair index (default 1); together
%                   with 'MaxFiles' this allows chunked runs
%     'OutFile'     CSV to write, relative to the project root
%                   (default 'reports/pesq_native_part.csv'); use a
%                   distinct name per chunk, then merge the chunks
%
%   Output CSV columns: file,method,pesq
%   (method in {noisy, spectralsubtraction, wiener, cnn}).

    p = inputParser;
    addParameter(p, 'MaxFiles', Inf);
    addParameter(p, 'StartIndex', 1);
    addParameter(p, 'OutFile', fullfile('reports', 'pesq_native_part.csv'));
    parse(p, varargin{:});

    results = table();
    if isempty(which('pesq'))
        fprintf(['pesq_native: no pesq function exists in this MATLAB ' ...
            'installation (verified: which/exist find nothing, even ' ...
            'with the Audio Toolbox licensed - MATLAB ships no PESQ ' ...
            'implementation).\nNo file written, no scores produced. ' ...
            '(The offline route is tools/python_metrics.py.)\n']);
        return;
    end

    rootDir = setup();
    testClean = fullfile(rootDir, 'data', 'voicebank', 'clean_test');
    testNoisy = fullfile(rootDir, 'data', 'voicebank', 'noisy_test');
    pairs = listPairs(testClean, testNoisy);
    if isempty(pairs)
        pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                          fullfile(rootDir, 'data', 'sample', 'noisy'));
    end
    if isempty(pairs)
        error('pesq_native:NoData', 'No test pairs found.');
    end
    firstIdx = min(p.Results.StartIndex, numel(pairs));
    lastIdx = min(numel(pairs), firstIdx + p.Results.MaxFiles - 1);
    pairs = pairs(firstIdx:lastIdx);
    fprintf('pesq_native: scoring %d pairs (Audio Toolbox pesq)\n', numel(pairs));

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

    scores = nan(numel(pairs), numMethods);
    numErrors = 0;
    firstError = '';
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
            deg = enhanced{m};
            deg = deg(1:min(numel(deg), n));
            try
                scores(i, m) = pesq(clean, deg, fs);
            catch ME
                numErrors = numErrors + 1;
                if isempty(firstError)
                    firstError = ME.message;
                end
            end
        end
        if mod(i, 50) == 0 || i == numel(pairs)
            fprintf('  %d / %d files\n', i, numel(pairs));
        end
    end

    fileCol = strings(numel(pairs) * numMethods, 1);
    methodCol = strings(numel(pairs) * numMethods, 1);
    pesqCol = nan(numel(pairs) * numMethods, 1);
    row = 0;
    for i = 1:numel(pairs)
        for m = 1:numMethods
            row = row + 1;
            fileCol(row) = string(pairs(i).id);
            methodCol(row) = lower(string(methodNames{m}));
            pesqCol(row) = scores(i, m);
        end
    end
    results = table(fileCol, methodCol, pesqCol, ...
        'VariableNames', {'file', 'method', 'pesq'});

    outPath = fullfile(rootDir, p.Results.OutFile);
    writetable(results, outPath);
    fprintf('pesq_native: wrote %s (%d scores)\n', p.Results.OutFile, height(results));
    for m = 1:numMethods
        col = scores(:, m);
        fprintf('  %-20s PESQ mean %.3f (N=%d finite)\n', ...
            lower(methodNames{m}), mean(col, 'omitnan'), sum(isfinite(col)));
    end
    if numErrors > 0
        fprintf('pesq_native: %d pesq calls errored; first error: %s\n', ...
            numErrors, firstError);
    end
end
